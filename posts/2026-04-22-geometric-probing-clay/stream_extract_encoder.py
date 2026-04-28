"""Stream-extract Clay v1.5 encoder weights from the 5 GB checkpoint without
loading the full state dict into RAM.

We only have ~4 GB of RAM in this sandbox; the full ckpt is 5.16 GB. But the
Clay encoder (patch_size=8, dim=1024, depth=24, heads=16) is ~300 M params
= ~1.2 GB at float32 — fits fine.

Strategy:
  1. PyTorch ckpts are ZIPs. Each tensor's storage is `archive/data/<N>`.
  2. `archive/data.pkl` is a small Python pickle of the state_dict tree.
     Tensors are recorded via `persistent_id` tuples that name their storage
     file, dtype, and numel.
  3. We unpickle data.pkl with a custom Unpickler whose `persistent_load`
     does NOT read the storage file. Instead it returns a tiny "LazyTensor"
     object recording (dtype, storage_key, numel).
  4. Tensor reconstruction is also overridden via pickle dispatch so
     `torch._utils._rebuild_tensor_v2` returns a LazyTensor.
  5. After unpickling, we filter to `model.encoder.*` keys and materialize
     just those by reading the specific zip entries with numpy.frombuffer.

Output: a torch.save'd small state_dict containing only the encoder weights
       (~1.2 GB). Written to `<repo>/model/checkpoints/clay-v1.5_encoder.pt`
       by default; override with the `CLAY_ENCODER_PT` environment variable.
"""
from __future__ import annotations

import io
import pickle
import struct
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

import os

# CLAY_CKPT env var overrides; otherwise resolve relative to this devlogs repo.
HERE = Path(__file__).resolve().parent
_REPO_ROOT = HERE.parent.parent.parent
CKPT = Path(os.environ.get("CLAY_CKPT") or _REPO_ROOT / "model/checkpoints/clay-v1.5.ckpt")
OUT  = Path(os.environ.get("CLAY_ENCODER_PT") or _REPO_ROOT / "model/checkpoints/clay-v1.5_encoder.pt")
OUT.parent.mkdir(parents=True, exist_ok=True)

# dtype string (PyTorch storage class) → (numpy dtype, bytes/element)
DTYPE_MAP = {
    "FloatStorage":  (np.float32, 4),
    "DoubleStorage": (np.float64, 8),
    "HalfStorage":   (np.float16, 2),
    "BFloat16Storage": (np.uint16, 2),  # will reinterpret later
    "LongStorage":   (np.int64, 8),
    "IntStorage":    (np.int32, 4),
    "ShortStorage":  (np.int16, 2),
    "CharStorage":   (np.int8, 1),
    "ByteStorage":   (np.uint8, 1),
    "BoolStorage":   (np.bool_, 1),
}


class LazyStorage:
    """Placeholder for a storage; records what to read later."""
    __slots__ = ("storage_type", "key", "location", "numel")

    def __init__(self, storage_type, key, location, numel):
        self.storage_type = storage_type
        self.key = key
        self.location = location
        self.numel = numel

    def __repr__(self):
        return f"LazyStorage({self.storage_type}, key={self.key}, n={self.numel})"


class LazyTensor:
    __slots__ = ("storage", "storage_offset", "size", "stride", "requires_grad", "dtype")

    def __init__(self, storage, storage_offset, size, stride, dtype=None):
        self.storage = storage
        self.storage_offset = storage_offset
        self.size = tuple(size)
        self.stride = tuple(stride)
        self.dtype = dtype

    def __repr__(self):
        return f"LazyTensor(size={self.size}, storage={self.storage})"


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad,
                      backward_hooks, metadata=None):
    """Stand-in for torch._utils._rebuild_tensor_v2 that returns a LazyTensor."""
    return LazyTensor(storage, storage_offset, size, stride)


def _rebuild_parameter(data, requires_grad, backward_hooks):
    return data  # treat Parameters as plain tensors


class ClayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Route reconstruction functions to our lazy versions
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict
        # For anything else (e.g. torch.FloatStorage class object referenced in pid),
        # return a thin stub carrying the class name.
        class _Stub:
            __name__ = name
            __qualname__ = name
        _Stub.__name__ = name
        _Stub.__qualname__ = name
        return _Stub

    def persistent_load(self, pid):
        # pid = ('storage', storage_type, key, location, numel)
        typename = pid[0]
        if typename == "storage":
            storage_type, key, location, numel = pid[1], pid[2], pid[3], pid[4]
            return LazyStorage(storage_type.__name__, str(key), location, numel)
        raise RuntimeError(f"Unknown pid: {pid!r}")


def extract_encoder(ckpt_path: Path, out_path: Path,
                    prefix: str = "model.encoder."):
    t0 = time.time()
    with zipfile.ZipFile(ckpt_path) as z:
        pkl_bytes = z.read("archive/data.pkl")
        print(f"[{time.time()-t0:.1f}s] read data.pkl ({len(pkl_bytes)} B)")

        unpickler = ClayUnpickler(io.BytesIO(pkl_bytes))
        obj = unpickler.load()
        print(f"[{time.time()-t0:.1f}s] unpickled tree")

        # obj may itself be a state_dict, or a nested dict (Lightning ckpt).
        # Traverse to find the inner state_dict.
        if isinstance(obj, dict):
            if "state_dict" in obj:
                sd = obj["state_dict"]
            else:
                sd = obj
        else:
            sd = obj
        all_keys = list(sd.keys())
        enc_keys = [k for k in all_keys if k.startswith(prefix)]
        print(f"[{time.time()-t0:.1f}s] total keys: {len(all_keys)}, "
              f"encoder keys: {len(enc_keys)}")

        out = {}
        for k in enc_keys:
            lt = sd[k]
            if not isinstance(lt, LazyTensor):
                # some non-tensor leaf
                out[k.removeprefix(prefix)] = lt
                continue
            storage_name = f"archive/data/{lt.storage.key}"
            np_dtype, elem_sz = DTYPE_MAP[lt.storage.storage_type]
            raw = z.read(storage_name)
            arr = np.frombuffer(raw, dtype=np_dtype, count=lt.storage.numel)
            # Apply storage_offset + shape + stride
            # Most tensors have contiguous storage with stride matching shape.
            # For simplicity, assume contiguous and reshape.
            n = int(np.prod(lt.size))
            arr_slice = arr[lt.storage_offset: lt.storage_offset + n]
            # Handle BFloat16 (stored as uint16, needs reinterpret)
            if lt.storage.storage_type == "BFloat16Storage":
                t = torch.from_numpy(arr_slice.copy()).view(torch.bfloat16)
            else:
                t = torch.from_numpy(arr_slice.copy()).clone()
            t = t.reshape(lt.size)
            # Strip the "model.encoder." prefix so this state dict can be
            # loaded directly into an openclay EmbeddingEncoder().
            short_key = k.removeprefix(prefix)
            out[short_key] = t
        print(f"[{time.time()-t0:.1f}s] materialized {len(out)} tensors")

    # Save small state dict
    torch.save(out, out_path)
    sz = out_path.stat().st_size / 1e9
    print(f"[{time.time()-t0:.1f}s] saved → {out_path} ({sz:.2f} GB)")


if __name__ == "__main__":
    extract_encoder(CKPT, OUT)
