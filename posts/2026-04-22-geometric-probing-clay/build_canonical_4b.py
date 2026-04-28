"""build_canonical_4b.py — extract canonical boat cutouts from 4-band NAIP.

Replaces build_canonical.py's RGB+JPG-source path. Reads the 4-band npz
sources produced by fetch_4band_sources.py, masks each boat using the
LAB chrominance signal (RGB-derived; identical to the v3 mask method),
straightens the hull to horizontal via minAreaRect, then saves a
5-channel cutout (R, G, B, NIR, alpha) so downstream compositing flows
real NIR through.

Outputs:
  assets/canonical_boat1_4b.npz  (5×h×w uint8)
  assets/canonical_boat2_4b.npz
  assets/canonical_boat1_4b_preview.png  (RGBA preview)
  assets/canonical_boat2_4b_preview.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import (binary_closing, binary_fill_holes, gaussian_filter,
                            label as cc_label, binary_erosion, binary_dilation)
from skimage.color import rgb2lab
from skimage.transform import rotate as sk_rotate

import cv2

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

# Mask threshold percentile of (chroma + brightness) distance from water.
# Higher = stricter mask. Tuned per boat: boat1's hull is greenish like the
# water (low chroma), so we lean on brightness; boat2 is reddish (strong chroma).
BRIGHT_PCT = {"boat1": 88.0, "boat2": 90.0}

# Manual rotation nudge applied AFTER the two-pass PCA. POSITIVE = CCW
# visually (sk_rotate convention). boat2 needs +2° CCW total
# (3° - 1° refinement) to fully horizontalize after PCA's bow-asymmetry bias.
EXTRA_ROT = {"boat1": 0.0, "boat2": 2.0}

# Post-process knobs from canonical_meta.json (v3 pipeline)
# Boat1: dilate=1, convex_hull, erode=10, symmetry intersection, +-2°.


def load_4b(name: str) -> np.ndarray:
    return np.load(ASSETS / f"{name}_native_4b.npz")["img"]


def water_lab(rgb: np.ndarray) -> np.ndarray:
    """LAB a/b channels — boats are chromatically distinct from water."""
    lab = rgb2lab(rgb / 255.0)
    return lab[..., 1:3]   # a, b (drop L)


def boat_mask(full4: np.ndarray, water4_ref: np.ndarray,
               nir_thr_factor: float = 1.6) -> np.ndarray:
    """Mask boat pixels using the NIR channel — water has very low NIR
    (~16) while ship hulls and decks reflect strongly. Threshold = water_NIR
    + nir_thr_factor × water_NIR_std. Falls back to LAB distance if NIR is
    too noisy.
    full4 / water4_ref are (4, H, W) uint8 arrays (R, G, B, NIR)."""
    nir = full4[3].astype(np.float32)
    water_nir = water4_ref[3].astype(np.float32)
    nir_mean = water_nir.mean()
    nir_std = water_nir.std()
    thr = nir_mean + nir_thr_factor * max(nir_std, 4.0)
    mask = nir > thr
    print(f"    NIR threshold: water mean={nir_mean:.1f} std={nir_std:.1f} "
          f"thr={thr:.1f} → {mask.sum()} px")
    mask = binary_closing(mask, iterations=2)
    mask = binary_fill_holes(mask)
    # Largest connected component
    lab_, n = cc_label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(lab_.ravel())
    sizes[0] = 0
    keep_id = sizes.argmax()
    mask = (lab_ == keep_id)
    mask = binary_fill_holes(mask)
    mask = binary_closing(mask, iterations=2)
    return mask


def _pca_angle(mask: np.ndarray) -> float:
    """Math-convention angle (degrees, atan2(vy, vx)) of the mask's
    principal axis. NaN if mask too small."""
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return float("nan")
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    pts -= pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, -1]
    return float(np.degrees(np.arctan2(principal[1], principal[0])))


def _apply_rotation(rgbn: np.ndarray, mask: np.ndarray, deg: float) -> tuple[np.ndarray, np.ndarray]:
    rotated_chans = []
    for c in range(4):
        rot = sk_rotate(rgbn[c].astype(np.float32) / 255.0, deg,
                         resize=True, mode="constant", cval=0.0, order=3)
        rotated_chans.append((rot * 255).clip(0, 255).astype(np.uint8))
    rotated_rgbn = np.stack(rotated_chans, axis=0)
    rotated_mask = sk_rotate(mask.astype(np.float32), deg, resize=True,
                              mode="constant", cval=0.0, order=0) > 0.5
    return rotated_rgbn, rotated_mask


def straighten_to_horizontal(rgbn: np.ndarray, mask: np.ndarray,
                              extra_rot_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Two-pass PCA, then apply extra_rot_deg manual nudge.
      pass 1: rotate by PCA principal-axis math angle
      pass 2: measure residual on rotated mask, apply refinement
      pass 3: apply manual extra_rot_deg (so it doesn't get cancelled by
              pass-2 PCA refinement)."""
    assert rgbn.shape[0] == 4
    angle1 = _pca_angle(mask)
    if np.isnan(angle1):
        return rgbn, mask, 0.0
    rot1 = angle1
    while rot1 > 90: rot1 -= 180
    while rot1 <= -90: rot1 += 180
    print(f"    PCA pass 1: principal @ {angle1:.1f}° → rotation {rot1:.1f}°")
    rgbn1, mask1 = _apply_rotation(rgbn, mask, rot1)
    angle2 = _pca_angle(mask1)
    while angle2 > 90: angle2 -= 180
    while angle2 <= -90: angle2 += 180
    print(f"    PCA pass 2: residual @ {angle2:.1f}° → applying refinement")
    rgbn2, mask2 = _apply_rotation(rgbn1, mask1, angle2)
    rotation_total = rot1 + angle2
    # Pass 3: manual nudge AFTER PCA, never undone
    if extra_rot_deg != 0.0:
        print(f"    manual extra_rot: {extra_rot_deg:+.1f}°")
        rgbn2, mask2 = _apply_rotation(rgbn2, mask2, extra_rot_deg)
        rotation_total += extra_rot_deg
    # Verify horizontal: long extent along x
    ys2, xs2 = np.where(mask2)
    if len(ys2) > 10:
        h_extent = xs2.max() - xs2.min()
        v_extent = ys2.max() - ys2.min()
        if v_extent > h_extent:
            print(f"    extent {h_extent}x{v_extent} → +90°")
            rgbn2, mask2 = _apply_rotation(rgbn2, mask2, 90.0)
            rotation_total += 90.0
    return rgbn2, mask2, rotation_total


def symmetrize_alpha(cutout5: np.ndarray, dilate_first: int = 4,
                       smooth_sigma: float = 1.5) -> np.ndarray:
    """Centerline symmetry + edge smoothing. Preserves bow/stern shape:
      1. dilate the alpha by `dilate_first` px (4 is generous; lets the
         entire bow/stern asymmetry overlap with its mirror)
      2. mirror-AND across the alpha-centroid horizontal centerline
      3. fill holes + keep largest CC + UNION with the original alpha
         (the original is symmetric where the boat actually is — this
         prevents the AND from chewing into the real hull)
      4. Gaussian-blur the alpha and re-threshold for smooth edges
    Centerline = alpha centroid (not geometric center; bow ornament biases that)."""
    out = cutout5.copy()
    alpha = out[4] > 0
    if alpha.sum() == 0:
        return out
    ys, _ = np.where(alpha)
    cy = int(round(ys.mean()))
    h, w = alpha.shape
    half = min(cy, h - 1 - cy)
    if half < 2:
        return out
    # Mirror-AND of the dilated mask
    alpha_thick = binary_dilation(alpha, iterations=dilate_first)
    band = alpha_thick[cy - half: cy + half + 1]
    flipped = band[::-1]
    sym_band = band & flipped
    sym = np.zeros_like(alpha)
    sym[cy - half: cy + half + 1] = sym_band
    sym = binary_fill_holes(sym)
    # Largest CC of the symmetry result
    lab_, n = cc_label(sym)
    if n > 0:
        sizes = np.bincount(lab_.ravel()); sizes[0] = 0
        sym = (lab_ == sizes.argmax())
    # Intersect symmetric region (which kills wake) with a slightly DILATED
    # original — keeps everything in the original hull that's roughly inside
    # the symmetric envelope. This restores bow/stern detail that strict
    # symmetry-AND would erase due to deck-feature asymmetry on those ends.
    final = sym & binary_dilation(alpha, iterations=2)
    final = binary_fill_holes(final)
    lab_, n = cc_label(final)
    if n > 0:
        sizes = np.bincount(lab_.ravel()); sizes[0] = 0
        final = (lab_ == sizes.argmax())
    # Smooth edges via Gaussian-blur + threshold
    if smooth_sigma > 0:
        f = gaussian_filter(final.astype(np.float32), sigma=smooth_sigma)
        final = f > 0.5
        final = binary_fill_holes(final)
    out[4] = final.astype(np.uint8) * 255
    return out


def tight_crop_5ch(rgbn: np.ndarray, mask: np.ndarray, pad: int = 4) -> np.ndarray:
    """Tight-crop around the mask + return 5-channel (RGBN + alpha as uint8)."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        # Fallback — return tiny dummy
        return np.zeros((5, 8, 8), dtype=np.uint8)
    y0 = max(0, ys.min() - pad); y1 = min(mask.shape[0], ys.max() + 1 + pad)
    x0 = max(0, xs.min() - pad); x1 = min(mask.shape[1], xs.max() + 1 + pad)
    rgbn_c = rgbn[:, y0:y1, x0:x1]
    mask_c = mask[y0:y1, x0:x1]
    # Convex-hull + erosion + symmetry as in v3 pipeline (boat1 only got these)
    alpha = (mask_c.astype(np.uint8) * 255)
    out = np.concatenate([rgbn_c, alpha[None]], axis=0)  # (5, h, w)
    return out


def post_process_v3(cutout5: np.ndarray, *, dilate: int = 1, hull: bool = True,
                     erode: int = 10, symmetry: bool = True) -> np.ndarray:
    """v3 cleanup: dilate the alpha, take convex hull, erode, mirror-and-AND for
    centerline symmetry. Reused from boat1's pipeline in build_canonical.py."""
    alpha = cutout5[4] > 0
    if dilate:
        alpha = binary_dilation(alpha, iterations=dilate)
    if hull:
        from scipy.spatial import ConvexHull
        ys, xs = np.where(alpha)
        if len(ys) >= 3:
            try:
                pts = np.column_stack([xs, ys])
                hull_idx = ConvexHull(pts)
                # Build hull mask via cv2.fillConvexPoly
                hull_pts = pts[hull_idx.vertices].astype(np.int32)
                m = np.zeros_like(alpha, dtype=np.uint8)
                cv2.fillConvexPoly(m, hull_pts, 1)
                alpha = m.astype(bool)
            except Exception:
                pass
    if erode:
        alpha = binary_erosion(alpha, iterations=erode)
    if symmetry:
        # Mirror across centerline (horizontal axis) and intersect
        alpha_flip = alpha[::-1]
        alpha = alpha & alpha_flip
    out = cutout5.copy()
    out[4] = (alpha.astype(np.uint8) * 255)
    return out


def main():
    water_full = np.load(ASSETS / "water_native_4b.npz")["img"]
    summary = {}
    for boat in ("boat1", "boat2"):
        print(f"--- {boat} ---")
        full4 = load_4b(boat)            # (4, 540, 540)
        rgbn = full4
        full_mask = boat_mask(full4, water_full, nir_thr_factor=1.6)
        print(f"  mask px: {full_mask.sum()}")
        # Straighten
        rotated_rgbn, rotated_mask, rotation = straighten_to_horizontal(
            rgbn, full_mask, extra_rot_deg=EXTRA_ROT[boat])
        print(f"  rotation applied: {rotation:.2f}°")
        # Tight crop
        cutout5 = tight_crop_5ch(rotated_rgbn, rotated_mask, pad=6)
        print(f"  cutout 5ch shape: {cutout5.shape}")
        # Light morphological cleanup
        alpha = cutout5[4] > 0
        alpha = binary_dilation(alpha, iterations=1)
        alpha = binary_fill_holes(alpha)
        cutout5[4] = alpha.astype(np.uint8) * 255
        print(f"  after fill+dilate: alpha px={(cutout5[4]>0).sum()}")
        # Centerline symmetry — kills wake-glint 'fingers' extending out from
        # the hull. A boat hull is symmetric top/bottom around its centerline
        # when oriented horizontally; a wake is not.
        cutout5 = symmetrize_alpha(cutout5)
        print(f"  after symmetry-AND: alpha px={(cutout5[4]>0).sum()}")
        # Save
        npz_path = ASSETS / f"canonical_{boat}_4b.npz"
        np.savez_compressed(npz_path, img=cutout5)
        # Preview as RGBA PNG
        rgba = np.concatenate([cutout5[:3], cutout5[4:5]], axis=0).transpose(1, 2, 0)
        Image.fromarray(rgba, mode="RGBA").save(
            ASSETS / f"canonical_{boat}_4b_preview.png")
        # NIR-only preview as L-mode
        Image.fromarray(cutout5[3], mode="L").save(
            ASSETS / f"canonical_{boat}_4b_nir_preview.png")
        # Sanity stats
        for c, bn in enumerate(["R", "G", "B", "NIR", "alpha"]):
            v = cutout5[c]
            in_mask = v[cutout5[4] > 0] if c < 4 else v
            mu = in_mask.mean() if len(in_mask) else 0.0
            print(f"    {bn}: mean(in-mask)={mu:.1f}")
        summary[boat] = {
            "rotation_applied_deg": float(rotation),
            "cutout_shape": list(cutout5.shape),
            "alpha_px": int((cutout5[4] > 0).sum()),
        }
    # Persist a small meta sidecar
    import json
    (ASSETS / "canonical_meta_4b.json").write_text(json.dumps(summary, indent=2))
    print("\nsaved canonical_meta_4b.json")


if __name__ == "__main__":
    main()
