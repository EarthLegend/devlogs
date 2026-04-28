"""pipeline.py — minimal compose → embed → probe → binding → ELLE pipeline.

Reads:  assets/canonical_<hull>_4b.npz, assets/water_native_4b.npz, encoder weights
Writes: assets/variants_4b/chips/*.npz, assets/variants_4b/embeddings.npy,
        assets/variants_4b/variants_meta.parquet, assets/probes_4b.json,
        assets/binding_4b.json, assets/binding_4b_emb.npy, assets/cos_vs_probe_4b.json,
        assets/dims_4b.json, assets/polygon_4b.json, assets/polygon_specific_4b.json,
        assets/elle_4b.json, assets/clipping_4b.json

CLI: python pipeline.py {compose | embed [SECS] | probes | binding | polygon |
                          cosvsprobe | dims | elle | clipping | widget [SECS] | all}
"""
from __future__ import annotations
import argparse, json, os, sys, time, warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib, numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.simplefilter("ignore", ConvergenceWarning)

# ── paths
HERE = Path(__file__).resolve().parent
A    = HERE / "assets"
V    = A / "variants_4b"
ROOT = HERE.parent.parent.parent
def _first(*ps):
    for p in ps:
        if p and Path(p).exists(): return Path(p)
    raise FileNotFoundError(ps)
sys.path.insert(0, str(_first(os.environ.get("OPENCLAY_DIR"), ROOT/"openclay",
                                Path.home()/"code/lgnd/openclay")))
from openclay.model.encoder import EmbeddingEncoder
from openclay.metadata import load_metadata, get_band_statistics, get_gsd
from openclay.datacube import create_datacube
def _enc_pt():
    return _first(os.environ.get("CLAY_ENCODER_PT"),
                   ROOT/"model/checkpoints/clay-v1.5_encoder.pt",
                   ROOT.parent/"outputs/clay_test/encoder_only.pt",  # local fallback
                   Path("/tmp/clay_encoder/encoder_only.pt"),
                   Path.home()/".cache/clay/encoder_only.pt")
META_YAML = _first(os.environ.get("OPENCLAY_DIR"), ROOT/"openclay",
                    Path.home()/"code/lgnd/openclay") / "metadata.yaml"

# ── constants
CHIP = 256
BOATS = ["boat1","boat2","boat_03","boat_04","boat_05","boat_06","boat_07","boat_08"]
OBL = [f"oblique_{b}" for b in BOATS]
POSITIONS = ["top_left","top","top_right","left","center","right","bottom_left","bottom","bottom_right"]
POS_FRAC  = {p:(0.20+(i%3)*0.30, 0.20+(i//3)*0.30) for i,p in enumerate(POSITIONS)}
SIZES   = [0.10, 0.25, 0.50, 0.75, 1.00]
COUNTS  = [0, 1, 2, 3, 5, 8]
ROT_CARD = [0, 90, 180, 270]
ROT_OBL  = [30, 60, 120, 150, 210, 240, 300, 330]    # 8 obliques (extends 4-cls cardinal)
ROT_8BIN = [0, 45, 90, 135, 180, 225, 270, 315]
LATLON, DATE = (37.72, -122.34), datetime(2020, 5, 24)

# ── 5-channel transform helpers (R, G, B, NIR, alpha)
def _resize(img, s):
    h, w = img.shape[1], img.shape[2]
    nw, nh = max(1,int(round(w*s))), max(1,int(round(h*s)))
    return np.stack([np.asarray(Image.fromarray(img[c]).resize((nw, nh), Image.BILINEAR))
                      for c in range(img.shape[0])])
def _rotate(img, deg):
    if not deg: return img
    return np.stack([np.asarray(Image.fromarray(img[c]).rotate(deg, resample=Image.BILINEAR, expand=True))
                      for c in range(img.shape[0])])
def _paste(canvas, sprite, cy, cx):
    h, w = sprite.shape[1], sprite.shape[2]
    y0, x0 = cy-h//2, cx-w//2
    ty0, tx0 = max(0,y0), max(0,x0)
    ty1, tx1 = min(CHIP, y0+h), min(CHIP, x0+w)
    sy0, sx0 = ty0-y0, tx0-x0
    sy1, sx1 = sy0+(ty1-ty0), sx0+(tx1-tx0)
    if ty1<=ty0 or tx1<=tx0: return canvas
    a = sprite[4, sy0:sy1, sx0:sx1].astype(np.float32) / 255.0
    canvas[:4, ty0:ty1, tx0:tx1] = (canvas[:4, ty0:ty1, tx0:tx1].astype(np.float32) * (1-a)
                                     + sprite[:4, sy0:sy1, sx0:sx1].astype(np.float32) * a).astype(np.uint8)
    return canvas

def _make_water_chip(water, seed):
    rng = np.random.default_rng(seed)
    H, W = water.shape[1], water.shape[2]
    y, x = rng.integers(0, H-CHIP), rng.integers(0, W-CHIP)
    chip = water[:, y:y+CHIP, x:x+CHIP].copy()
    if chip.shape[0] == 4:
        chip = np.concatenate([chip, np.full((1, CHIP, CHIP), 255, np.uint8)])
    return chip

# ── compose 1042-chip dataset
def cmd_compose():
    """Generate 1042 4-band chips matching the post's variant grid."""
    V.mkdir(parents=True, exist_ok=True); (V/"chips").mkdir(exist_ok=True)
    water = np.load(A/"water_native_4b.npz")["img"]
    if water.shape[0] == 4: water = np.concatenate([water, np.full((1,)+water.shape[1:], 255, np.uint8)])
    canon = {b: np.load(A/f"canonical_{b}_4b.npz")["img"] for b in BOATS}
    rows, idx = [], 0
    def emit(src, **kw):
        nonlocal idx
        chip = kw.pop("chip")
        np.savez_compressed(V/"chips"/f"chip_{idx:05d}.npz", img=chip[:4])
        Image.fromarray(chip[:3].transpose(1,2,0)).save(V/"chips"/f"chip_{idx:05d}.jpg", quality=82)
        rows.append({"chip_idx":idx,"chip_path":f"variants_4b/chips/chip_{idx:05d}.npz",
                      "source_id":src,**kw}); idx += 1
    for b in BOATS:
        boat = canon[b]
        # 36 pos × rot @ scale 0.25
        for pos in POSITIONS:
            for rot in ROT_CARD:
                cy, cx = (round(POS_FRAC[pos][1]*CHIP), round(POS_FRAC[pos][0]*CHIP))
                chip = _paste(_make_water_chip(water, hash(f"{b}{pos}{rot}")&0xFFFFFFFF),
                              _rotate(_resize(boat, 0.25), rot), cy, cx)
                emit(b, chip=chip, position=pos, rotation_deg=rot, scale=0.25)
        # 5 scales × 4 rot @ center
        for s in SIZES:
            for rot in ROT_CARD:
                if s == 0.25 and rot in ROT_CARD: continue  # dedup with pos×rot center cells
                chip = _paste(_make_water_chip(water, hash(f"{b}s{s}r{rot}")&0xFFFFFFFF),
                              _rotate(_resize(boat, s), rot), CHIP//2, CHIP//2)
                emit(b, chip=chip, position="center", rotation_deg=rot, scale=s)
        # cardinality (single hull) — N ∈ {0,1,2,3,5,8}, scale 0.10, deterministic seed
        for n in COUNTS:
            seed = hash(f"{b}n{n}")&0xFFFFFFFF
            rng = np.random.default_rng(seed)
            chip = _make_water_chip(water, seed)
            for _ in range(n):
                cy = int(rng.integers(48, CHIP-48)); cx = int(rng.integers(48, CHIP-48))
                chip = _paste(chip, _resize(boat, 0.10), cy, cx)
            emit(b, chip=chip, n_boats=n, scale=0.10, arrangement_seed=seed)
        # 8 oblique angles @ scale 0.25 center
        for rot in ROT_OBL:
            chip = _paste(_make_water_chip(water, hash(f"o{b}r{rot}")&0xFFFFFFFF),
                          _rotate(_resize(boat, 0.25), rot), CHIP//2, CHIP//2)
            emit(f"oblique_{b}", chip=chip, position="center", rotation_deg=rot, scale=0.25)
    # mixed-cardinality (sample N from full hull pool)
    for n in COUNTS:
        seed = hash(f"mixn{n}")&0xFFFFFFFF; rng = np.random.default_rng(seed)
        chip = _make_water_chip(water, seed)
        for _ in range(n):
            cy = int(rng.integers(48, CHIP-48)); cx = int(rng.integers(48, CHIP-48))
            chip = _paste(chip, _resize(canon[BOATS[int(rng.integers(0, len(BOATS)))]], 0.10), cy, cx)
        emit("mixed_boats", chip=chip, n_boats=n, scale=0.10, arrangement_seed=seed)
    # 36 nulls
    for k in range(36):
        emit("null_water", chip=_make_water_chip(water, k))
    rng = np.random.default_rng(0)
    for k in range(36):
        chip = (rng.integers(0, 256, (4, CHIP, CHIP), dtype=np.uint8))
        chip = np.concatenate([chip, np.full((1, CHIP, CHIP), 255, np.uint8)])
        emit("null_noise", chip=chip)
    pd.DataFrame(rows).to_parquet(V/"variants_meta.parquet", index=False)
    print(f"composed {len(rows)} chips")

# ── embed (resumable)
def _enc():
    sd = torch.load(_enc_pt(), map_location="cpu", weights_only=True)
    e = EmbeddingEncoder(img_size=256, patch_size=8, dim=1024, depth=24, heads=16, dim_head=64, mlp_ratio=4.0)
    e.load_state_dict(sd, strict=False); e.eval()
    return e

def cmd_embed(secs=40.0):
    df = pd.read_parquet(V/"variants_meta.parquet").reset_index(drop=True); n = len(df)
    EMB, DONE = V/"embeddings.npy", V/"embeddings_done.npy"
    emb = np.load(EMB) if EMB.exists() else np.zeros((n,1024), np.float32)
    done = set(np.load(DONE).tolist()) if DONE.exists() else set()
    todo = [k for k in range(n) if k not in done]
    if not todo: print(f"all {n} done"); return
    enc = _enc(); m = load_metadata(str(META_YAML))
    mn = get_band_statistics(m, "naip"); g = get_gsd(m, "naip")
    t0 = time.time()
    for i in range(0, len(todo), 8):
        batch = todo[i:i+8]
        pix = np.stack([np.load(A/df.loc[k,"chip_path"])["img"].astype(np.float32) for k in batch])
        with torch.no_grad():
            e = enc(create_datacube(pix, mn, [DATE]*len(batch), [LATLON]*len(batch), gsd=g, device="cpu")).numpy()
        emb[batch] = e; done.update(batch)
        np.save(EMB, emb); np.save(DONE, np.array(sorted(done), np.int64))
        print(f"  [{time.time()-t0:.0f}s] {len(done)}/{n}", flush=True)
        if time.time()-t0 >= secs: break

# ── probe definitions (filter, label, kind, chance)
_PL = {p:i for i,p in enumerate(POSITIONS)}
PROBES = {
    "rotation_4cls": (lambda d: d.source_id.isin(BOATS) & d.rotation_deg.isin(ROT_CARD) & d.scale.notna() & d.n_boats.isna(),
                      lambda s: s.rotation_deg.astype(int).to_numpy(), "clf", 0.25),
    "rotation_8bin": (lambda d: d.source_id.isin(BOATS+OBL) & d.scale.notna() & d.n_boats.isna() & d.rotation_deg.notna(),
                      lambda s: ((s.rotation_deg.astype(int).to_numpy()//45)%8), "clf", 0.125),
    "scale_R2":      (lambda d: d.source_id.isin(BOATS) & d.scale.notna() & d.n_boats.isna(),
                      lambda s: np.log(s.scale.astype(float).to_numpy()), "reg", 0.0),
    "count_R2":      (lambda d: d.source_id.isin(BOATS) & d.n_boats.notna(),
                      lambda s: s.n_boats.astype(float).to_numpy(), "reg", 0.0),
    "count_R2_mixed":(lambda d: d.source_id == "mixed_boats",
                      lambda s: s.n_boats.astype(float).to_numpy(), "reg", 0.0),
    "position_9cls_fixed_scale": (
        lambda d: d.source_id.isin(BOATS) & (d.scale==0.25) & d.rotation_deg.isin(ROT_CARD) & d.position.isin(POSITIONS),
        lambda s: np.array([_PL[p] for p in s.position]), "clf", 1/9),
}
_clf = lambda: make_pipeline(StandardScaler(), LinearSVC(C=1, dual=True, max_iter=3000, class_weight="balanced"))
_reg = lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))

# ── probes (within + LOBO + rotcont + hullsim + bootstrap, all in one)
def cmd_probes():
    df = pd.read_parquet(V/"variants_meta.parquet").reset_index(drop=True)
    emb = np.load(V/"embeddings.npy")
    out = {"n_chips": len(df), "lobo": {}, "lobo_per_hull": {}, "lobo_bootstrap_ci": {}}
    # within
    for k,(f,lab,kind,ch) in PROBES.items():
        sub = df[f(df)]; X, y = emb[sub.chip_idx.to_numpy()], lab(sub)
        cv = StratifiedKFold(5, shuffle=True, random_state=0) if kind=="clf" else KFold(5, shuffle=True, random_state=0)
        s = float(cross_val_score(_clf() if kind=="clf" else _reg(), X, y, cv=cv).mean())
        out[k] = {"n":len(y), ("acc" if kind=="clf" else "R2"): s, "chance": ch}
    # LOBO + per-hull + bootstrap
    rng = np.random.default_rng(0)
    for k,(f,lab,kind,_) in PROBES.items():
        if k == "count_R2_mixed": continue   # no per-hull split
        per = []
        for h in BOATS:
            base = f(df)
            tr = base & (df.source_id != h) & (df.source_id != f"oblique_{h}")
            te = base & ((df.source_id == h) | (df.source_id == f"oblique_{h}"))
            if tr.sum() < 5 or te.sum() < 5: continue
            p = _clf() if kind=="clf" else _reg()
            p.fit(emb[df[tr].chip_idx.to_numpy()], lab(df[tr]))
            ph = p.predict(emb[df[te].chip_idx.to_numpy()]); yt = lab(df[te])
            per.append((h, float((ph==yt).mean()) if kind=="clf" else float(r2_score(yt, ph))))
        scores = np.array([s for _,s in per])
        out["lobo"][k] = float(scores.mean()) if len(scores) else float("nan")
        out["lobo_per_hull"][k] = per
        if len(scores) > 1:
            boots = np.array([rng.choice(scores, len(scores), replace=True).mean() for _ in range(2000)])
            out["lobo_bootstrap_ci"][k] = {"mean":float(scores.mean()),
                "ci_lo_95":float(np.quantile(boots,.025)), "ci_hi_95":float(np.quantile(boots,.975)),
                "min":float(scores.min()), "max":float(scores.max()), "std":float(scores.std(ddof=1)), "n":len(scores)}
    # continuous rotation
    f, _, _, _ = PROBES["rotation_8bin"]; sub = df[f(df)]
    X = emb[sub.chip_idx.to_numpy()]; th = np.deg2rad(sub.rotation_deg.astype(float).to_numpy())
    cv = KFold(5, shuffle=True, random_state=0)
    yh = cross_val_predict(_reg(), X, np.sin(th), cv=cv)
    ch_ = cross_val_predict(_reg(), X, np.cos(th), cv=cv)
    err = np.rad2deg(np.arctan2(np.sin(np.arctan2(yh, ch_)-th), np.cos(np.arctan2(yh, ch_)-th)))
    per_l = []
    for h in BOATS:
        m = f(df) & ((df.source_id != h) & (df.source_id != f"oblique_{h}"))
        m_t = f(df) & ((df.source_id == h) | (df.source_id == f"oblique_{h}"))
        if m.sum() < 5 or m_t.sum() < 3: continue
        th_tr = np.deg2rad(df[m].rotation_deg.astype(float).to_numpy())
        th_te = np.deg2rad(df[m_t].rotation_deg.astype(float).to_numpy())
        Xt = emb[df[m].chip_idx.to_numpy()]; Xv = emb[df[m_t].chip_idx.to_numpy()]
        ps = _reg(); ps.fit(Xt, np.sin(th_tr))
        pc = _reg(); pc.fit(Xt, np.cos(th_tr))
        pth = np.arctan2(ps.predict(Xv), pc.predict(Xv))
        e = np.rad2deg(np.arctan2(np.sin(pth-th_te), np.cos(pth-th_te)))
        per_l.append((h, float(np.abs(e).mean())))
    out["rotation_continuous"] = {"within_mae_deg": float(np.abs(err).mean()),
        "within_median_deg": float(np.median(np.abs(err))),
        "lobo_mae_deg": float(np.mean([v for _,v in per_l])) if per_l else float("nan"),
        "lobo_per_hull": per_l, "n": len(th), "chance_random_mae_deg": 90.0}
    # hull similarity
    rows = [(b, emb[df[(df.source_id==b)&(df.scale==0.25)&(df.rotation_deg==0)&(df.position=="center")].chip_idx.iloc[0]])
            for b in BOATS]
    M = np.stack([v for _,v in rows]); M /= np.linalg.norm(M, axis=1, keepdims=True)
    cos = M @ M.T
    out["hull_similarity"] = {"boat_names":[b for b,_ in rows], "cosine_matrix": cos.tolist(),
        "off_diag_mean": float(cos[np.triu_indices_from(cos,1)].mean()),
        "off_diag_max":  float(cos[np.triu_indices_from(cos,1)].max()),
        "off_diag_min":  float(cos[np.triu_indices_from(cos,1)].min())}
    (A/"probes_4b.json").write_text(json.dumps(out, indent=2))
    print("wrote probes_4b.json")

# ── binding swap test
def cmd_binding():
    """Render 96-chip swap dataset, embed, analyse."""
    chips_dir = A/"binding_4b_chips"; chips_dir.mkdir(exist_ok=True)
    EMB = A/"binding_4b_emb.npy"; META = A/"binding_4b_meta.json"
    if not META.exists():
        b1 = np.load(A/"canonical_boat1_4b.npz")["img"]; b2 = np.load(A/"canonical_boat2_4b.npz")["img"]
        water = np.load(A/"water_native_4b.npz")["img"]
        if water.shape[0]==4: water = np.concatenate([water, np.full((1,)+water.shape[1:], 255, np.uint8)])
        TOP, BOT = (CHIP//2, 64), (CHIP//2, 192)
        items = []; pid = 0
        for s in [0.22, 0.30]:
            for r in [0, 90]:
                for sd in [11,23,37,53]:
                    for cfg, b1p, b2p, top in [("A", TOP, BOT, 1), ("B", BOT, TOP, 0)]:
                        items.append(dict(config=cfg, pair_id=pid, scale=s, rot=r, seed=sd, boat1_at_top=top, b1_pos=b1p, b2_pos=b2p))
                    pid += 1
                    items.append(dict(config="A_re", pair_id=pid-1, scale=s, rot=r, seed=sd+1000, boat1_at_top=1, b1_pos=TOP, b2_pos=BOT))
        for s in [0.22, 0.30]:
            for r in [0, 90]:
                for sd in [11, 23]:
                    items.append(dict(config="C1", pair_id=-1, scale=s, rot=r, seed=sd, boat1_at_top=1, b1_pos=TOP, b2_pos=None))
                    items.append(dict(config="C2", pair_id=-1, scale=s, rot=r, seed=sd, boat1_at_top=0, b1_pos=BOT, b2_pos=None))
        for k, it in enumerate(items):
            chip = _make_water_chip(water, it["seed"])
            b1s = _rotate(_resize(b1, it["scale"]), it["rot"])
            chip = _paste(chip, b1s, *(it["b1_pos"][::-1] if it["config"] in ("A","B") and not it["boat1_at_top"] else it["b1_pos"][::-1]))
            if it["b2_pos"] is not None:
                chip = _paste(chip, _rotate(_resize(b2, it["scale"]), it["rot"]), it["b2_pos"][1], it["b2_pos"][0])
            np.savez_compressed(chips_dir/f"chip_{k:03d}.npz", img=chip[:4])
        META.write_text(json.dumps(items, default=str))
    items = json.loads(META.read_text())
    DONE = A/"binding_4b_done.npy"
    emb = np.load(EMB) if EMB.exists() else np.zeros((len(items), 1024), np.float32)
    done = set(np.load(DONE).tolist()) if DONE.exists() else set()
    todo = [k for k in range(len(items)) if k not in done]
    if todo:
        enc = _enc(); m = load_metadata(str(META_YAML))
        mn = get_band_statistics(m, "naip"); g = get_gsd(m, "naip")
        for i in range(0, len(todo), 8):
            batch = todo[i:i+8]
            pix = np.stack([np.load(chips_dir/f"chip_{k:03d}.npz")["img"].astype(np.float32) for k in batch])
            with torch.no_grad():
                e = enc(create_datacube(pix, mn, [DATE]*len(batch), [LATLON]*len(batch), gsd=g, device="cpu")).numpy()
            emb[batch] = e; done.update(batch)
            np.save(EMB, emb); np.save(DONE, np.array(sorted(done), np.int64))
    # analyse
    by_pair = {}
    for k, it in enumerate(items):
        if it["pair_id"] >= 0: by_pair.setdefault(it["pair_id"], {})[it["config"]] = k
    pairs = [p for p,d in by_pair.items() if {"A","B","A_re"}.issubset(d)]
    cs = lambda a,b: float((a/np.linalg.norm(a)) @ (b/np.linalg.norm(b)))
    swap_cos  = np.array([cs(emb[by_pair[p]["A"]], emb[by_pair[p]["B"]]) for p in pairs])
    noise_cos = np.array([cs(emb[by_pair[p]["A"]], emb[by_pair[p]["A_re"]]) for p in pairs])
    AB = [k for k,it in enumerate(items) if it["config"] in ("A","B")]
    X = emb[AB]; y = np.array([items[k]["boat1_at_top"] for k in AB])
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    acc = float(cross_val_score(pipe, X, y, cv=cv).mean())
    rng = np.random.default_rng(0)
    perm = np.array([cross_val_score(pipe, X, rng.permutation(y), cv=cv).mean() for _ in range(200)])
    deltas = np.stack([emb[by_pair[p]["A"]] - emb[by_pair[p]["B"]] for p in pairs])
    deltas /= np.linalg.norm(deltas, axis=1, keepdims=True) + 1e-12
    dc = [float((deltas[i]*deltas[j]).sum()) for i in range(len(deltas)) for j in range(i+1,len(deltas))]
    (A/"binding_4b.json").write_text(json.dumps({
        "n_pairs": len(pairs),
        "swap_cos_mean": float(swap_cos.mean()), "swap_cos_std": float(swap_cos.std()),
        "noise_cos_mean": float(noise_cos.mean()), "noise_cos_std": float(noise_cos.std()),
        "binding_probe_acc": acc, "binding_probe_perm_p": float((perm>=acc).mean()),
        "binding_probe_perm_mean": float(perm.mean()),
        "delta_consistency_mean_pairwise_cos": float(np.mean(dc)),
    }, indent=2))
    print("wrote binding_4b.json")

# ── polygon retrieval (general + specific-boat) ── reconstruct GT mask analytically
def _paste_alpha(cv, alpha_img, cy, cx, scale, rot):
    """Paste a rotated/scaled alpha onto a 256×256 float32 canvas via per-pixel max."""
    H, W = cv.shape
    a_im = Image.fromarray(alpha_img).resize(
        (max(1, int(round(alpha_img.shape[1]*scale))), max(1, int(round(alpha_img.shape[0]*scale)))),
        Image.BILINEAR)
    if rot: a_im = a_im.rotate(rot, resample=Image.BILINEAR, expand=True)
    a = np.asarray(a_im, np.float32) / 255
    h, w = a.shape; y0, x0 = cy-h//2, cx-w//2
    ty0, tx0, ty1, tx1 = max(0,y0), max(0,x0), min(H,y0+h), min(W,x0+w)
    sy0, sx0 = ty0-y0, tx0-x0; sy1, sx1 = sy0+(ty1-ty0), sx0+(tx1-tx0)
    if ty1 > ty0 and tx1 > tx0:
        np.maximum(cv[ty0:ty1, tx0:tx1], a[sy0:sy1, sx0:sx1], out=cv[ty0:ty1, tx0:tx1])
    return cv

def _gt_mask(row, alphas, G=32):
    """Reconstruct the chip's 32×32 alpha-mass mask analytically + return positions used."""
    H = W = CHIP; cell = H // G
    src = row["source_id"]
    hull = src if src in alphas else (src.replace("oblique_","") if src.startswith("oblique_") else None)
    m = np.zeros((H, W), np.float32)
    if hull is None: return m.reshape(G,cell,G,cell).mean(axis=(1,3)), []
    if pd.notna(row.get("n_boats")):
        rng = np.random.default_rng(int(row["arrangement_seed"]))
        ps = sorted([(int(rng.integers(48, H-48)), int(rng.integers(48, W-48))) for _ in range(int(row["n_boats"]))],
                    key=lambda p: p[1])
        for cy, cx in ps: _paste_alpha(m, alphas[hull], cy, cx, 0.10, 0)
        return m.reshape(G,cell,G,cell).mean(axis=(1,3)), ps
    pos = row.get("position"); scale = float(row["scale"])
    rot = int(row["rotation_deg"]) if pd.notna(row["rotation_deg"]) else 0
    cy, cx = (round(POS_FRAC[pos][1]*CHIP), round(POS_FRAC[pos][0]*CHIP)) if pos in POS_FRAC else (CHIP//2, CHIP//2)
    _paste_alpha(m, alphas[hull], cy, cx, scale, rot)
    return m.reshape(G,cell,G,cell).mean(axis=(1,3)), [(cy, cx)]

def cmd_polygon():
    df = pd.read_parquet(V/"variants_meta.parquet").reset_index(drop=True)
    emb = np.load(V/"embeddings.npy")
    alphas = {b: (np.load(A/f"canonical_{b}_4b.npz")["img"][4]).astype(np.uint8) for b in BOATS}
    cands = df[df.source_id.isin(BOATS+OBL)].reset_index(drop=True)
    Y = np.zeros((len(cands), 1024), np.float32); leftmost_pos = []
    for i, r in cands.iterrows():
        m, ps = _gt_mask(r, alphas); Y[i] = m.flatten(); leftmost_pos.append(ps[0] if ps else None)
    X = emb[cands.chip_idx.to_numpy()]
    # general: per-stratum soft-IoU
    def fold_iou(Xs, Ys):
        kf = KFold(5, shuffle=True, random_state=0); P = np.zeros_like(Ys)
        for tr, te in kf.split(Xs):
            P[te] = Ridge(alpha=1.0).fit(Xs[tr], Ys[tr]).predict(Xs[te])
        P = P.clip(0, 1); eps=1e-6
        i_,u_ = np.minimum(P,Ys).sum(1), np.maximum(P,Ys).sum(1)+eps
        return float((i_/u_).mean()), P
    def base_iou(Ys):
        kf = KFold(5, shuffle=True, random_state=0); ious=[]
        for tr, te in kf.split(Ys):
            mu = Ys[tr].mean(0); P = np.broadcast_to(mu, Ys[te].shape).clip(0,1)
            i_,u_ = np.minimum(P,Ys[te]).sum(1), np.maximum(P,Ys[te]).sum(1)+1e-6
            ious.append((i_/u_).mean())
        return float(np.mean(ious))
    strata = {
        "single instance": cands.source_id.isin(BOATS) & cands.n_boats.isna(),
        "oblique single":  cands.source_id.str.startswith("oblique_"),
        **{f"multi N={n}": cands.source_id.isin(BOATS) & (cands.n_boats==n) for n in [2,3,5,8]},
    }
    rows = []
    for name, mask in strata.items():
        idx = mask.to_numpy().nonzero()[0]
        if len(idx) < 10: continue
        ip, _ = fold_iou(X[idx], Y[idx])
        rows.append({"label":name, "n":int(len(idx)), "iou_probe":ip, "iou_baseline":base_iou(Y[idx])})
    (A/"polygon_4b.json").write_text(json.dumps(rows, indent=2))
    # specific-boat (leftmost): ridge on leftmost-only target per-N
    spec = []
    for n in [1,2,3,5,8]:
        sub = (cands.source_id.isin(BOATS) & (cands.n_boats==n)).to_numpy().nonzero()[0]
        if len(sub) < 10: continue
        Y_left = np.zeros((len(sub), 1024), np.float32); Y_uni = np.zeros_like(Y_left)
        for i, k in enumerate(sub):
            r = cands.iloc[k]; rng = np.random.default_rng(int(r.arrangement_seed))
            ps = sorted([(int(rng.integers(48, CHIP-48)), int(rng.integers(48, CHIP-48))) for _ in range(int(r.n_boats))], key=lambda p: p[1])
            hull = r.source_id; left, uni = np.zeros((CHIP,CHIP),np.float32), np.zeros((CHIP,CHIP),np.float32)
            for j, (cy,cx) in enumerate(ps):
                tmp = np.zeros((CHIP,CHIP), np.float32)
                a = (np.asarray(Image.fromarray(alphas[hull]).resize(
                    (max(1,int(round(alphas[hull].shape[1]*0.10))), max(1,int(round(alphas[hull].shape[0]*0.10)))),
                    Image.BILINEAR))).astype(np.float32)/255
                h,w = a.shape; y0,x0 = cy-h//2,cx-w//2; y1,x1=y0+h,x0+w
                ty0,tx0,ty1,tx1=max(0,y0),max(0,x0),min(CHIP,y1),min(CHIP,x1)
                sy0,sx0=ty0-y0,tx0-x0; sy1,sx1=sy0+(ty1-ty0),sx0+(tx1-tx0)
                if ty1>ty0 and tx1>tx0: tmp[ty0:ty1, tx0:tx1] = a[sy0:sy1, sx0:sx1]
                np.maximum(uni, tmp, out=uni)
                if j == 0: left = tmp
            Y_left[i] = left.reshape(32,8,32,8).mean(axis=(1,3)).flatten()
            Y_uni[i]  = uni.reshape(32,8,32,8).mean(axis=(1,3)).flatten()
        Xs = X[sub]; kf = KFold(5, shuffle=True, random_state=0); P = np.zeros_like(Y_left)
        for tr, te in kf.split(Xs):
            P[te] = Ridge(alpha=1.0).fit(Xs[tr], Y_left[tr]).predict(Xs[te])
        P = P.clip(0, 1)
        ip = float((np.minimum(P, Y_left).sum(1)/(np.maximum(P, Y_left).sum(1)+1e-6)).mean())
        # mean-mask baseline + union-oracle ceiling
        ib = []
        for tr, te in kf.split(Y_left):
            mu = Y_left[tr].mean(0); pr = np.broadcast_to(mu, Y_left[te].shape).clip(0,1)
            ib.append((np.minimum(pr, Y_left[te]).sum(1)/(np.maximum(pr, Y_left[te]).sum(1)+1e-6)).mean())
        oracle = (np.minimum(Y_uni, Y_left).sum(1)/(np.maximum(Y_uni, Y_left).sum(1)+1e-6)).mean()
        spec.append({"N":f"N={n}", "n":int(len(sub)), "iou_probe":ip, "iou_baseline":float(np.mean(ib)),
                      "iou_oracle_union":float(oracle)})
    (A/"polygon_specific_4b.json").write_text(json.dumps(spec, indent=2))
    print("wrote polygon_4b.json + polygon_specific_4b.json")

# ── cosine NN vs linear probe under LOBO
def cmd_cosvsprobe():
    df = pd.read_parquet(V/"variants_meta.parquet").reset_index(drop=True)
    emb = np.load(V/"embeddings.npy")
    out = {}
    for k,(f,lab,kind,ch) in PROBES.items():
        if k == "count_R2_mixed": continue
        cos_per, pr_per = [], []
        for h in BOATS:
            base = f(df)
            tr = base & (df.source_id != h) & (df.source_id != f"oblique_{h}")
            te = base & ((df.source_id == h) | (df.source_id == f"oblique_{h}"))
            if tr.sum()<5 or te.sum()<5: continue
            Xt, Xv = emb[df[tr].chip_idx.to_numpy()], emb[df[te].chip_idx.to_numpy()]
            yt, yv = lab(df[tr]), lab(df[te])
            # cosine top-5 NN voting
            Xtn = Xt / (np.linalg.norm(Xt,axis=1,keepdims=True)+1e-12)
            Xvn = Xv / (np.linalg.norm(Xv,axis=1,keepdims=True)+1e-12)
            idx = np.argsort(-(Xvn @ Xtn.T), axis=1)[:, :5]
            if kind == "clf":
                preds = np.array([Counter(yt[r].tolist()).most_common(1)[0][0] for r in idx])
                cs = float((preds == yv).mean())
            else:
                cs = float(r2_score(yv, yt[idx].mean(axis=1)))
            cos_per.append((h, cs))
            p = (_clf() if kind=="clf" else _reg()); p.fit(Xt, yt)
            ph = p.predict(Xv)
            pr_per.append((h, float((ph==yv).mean()) if kind=="clf" else float(r2_score(yv, ph))))
        out[k] = {"kind":kind, "chance":ch, "cos_per_hull":cos_per, "probe_per_hull":pr_per,
                   "cos_mean":float(np.mean([s for _,s in cos_per])) if cos_per else float("nan"),
                   "probe_mean":float(np.mean([s for _,s in pr_per])) if pr_per else float("nan")}
    (A/"cos_vs_probe_4b.json").write_text(json.dumps(out, indent=2))
    print("wrote cos_vs_probe_4b.json")

# ── dimensionality (native dims-0:K vs PCA-K vs raw 1024)
def cmd_dims():
    df = pd.read_parquet(V/"variants_meta.parquet").reset_index(drop=True)
    emb = np.load(V/"embeddings.npy")
    out = {}
    # PCA features are already zero-mean with descending-variance components;
    # an inner StandardScaler would unit-scale every PC and force Ridge to
    # regularise noise PCs equally with signal PCs (collapses regression at K≥256).
    _pca_reg = lambda: Ridge(alpha=1.0)
    _pca_clf = lambda: LinearSVC(C=1, dual=True, max_iter=3000, class_weight="balanced")
    for k,(f,lab,kind,ch) in PROBES.items():
        if k in ("rotation_8bin", "count_R2_mixed"): continue
        sub = df[f(df)]; X = emb[sub.chip_idx.to_numpy()]; y = lab(sub)
        cv = StratifiedKFold(5, shuffle=True, random_state=0) if kind=="clf" else KFold(5, shuffle=True, random_state=0)
        Xp = PCA(n_components=min(256, len(X)-1, X.shape[1])).fit_transform(StandardScaler().fit_transform(X))
        row = {"native_prefix":[], "pca_k":[], "chance":ch, "raw_1024":None}
        for K in [2, 16, 64, 256, 1024]:
            if K == 1024:
                row["raw_1024"] = float(cross_val_score((_clf() if kind=="clf" else _reg()), X, y, cv=cv).mean())
            else:
                row["native_prefix"].append([K, float(cross_val_score((_clf() if kind=="clf" else _reg()), X[:,:K], y, cv=cv).mean())])
                if K <= Xp.shape[1]:
                    row["pca_k"].append([K, float(cross_val_score((_pca_clf() if kind=="clf" else _pca_reg()), Xp[:,:K], y, cv=cv).mean())])
        out[{"rotation_4cls":"rotation 4-cls","scale_R2":"scale R²","count_R2":"count R²",
              "position_9cls_fixed_scale":"position 9-cls"}[k]] = row
    (A/"dims_4b.json").write_text(json.dumps(out, indent=2))
    print("wrote dims_4b.json")

# ── ELLE applied to this build
def cmd_elle():
    df = pd.read_parquet(V/"variants_meta.parquet").reset_index(drop=True)
    emb = np.load(V/"embeddings.npy")
    pairs = _first(ROOT/"elle/data/clay_naip_pairs.pt", Path.home()/"code/lgnd/elle/data/clay_naip_pairs.pt")
    d = torch.load(pairs, map_location="cpu", weights_only=False)
    Xt, yt = d["cls_emb"].numpy().astype(np.float32), d["loss"].numpy().astype(np.float32)
    head = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xt, yt)
    z = (head.predict(emb) - yt.mean()) / yt.std()
    def stratum(s):
        if s.startswith("oblique_"): return "oblique"
        if s in BOATS: return "single hull"
        if s == "mixed_boats": return "mixed cardinality"
        return s
    df["z"] = z; df["stratum"] = df.source_id.map(stratum)
    by_str = {s: {"n":int((df.stratum==s).sum()),
                   "elle_z_mean":float(df[df.stratum==s].z.mean()),
                   "elle_z_std":float(df[df.stratum==s].z.std())}
               for s in df.stratum.unique()}
    probes = json.loads((A/"probes_4b.json").read_text())
    perh = {k: dict(probes["lobo_per_hull"][k]) for k in ("rotation_4cls","count_R2","scale_R2")}
    by_hull = {}
    for h in BOATS:
        m = (df.source_id == h) | (df.source_id == f"oblique_{h}")
        if not m.any(): continue
        by_hull[h] = {"elle_z": float(df[m].z.mean()),
                       "rot4cls": perh["rotation_4cls"].get(h, np.nan),
                       "countR2": perh["count_R2"].get(h, np.nan),
                       "scaleR2": perh["scale_R2"].get(h, np.nan)}
    def r(ys):
        v = [(by_hull[h]["elle_z"], by_hull[h][ys]) for h in by_hull if not np.isnan(by_hull[h][ys])]
        return float(np.corrcoef(*zip(*v))[0,1]) if len(v) > 2 else None
    (A/"elle_4b.json").write_text(json.dumps({
        "by_stratum":by_str, "by_hull":by_hull,
        "pearson_r":{"rotation 4-cls": r("rot4cls"), "count R²": r("countR2"), "scale R²": r("scaleR2")}
    }, indent=2))
    print("wrote elle_4b.json")

# ── clipping audit
def cmd_clipping():
    df = pd.read_parquet(V/"variants_meta.parquet").reset_index(drop=True)
    emb = np.load(V/"embeddings.npy")
    alphas = {b: np.load(A/f"canonical_{b}_4b.npz")["img"][4].astype(np.uint8) for b in BOATS}
    ovf = np.zeros(len(df), np.float32)
    for i, r in df.iterrows():
        src = r["source_id"]
        h = src if src in alphas else (src.replace("oblique_","") if src.startswith("oblique_") else None)
        if h is None or pd.notna(r.get("n_boats")) or r.get("position") not in POS_FRAC: continue
        cy, cx = (round(POS_FRAC[r.position][1]*CHIP), round(POS_FRAC[r.position][0]*CHIP))
        a_im = Image.fromarray(alphas[h]).resize(
            (max(1,int(round(alphas[h].shape[1]*r.scale))), max(1,int(round(alphas[h].shape[0]*r.scale)))),
            Image.BILINEAR)
        if r.rotation_deg: a_im = a_im.rotate(int(r.rotation_deg), resample=Image.BILINEAR, expand=True)
        ar = np.asarray(a_im, np.float32) / 255
        total = ar.sum()
        if total < 1: continue
        ah, aw = ar.shape; y0, x0 = cy-ah//2, cx-aw//2
        ty0,tx0,ty1,tx1 = max(0,y0),max(0,x0),min(CHIP,y0+ah),min(CHIP,x0+aw)
        sy0,sx0 = ty0-y0,tx0-x0; sy1,sx1 = sy0+(ty1-ty0), sx0+(tx1-tx0)
        vis = ar[sy0:sy1, sx0:sx1].sum() if ty1>ty0 and tx1>tx0 else 0
        ovf[i] = 1 - vis/total
    out = {"clipping_threshold":0.20, "n_total":len(df),
           "n_clipped_>=20%": int((ovf>=0.20).sum()),
           "n_clipped_>=50%": int((ovf>=0.50).sum())}
    np.save(A/"_overflow.npy", ovf)
    (A/"clipping_4b.json").write_text(json.dumps(out, indent=2))
    print(f"clipping audit: {out['n_clipped_>=20%']}/{out['n_total']} chips ≥ 20% overflow")

# ── widget data (boat_06 cross-product, with live-paint browser rendering)
def cmd_widget(secs=38.0):
    """Render unique chips, embed, train probes once, build widget_data_4b.json."""
    HULL = "boat_06"; CHIPS = A/"_w06_chips"; CHIPS.mkdir(exist_ok=True)
    IDX, EMB, DONE = A/"widget_full_boat06_index.json", A/"widget_full_boat06_emb.npy", A/"widget_full_boat06_done.npy"
    POS9_W = {p:(0.20+(i%3)*0.30, 0.20+(i//3)*0.30) for i,p in enumerate(POSITIONS)}
    OFFSETS = [(0,0),(-55,-35),(55,-35),(-55,35),(55,35),(0,-65),(0,65),(-75,0),(75,0)]
    if not IDX.exists():
        water_jpg = np.asarray(Image.open(A/"water_native_4b.jpg").convert("RGB"))
        water_npz = np.load(A/"water_native_4b.npz")["img"]
        bg4 = np.concatenate([water_jpg[:CHIP,:CHIP].transpose(2,0,1), water_npz[3:4,:CHIP,:CHIP]])
        canon_png = np.asarray(Image.open(A/f"canonical_{HULL}_4b_preview.png").convert("RGBA"))
        canon_nir = np.load(A/f"canonical_{HULL}_4b.npz")["img"][3]
        seen, idx, recs = {}, 0, []
        for s in SIZES:
            for n in COUNTS:
                for p in POSITIONS:
                    for r in ROT_8BIN:
                        key = (None,0,None,None) if n==0 else (s,n,p,r)
                        if key not in seen:
                            chip = bg4.astype(np.float32).copy()
                            if n > 0:
                                target = max(4, int(round(s*CHIP)))
                                ratio = target / max(canon_png.shape[:2])
                                nh, nw = max(1,int(round(canon_png.shape[0]*ratio))), max(1,int(round(canon_png.shape[1]*ratio)))
                                pil_rgba = Image.fromarray(canon_png).resize((nw,nh), Image.BICUBIC)
                                pil_nir  = Image.fromarray(canon_nir).resize((nw,nh), Image.BICUBIC)
                                if r:
                                    pil_rgba = pil_rgba.rotate(r, resample=Image.BICUBIC, expand=True)
                                    pil_nir  = pil_nir.rotate(r, resample=Image.BICUBIC, expand=True)
                                rgba = np.asarray(pil_rgba); nir = np.asarray(pil_nir).astype(np.float32)
                                fx, fy = POS9_W[p]; bcy, bcx = round(fy*CHIP), round(fx*CHIP)
                                for i in range(n):
                                    ox, oy = OFFSETS[i]
                                    py = max(36, min(CHIP-36, bcy+oy)); px = max(36, min(CHIP-36, bcx+ox))
                                    ah, aw = rgba.shape[:2]
                                    y0, x0 = py-ah//2, px-aw//2
                                    ty0,tx0,ty1,tx1 = max(0,y0),max(0,x0),min(CHIP,y0+ah),min(CHIP,x0+aw)
                                    sy0,sx0 = ty0-y0,tx0-x0; sy1,sx1 = sy0+(ty1-ty0), sx0+(tx1-tx0)
                                    if ty1<=ty0 or tx1<=tx0: continue
                                    a = rgba[sy0:sy1,sx0:sx1,3].astype(np.float32)/255
                                    chip[:3, ty0:ty1, tx0:tx1] = chip[:3,ty0:ty1,tx0:tx1]*(1-a) + rgba[sy0:sy1,sx0:sx1,:3].astype(np.float32).transpose(2,0,1)*a
                                    chip[3,  ty0:ty1, tx0:tx1] = chip[3, ty0:ty1, tx0:tx1]*(1-a) + nir[sy0:sy1,sx0:sx1]*a
                            np.savez_compressed(CHIPS/f"chip_{idx:05d}.npz", img=chip.clip(0,255).astype(np.uint8))
                            seen[key] = idx; idx += 1
                        recs.append({"size":s,"count":n,"position":p,"rotation":r,"chip_uid":seen[key]})
        IDX.write_text(json.dumps({"index":recs,"n_unique":len(seen)}, indent=2))
    info = json.loads(IDX.read_text()); n = info["n_unique"]
    emb = np.load(EMB) if EMB.exists() else np.zeros((n,1024), np.float32)
    done = set(np.load(DONE).tolist()) if DONE.exists() else set()
    todo = [k for k in range(n) if k not in done]
    if todo:
        enc = _enc(); m = load_metadata(str(META_YAML))
        mn, g = get_band_statistics(m, "naip"), get_gsd(m, "naip")
        t0 = time.time()
        for i in range(0, len(todo), 8):
            batch = todo[i:i+8]
            pix = np.stack([np.load(CHIPS/f"chip_{k:05d}.npz")["img"].astype(np.float32) for k in batch])
            with torch.no_grad():
                e = enc(create_datacube(pix, mn, [DATE]*len(batch), [LATLON]*len(batch), gsd=g, device="cpu")).numpy()
            emb[batch] = e; done.update(batch)
            np.save(EMB, emb); np.save(DONE, np.array(sorted(done), np.int64))
            print(f"  widget [{time.time()-t0:.0f}s] {len(done)}/{n}", flush=True)
            if time.time()-t0 >= secs: return
    # train within-object probes once on existing variants_4b data
    df = pd.read_parquet(V/"variants_meta.parquet").reset_index(drop=True)
    big = np.load(V/"embeddings.npy")
    pl = {p:i for i,p in enumerate(POSITIONS)}
    fits = {}
    h_m = df.source_id.isin(BOATS+OBL); h_y = np.array([BOATS.index(s.replace("oblique_","")) for s in df[h_m].source_id])
    fits["hull"] = ("clf", _clf().fit(big[df[h_m].chip_idx.to_numpy()], h_y), BOATS)
    s_m = df.source_id.isin(BOATS) & df.scale.notna() & df.n_boats.isna()
    fits["size"] = ("reg_log", _reg().fit(big[df[s_m].chip_idx.to_numpy()], np.log(df[s_m].scale.astype(float))), None)
    c_m = df.source_id.isin(BOATS+["mixed_boats"]) & df.n_boats.notna()
    fits["count"] = ("reg", _reg().fit(big[df[c_m].chip_idx.to_numpy()], df[c_m].n_boats.astype(float)), None)
    p_m = df.source_id.isin(BOATS) & (df.scale==0.25) & df.rotation_deg.isin(ROT_CARD) & df.position.isin(POSITIONS)
    fits["position"] = ("clf", _clf().fit(big[df[p_m].chip_idx.to_numpy()], np.array([pl[p] for p in df[p_m].position])), POSITIONS)
    r_m = df.source_id.isin(BOATS+OBL) & df.scale.notna() & df.n_boats.isna() & df.rotation_deg.notna()
    fits["rotation"] = ("clf", _clf().fit(big[df[r_m].chip_idx.to_numpy()], (df[r_m].rotation_deg.astype(int).to_numpy()//45)%8), ROT_8BIN)
    cache = {axis: [None]*n for axis in fits}
    for axis,(kind, mdl, levels) in fits.items():
        yh = mdl.predict(emb)
        for i, y in enumerate(yh):
            cache[axis][i] = (levels[int(y)] if kind=="clf" else (float(np.exp(y)) if kind=="reg_log" else float(y)))
    pp = json.loads((A/"probes_4b.json").read_text())
    chips_out = []
    for r in info["index"]:
        u = r["chip_uid"]; isn = (r["count"]==0)
        chips_out.append({"source_id":"null_water" if isn else HULL, "hull":None if isn else HULL,
            "size":r["size"], "count":r["count"], "position":r["position"], "rotation":r["rotation"], "is_null":isn,
            "gt": {"hull":None if isn else HULL, "size":None if isn else r["size"], "count":float(r["count"]),
                    "position":None if isn else r["position"], "rotation":None if isn else r["rotation"]},
            "pred":{"hull":cache["hull"][u], "size":cache["size"][u], "count":cache["count"][u],
                     "position":cache["position"][u], "rotation":int(cache["rotation"][u]) if cache["rotation"][u] is not None else None}})
    (A/"widget_data_4b.json").write_text(json.dumps({
        "axes":{
            "hull":{"levels":[HULL], "kind":"classification", "within_object_5fold":"—", "cross_hull_lobo":"—"},
            "size":{"levels":SIZES, "kind":"regression",
                     "within_object_5fold":f"R² {pp['scale_R2']['R2']:.2f}", "cross_hull_lobo":f"R² {pp['lobo']['scale_R2']:.2f}"},
            "count":{"levels":COUNTS, "kind":"regression",
                      "within_object_5fold":f"R² {pp['count_R2']['R2']:.2f}", "cross_hull_lobo":f"R² {pp['lobo']['count_R2']:.2f}"},
            "position":{"levels":POSITIONS, "kind":"classification",
                         "within_object_5fold":f"{pp['position_9cls_fixed_scale']['acc']:.2f}",
                         "cross_hull_lobo":f"{pp['lobo']['position_9cls_fixed_scale']:.2f}"},
            "rotation":{"levels":ROT_8BIN, "kind":"classification",
                         "within_object_5fold":f"{pp['rotation_8bin']['acc']:.2f}",
                         "cross_hull_lobo":f"{pp['lobo']['rotation_8bin']:.2f}"},
        }, "chips":chips_out}, default=str))
    print("wrote widget_data_4b.json")

# ── CLI
def main():
    ap = argparse.ArgumentParser(prog="pipeline.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("compose","probes","binding","polygon","cosvsprobe","dims","elle","clipping","all"):
        sub.add_parser(name)
    e = sub.add_parser("embed"); e.add_argument("secs", nargs="?", type=float, default=40.0)
    w = sub.add_parser("widget"); w.add_argument("secs", nargs="?", type=float, default=38.0)
    args = ap.parse_args()
    if args.cmd == "all":
        for c in (cmd_compose, cmd_embed, cmd_probes, cmd_binding, cmd_polygon, cmd_cosvsprobe, cmd_dims, cmd_elle, cmd_clipping, cmd_widget): c()
    else:
        {"compose":cmd_compose, "embed":lambda: cmd_embed(args.secs), "probes":cmd_probes, "binding":cmd_binding,
          "polygon":cmd_polygon, "cosvsprobe":cmd_cosvsprobe, "dims":cmd_dims, "elle":cmd_elle,
          "clipping":cmd_clipping, "widget":lambda: cmd_widget(args.secs)}[args.cmd]()

if __name__ == "__main__": main()
