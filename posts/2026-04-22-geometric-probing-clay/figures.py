"""figures.py — all 17 figures for the geometric-probing-clay devlog.

Reads from assets/*.json + assets/variants_4b/{embeddings.npy,variants_meta.parquet,chips/}.
Each fig_* is independent. CLI: python figures.py {all | <name>}.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
A    = HERE / "assets"
V    = A / "variants_4b"
sys.path.insert(0, str(HERE))
from pipeline import (BOATS, OBL, POSITIONS, POS_FRAC, SIZES, COUNTS, ROT_CARD,
                       CHIP, _resize, _rotate)   # reuse primitives

plt.rcParams.update({"figure.facecolor":"white", "axes.facecolor":"white",
                      "savefig.facecolor":"white", "font.family":"DejaVu Sans",
                      "font.size":10, "axes.grid":True, "grid.alpha":0.18})
COS, PRB = "#5b9bd5", "#f58518"

def _save(fig, name):
    out = A / f"fig_{name}_4b.png"; fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out.name}")

def _chip(idx):
    j = V/"chips"/f"chip_{idx:05d}.jpg"
    return np.asarray(Image.open(j)) if j.exists() else \
           np.load(V/"chips"/f"chip_{idx:05d}.npz")["img"][:3].transpose(1,2,0).clip(0,255).astype(np.uint8)

# ── 1. canonicals (RGB + NIR for 8 hulls)
def fig_canonicals():
    fig, ax = plt.subplots(2, 8, figsize=(13, 4))
    for j, b in enumerate(BOATS):
        ax[0,j].imshow(np.asarray(Image.open(A/f"canonical_{b}_4b_preview.png")))
        ax[0,j].set_title(b.replace("boat_","b").replace("boat","b"), fontsize=10); ax[0,j].axis("off")
        ax[1,j].imshow(np.asarray(Image.open(A/f"canonical_{b}_4b_nir_preview.png")), cmap="gray"); ax[1,j].axis("off")
    fig.suptitle("8 hand-digitised NAIP vessel hulls — RGB (top) + native NIR (bottom)", y=1.02, fontsize=11)
    fig.tight_layout(); _save(fig, "canonicals")

# ── 2. perturb — 32 random samples from the dataset
def fig_perturb():
    df = pd.read_parquet(V/"variants_meta.parquet").reset_index(drop=True)
    rng = np.random.default_rng(7)
    pools = (df[df.source_id.isin(BOATS) & df.n_boats.isna()].index.to_numpy(),
             df[df.source_id.isin(OBL)].index.to_numpy(),
             df[df.source_id.isin(BOATS) & df.n_boats.notna()].index.to_numpy(),
             df[df.source_id.str.startswith("null_")].index.to_numpy())
    sizes = (18, 4, 6, 4)
    picks = np.concatenate([rng.choice(p, s, replace=False) for p, s in zip(pools, sizes)])
    rng.shuffle(picks); picks = picks[:32]
    fig, axes = plt.subplots(4, 8, figsize=(13.6, 7.8))
    for i, ax in enumerate(axes.flat):
        if i >= len(picks): ax.axis("off"); continue
        r = df.iloc[picks[i]]; ax.imshow(_chip(int(r.chip_idx))); ax.set_xticks([]); ax.set_yticks([])
        bits = [str(r.source_id).replace("boat","b").replace("oblique_","obl ")]
        if pd.notna(r.scale): bits.append(f"s={r.scale:.2f}")
        if pd.notna(r.rotation_deg): bits.append(f"{int(r.rotation_deg)}°")
        if isinstance(r.position, str): bits.append(r.position.replace("_"," "))
        if pd.notna(r.n_boats): bits.append(f"N={int(r.n_boats)}")
        ax.set_title(" • ".join(bits), fontsize=7.5, pad=2)
    fig.suptitle("32 random samples from the 1042-chip dataset (seed=7)", fontsize=11, y=1.0)
    fig.tight_layout(); _save(fig, "perturb")

# ── 3. TL;DR within vs LOBO with 95% bootstrap CIs
def fig_tldr():
    p = json.loads((A/"probes_4b.json").read_text())
    keys = ["rotation_4cls","rotation_8bin","scale_R2","count_R2","count_R2_mixed","position_9cls_fixed_scale"]
    labels = ["rotation\n4-cls","rotation\n8-bin","scale\nR²","count R²\nsingle hull","count R²\nmixed hulls","position\n9-cls"]
    chance = [0.25, 0.125, 0, 0, 0, 1/9]
    within = [p[k].get("acc", p[k].get("R2")) for k in keys]
    lobo = [p["lobo_bootstrap_ci"][k]["mean"] if k in p["lobo_bootstrap_ci"] else np.nan for k in keys]
    err = [[max(0, p["lobo_bootstrap_ci"][k]["mean"]-p["lobo_bootstrap_ci"][k]["ci_lo_95"]) if k in p["lobo_bootstrap_ci"] else 0,
            max(0, p["lobo_bootstrap_ci"][k]["ci_hi_95"]-p["lobo_bootstrap_ci"][k]["mean"]) if k in p["lobo_bootstrap_ci"] else 0]
           for k in keys]
    err = list(zip(*err))
    x = np.arange(len(keys)); w = 0.36
    fig, ax = plt.subplots(figsize=(11, 4.6))
    b1 = ax.bar(x-w/2, within, w, color="#4c78a8", label="within-object 5-fold")
    b2 = ax.bar(x+w/2, lobo, w, color=PRB, yerr=err, capsize=4, ecolor="#333", label="LOBO (held-out hull)")
    for xi, c in zip(x, chance): ax.hlines(c, xi-w*1.05, xi+w*1.05, colors="#888", linestyles=":", lw=1)
    for r, v in zip(b1, within): ax.text(r.get_x()+r.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontsize=8.5)
    for r, v in zip(b2, lobo):
        if not np.isnan(v): ax.text(r.get_x()+r.get_width()/2, v+0.05, f"{v:.2f}", ha="center", fontsize=8.5, color="#a64500")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(-0.15, 1.12); ax.set_ylabel("score (R² or accuracy)")
    ax.set_title("Within-object 5-fold (blue) vs LOBO (orange, 95% bootstrap CI) — 4-band, 8 hulls")
    ax.legend(loc="lower left"); fig.tight_layout(); _save(fig, "tldr")

# ── 4. LOBO per-hull strip plot
def fig_lobo_strip():
    p = json.loads((A/"probes_4b.json").read_text())
    keys = ["rotation_4cls","rotation_8bin","scale_R2","count_R2","position_9cls_fixed_scale"]
    pretty = {"rotation_4cls":"rotation 4-cls","rotation_8bin":"rotation 8-bin","scale_R2":"scale R²",
              "count_R2":"count R² (single)","position_9cls_fixed_scale":"position 9-cls"}
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    for ax, k in zip(axes, keys):
        per = dict(p["lobo_per_hull"][k]); ys = [per[b] for b in BOATS]; xs = np.arange(len(BOATS))
        ax.scatter(xs, ys, c=PRB, s=42, zorder=3, edgecolor="#a64500", lw=0.8)
        ax.axhline(np.mean(ys), color="#444", lw=1, ls="--")
        ax.set_xticks(xs); ax.set_xticklabels([b.replace("boat","b") for b in BOATS], rotation=45, ha="right", fontsize=8)
        ax.set_title(pretty[k], fontsize=10)
        ax.text(0.97, 0.04, f"μ={np.mean(ys):.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="#ddd", pad=2))
    fig.suptitle("LOBO per-hull spread — each dot is the held-out hull (n=8)", y=1.02, fontsize=11)
    fig.tight_layout(); _save(fig, "lobo_strip")

# ── 5. hull similarity matrix
def fig_hullsim():
    h = json.loads((A/"probes_4b.json").read_text())["hull_similarity"]
    M = np.array(h["cosine_matrix"]); names = [b.replace("boat","b") for b in h["boat_names"]]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=0.94, vmax=1.0)
    for i, j in np.ndindex(M.shape):
        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if M[i,j]>0.97 else "#222")
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9); ax.set_yticklabels(names, fontsize=9)
    ax.set_title(f"Canonical hull cosines — off-diag mean {h['off_diag_mean']:.3f}", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout(); _save(fig, "hullsim")

# ── 6. continuous rotation
def fig_rotcont():
    r = json.loads((A/"probes_4b.json").read_text())["rotation_continuous"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].bar(["within\n5-fold","LOBO\nheld-out hull"], [r["within_mae_deg"], r["lobo_mae_deg"]],
              color=["#4c78a8", PRB], edgecolor="#333")
    for i, v in enumerate([r["within_mae_deg"], r["lobo_mae_deg"]]):
        ax[0].text(i, v+2, f"{v:.1f}°", ha="center")
    ax[0].axhline(r["chance_random_mae_deg"], ls=":", color="#888", lw=1, label="chance (random)")
    ax[0].set_ylabel("mean absolute angular error (°)")
    ax[0].set_title(f"continuous rotation regression\nn={r['n']} chips, sin/cos targets")
    ax[0].legend(); ax[0].set_ylim(0, max(95, r["lobo_mae_deg"]*1.15))
    per = r["lobo_per_hull"]; vals = [v for _,v in per]
    ax[1].bar(range(len(per)), vals, color=PRB, edgecolor="#a64500")
    ax[1].axhline(np.mean(vals), ls="--", color="#444", lw=1)
    ax[1].set_xticks(range(len(per))); ax[1].set_xticklabels([b.replace("boat","b") for b,_ in per], rotation=45, ha="right", fontsize=9)
    ax[1].set_title("LOBO per-hull MAE (°)")
    fig.tight_layout(); _save(fig, "rotcont")

# ── 7. binding 3-panel
def fig_binding():
    p = json.loads((A/"binding_4b.json").read_text())
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    ax[0].bar(["SWAP\n(A↔B)","noise floor\n(A re-seeded)"], [p["swap_cos_mean"], p["noise_cos_mean"]],
              yerr=[p["swap_cos_std"], p["noise_cos_std"]], capsize=6, color=["#d62728","#888"], edgecolor="#222")
    for i, m in enumerate([p["swap_cos_mean"], p["noise_cos_mean"]]):
        ax[0].text(i, m+0.0005, f"{m:.4f}", ha="center", fontsize=9)
    ax[0].set_ylim(0.997, 1.0); ax[0].set_ylabel("cosine similarity")
    ax[0].set_title("cosine cannot tell the swap apart")
    ax[1].bar(["binding\nprobe","shuffled\nmean"], [p["binding_probe_acc"], p["binding_probe_perm_mean"]],
              color=["#2ca02c","#bbb"], edgecolor="#222")
    ax[1].text(0, p["binding_probe_acc"]+0.02, f"{p['binding_probe_acc']:.2f}", ha="center")
    ax[1].text(1, p["binding_probe_perm_mean"]+0.02, f"{p['binding_probe_perm_mean']:.2f}", ha="center")
    ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("accuracy")
    ax[1].set_title(f"linear probe recovers binding\n(perm-p={p['binding_probe_perm_p']:.3f})")
    v = p["delta_consistency_mean_pairwise_cos"]
    ax[2].bar(["mean pairwise\ncos(Δ_i, Δ_j)"], [v], color="#9467bd", edgecolor="#222")
    ax[2].text(0, v+0.01, f"{v:.3f}", ha="center"); ax[2].set_ylim(-0.05, 1.0)
    ax[2].set_title("Δ direction partially shared\nacross swap pairs")
    fig.tight_layout(); _save(fig, "binding")

# ── 8. cosine top-5 retrieval split
def fig_retrieval():
    df = pd.read_parquet(V/"variants_meta.parquet").reset_index(drop=True)
    emb = np.load(V/"embeddings.npy"); Mn = emb / (np.linalg.norm(emb, axis=1, keepdims=True)+1e-12)
    rows = []
    for hull in ("boat1", "boat_05"):
        for s in (0.25, 1.00):
            q = df[(df.source_id==hull) & (df.scale==s) & (df.rotation_deg==0) & (df.position=="center")]
            if not len(q): continue
            qi = int(q.iloc[0].chip_idx); sims = Mn @ Mn[qi]; sims[qi] = -np.inf
            rows.append((f"{hull} @ s={s}", qi, np.argsort(-sims)[:5], sims[np.argsort(-sims)[:5]]))
    fig, axes = plt.subplots(len(rows), 6, figsize=(13, 2.0*len(rows)))
    if len(rows) == 1: axes = axes[None,:]
    for i, (label, qi, top5, sims) in enumerate(rows):
        axes[i,0].imshow(_chip(qi)); axes[i,0].set_title("query", fontsize=9); axes[i,0].set_ylabel(label, fontsize=10)
        axes[i,0].set_xticks([]); axes[i,0].set_yticks([])
        for j, (k, s) in enumerate(zip(top5, sims), start=1):
            axes[i,j].imshow(_chip(k)); r = df.iloc[k]
            axes[i,j].set_title(f"#{j}  cos={s:.4f}\n{r.source_id} s={r.scale}", fontsize=8)
            axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
    fig.suptitle("Top-5 cosine neighbours — at scale 0.25 they're near-duplicates; at scale 1.0 neighbours stay at scale 1.0", fontsize=10, y=1.005)
    fig.tight_layout(); _save(fig, "retrieval")

# ── 9. dimensionality
def fig_dims():
    out = json.loads((A/"dims_4b.json").read_text())
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (name, row) in zip(axes, out.items()):
        nx, nv = zip(*row["native_prefix"]); px, pv = (zip(*row["pca_k"]) if row["pca_k"] else ([], []))
        ax.plot(nx, nv, "o-", color="#4c78a8", label="native dims-0:K")
        if px: ax.plot(px, pv, "s--", color=PRB, label="PCA top-K")
        ax.axhline(row["raw_1024"], ls=":", color="#222", lw=1, label=f"raw 1024 ({row['raw_1024']:.2f})")
        ax.axhline(row["chance"], ls=":", color="#888", lw=1, label="chance")
        ax.set_xscale("log", base=2); ax.set_xlabel("K"); ax.set_title(name, fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Probe score vs input dimensionality", fontsize=11, y=1.02); fig.tight_layout(); _save(fig, "dims")

# ── 10/11. polygon retrieval bars (general + specific-boat)
def fig_polygon():
    rows = json.loads((A/"polygon_4b.json").read_text())
    labels = [r["label"] for r in rows]; pv = [r["iou_probe"] for r in rows]; bv = [r["iou_baseline"] for r in rows]
    x = np.arange(len(labels)); w = 0.4
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x-w/2, pv, w, color="#4c78a8", label="ridge probe (5-fold)")
    ax.bar(x+w/2, bv, w, color="#bbb", label="mean-mask baseline")
    for i, (p, b) in enumerate(zip(pv, bv)):
        ax.text(i-w/2, p+0.005, f"{p:.2f}", ha="center", fontsize=8)
        ax.text(i+w/2, b+0.005, f"{b:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("soft IoU @ 32×32"); ax.set_ylim(0, 0.5)
    ax.set_title("Polygon-from-CLS — single 5× baseline; multi-N maintains 2–3× lift"); ax.legend()
    fig.tight_layout(); _save(fig, "polygon")

def fig_polygon_specific():
    rows = json.loads((A/"polygon_specific_4b.json").read_text())
    labels = [r["N"] for r in rows]; x = np.arange(len(labels)); w = 0.27
    pv = [r["iou_probe"] for r in rows]; bv = [r["iou_baseline"] for r in rows]; ov = [r["iou_oracle_union"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x-w, pv, w, color=PRB, label="ridge probe (target = leftmost)")
    ax.bar(x,   bv, w, color="#bbb", label="mean-mask baseline")
    ax.bar(x+w, ov, w, color="#9467bd", label="union-oracle ceiling (1/N)")
    for xi, p, b, o in zip(x, pv, bv, ov):
        ax.text(xi-w, p+0.01, f"{p:.2f}", ha="center", fontsize=7.5)
        ax.text(xi,   b+0.01, f"{b:.2f}", ha="center", fontsize=7.5)
        ax.text(xi+w, o+0.01, f"{o:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_xlabel("N boats in chip")
    ax.set_ylabel("soft IoU vs leftmost target"); ax.set_ylim(0, max(0.4, max(pv+bv+ov)*1.15))
    ax.set_title("Specific-boat — probe lifts above baseline but caps at 1/N oracle ceiling"); ax.legend(loc="upper right")
    fig.tight_layout(); _save(fig, "polygon_specific")

# ── 12. polygon visual: chip / GT / prediction
def fig_polygon_visual():
    df = pd.read_parquet(V/"variants_meta.parquet").reset_index(drop=True)
    emb = np.load(V/"embeddings.npy")
    alphas = {b: np.load(A/f"canonical_{b}_4b.npz")["img"][4].astype(np.uint8) for b in BOATS}
    cands = df[df.source_id.isin(BOATS+OBL)].reset_index(drop=True)
    G = 32; cell = CHIP // G
    Y = np.zeros((len(cands), 1024), np.float32)
    from pipeline import _gt_mask
    for i, r in cands.iterrows():
        m, _ = _gt_mask(r, alphas, G); Y[i] = m.flatten()
    # 5-fold ridge + soft IoU
    Xs = emb[cands.chip_idx.to_numpy()]; kf = KFold(5, shuffle=True, random_state=0); P = np.zeros_like(Y)
    for tr, te in kf.split(Xs): P[te] = Ridge(alpha=1.0).fit(Xs[tr], Y[tr]).predict(Xs[te])
    P = P.clip(0, 1)
    iou = (np.minimum(P,Y).sum(1)/(np.maximum(P,Y).sum(1)+1e-6))
    def pick(mask, label):
        idx = np.where(mask)[0];  return None if not len(idx) else (idx[np.argsort(iou[idx])[len(idx)//2]], label)
    picks = [pick(cands.source_id.isin(BOATS) & cands.n_boats.isna(), "single"),
             pick(cands.source_id.str.startswith("oblique_"), "oblique"),
             pick(cands.source_id.isin(BOATS) & (cands.n_boats==3), "multi N=3"),
             pick(cands.source_id.isin(BOATS) & (cands.n_boats==8), "multi N=8")]
    picks = [p for p in picks if p]
    fig, axes = plt.subplots(3, len(picks), figsize=(2.2*len(picks), 6.6))
    for col, (i, label) in enumerate(picks):
        ci = int(cands.iloc[i].chip_idx)
        axes[0,col].imshow(_chip(ci)); axes[0,col].set_title(f"{label} • IoU={iou[i]:.2f}", fontsize=9.5)
        axes[1,col].imshow(Y[i].reshape(G,G), cmap="magma", vmin=0, vmax=1)
        axes[2,col].imshow(P[i].reshape(G,G), cmap="magma", vmin=0, vmax=1)
        for r in range(3): axes[r,col].set_xticks([]); axes[r,col].set_yticks([])
    for r, lbl in enumerate(["chip\n(RGB)", "ground truth\n(32×32)", "probe pred\n(32×32 soft)"]):
        axes[r,0].set_ylabel(lbl, fontsize=10, labelpad=4)
    fig.suptitle("Polygon-from-CLS — chip / GT mask / prediction (median-IoU per stratum)", fontsize=11, y=1.0)
    fig.tight_layout(); _save(fig, "polygon_visual")

# ── 13. confidence: distance-to-pool vs LOBO error
def fig_confidence():
    df = pd.read_parquet(V/"variants_meta.parquet").reset_index(drop=True)
    emb = np.load(V/"embeddings.npy")
    p = json.loads((A/"probes_4b.json").read_text())
    perh_count = dict(p["lobo_per_hull"]["count_R2"]); perh_rot = dict(p["lobo_per_hull"]["rotation_4cls"])
    canon = {b: emb[df[(df.source_id==b)&(df.scale==0.25)&(df.rotation_deg==0)&(df.position=="center")].chip_idx.iloc[0]] for b in BOATS}
    rows = []
    for held in BOATS:
        mu = np.stack([canon[b] for b in BOATS if b != held]).mean(0)
        rows.append((held, float(np.linalg.norm(canon[held]-mu)),
                      perh_count.get(held, np.nan), perh_rot.get(held, np.nan)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, ylabel, yi in [(axes[0], "count", "count R² (held-out)", 2),
                                   (axes[1], "rot", "rotation 4-cls (held-out)", 3)]:
        d = [r[1] for r in rows]; ys = [r[yi] for r in rows]
        ax.scatter(d, ys, s=70, color=PRB, edgecolor="#333")
        for di, vi, ni in zip(d, ys, [r[0] for r in rows]):
            ax.annotate(ni.replace("boat","b"), (di, vi), xytext=(4,4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("L2 distance: held-out canonical → mean of other 7"); ax.set_ylabel(ylabel)
        v = [(di, vi) for di, vi in zip(d, ys) if not np.isnan(vi)]
        if len(v) > 2: ax.set_title(f"Pearson r = {np.corrcoef(*zip(*v))[0,1]:+.2f}")
    fig.suptitle("Per-hull confidence proxy — distance-to-training-pool vs LOBO probe score", fontsize=11, y=1.02)
    fig.tight_layout(); _save(fig, "confidence")

# ── 14. cosine NN vs probe — bars + matrix
def _strip_panel(ax, cos, prb, *, chance=None, ceiling=None, ylim=(-0.2,1.05), title=""):
    cm, pm = float(np.nanmean(cos)), float(np.nanmean(prb))
    ax.bar([0],[cm], 0.55, color=COS, edgecolor="#1f6fb4", alpha=0.7, label=f"cosine top-5 NN  μ={cm:.2f}")
    ax.bar([1],[pm], 0.55, color=PRB, edgecolor="#a64500", alpha=0.7, label=f"linear probe  μ={pm:.2f}")
    rng = np.random.default_rng(0)
    ax.scatter(rng.uniform(-0.10,0.10, len(cos)), cos, color="#1f6fb4", s=42, zorder=3, edgecolor="white", lw=0.7)
    ax.scatter(1+rng.uniform(-0.10,0.10, len(prb)), prb, color="#a64500", s=42, zorder=3, edgecolor="white", lw=0.7)
    if chance is not None: ax.axhline(chance, color="#999", ls=":", lw=1, label=f"chance ({chance:.2f})")
    if ceiling is not None: ax.axhline(ceiling, color="#444", ls="--", lw=0.9, alpha=0.7, label=f"within-object ({ceiling:.2f})")
    ax.set_xticks([0,1]); ax.set_xticklabels(["cosine NN","linear probe"]); ax.set_xlim(-0.5,1.5)
    ax.set_ylim(*ylim); ax.set_ylabel("LOBO score"); ax.set_title(title, fontsize=10)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.92)

def fig_cosvsprobe():
    cvp = json.loads((A/"cos_vs_probe_4b.json").read_text())
    keys = ["rotation 4-cls","rotation 8-bin","scale R²","count R² single","position 9-cls"]
    cm  = [cvp[k]["cos_mean"]   for k in keys]
    pm  = [cvp[k]["probe_mean"] for k in keys]
    chance = [0.25, 0.125, 0, 0, 1/9]
    x = np.arange(len(keys)); w = 0.36
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(x-w/2, cm, w, color=COS, edgecolor="#1f6fb4", label="cosine top-5 NN (LOBO)")
    ax.bar(x+w/2, pm, w, color=PRB, edgecolor="#a64500", label="linear probe (LOBO)")
    for xi, c in zip(x, chance): ax.hlines(c, xi-w*1.05, xi+w*1.05, colors="#888", linestyles=":", lw=1)
    for xi, v in zip(x, cm): ax.text(xi-w/2, max(v,-0.15)+0.02, f"{v:.2f}", ha="center", fontsize=8.5, color="#1f6fb4")
    for xi, v in zip(x, pm): ax.text(xi+w/2, v+0.05, f"{v:.2f}", ha="center", fontsize=8.5, color="#a64500")
    ax.set_xticks(x); ax.set_xticklabels(keys); ax.set_ylim(-0.25, 1.12); ax.set_ylabel("score (acc / R²)")
    ax.set_title("Cosine top-5 NN vs linear probe — same 1024-D CLS, same LOBO split"); ax.legend(loc="lower left")
    fig.tight_layout(); _save(fig, "cos_vs_probe")

def fig_cosvsprobe_matrix():
    cvp = json.loads((A/"cos_vs_probe_4b.json").read_text())
    keys = ["rotation 4-cls","rotation 8-bin","scale R²","count R² single","position 9-cls"]
    Mc = np.full((len(keys), len(BOATS)), np.nan); Mp = Mc.copy()
    for i, k in enumerate(keys):
        for h, v in cvp[k]["cos_per_hull"]:
            if h in BOATS: Mc[i, BOATS.index(h)] = v
        for h, v in cvp[k]["probe_per_hull"]:
            if h in BOATS: Mp[i, BOATS.index(h)] = v
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    vmin = max(min(np.nanmin(Mc), np.nanmin(Mp)), -0.2); vmax = max(np.nanmax(Mc), np.nanmax(Mp))
    for ax, M, t in [(axes[0], Mc, "Cosine top-5 NN (LOBO)"), (axes[1], Mp, "Linear probe (LOBO)")]:
        im = ax.imshow(M, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
        for i, j in np.ndindex(M.shape):
            if not np.isnan(M[i,j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if M[i,j]<vmin+0.25 or M[i,j]>vmax-0.15 else "#222")
        ax.set_xticks(range(len(BOATS))); ax.set_xticklabels([b.replace("boat","b") for b in BOATS], rotation=45, ha="right")
        ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys); ax.set_title(t)
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.suptitle("Per-(probe × held-out hull) — same 1024-D CLS, same LOBO split", fontsize=12, y=1.02)
    fig.tight_layout(); _save(fig, "cos_vs_probe_matrix")

# ── 15. paired-probe trios (cosine-accessible / probe-accessible / agent-only)
def fig_pairs():
    cvp = json.loads((A/"cos_vs_probe_4b.json").read_text())
    pp  = json.loads((A/"probes_4b.json").read_text())
    binding = json.loads((A/"binding_4b.json").read_text())
    spec = json.loads((A/"polygon_specific_4b.json").read_text())
    # cosine-accessible: scale + count
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for axi, key, title, ylim in [(ax[0], "scale R²", "Scale R² — cosine 0.98 vs probe 0.94", (0.5, 1.05)),
                                     (ax[1], "count R² single", "Count R² — cosine 0.90 vs probe 0.84", (0.4, 1.05))]:
        c = [v for _,v in cvp[key]["cos_per_hull"]]; p = [v for _,v in cvp[key]["probe_per_hull"]]
        ceil = pp["scale_R2"]["R2"] if "scale" in key else pp["count_R2"]["R2"]
        _strip_panel(axi, c, p, ceiling=ceil, ylim=ylim, title=title)
    fig.suptitle("Cosine-accessible — both readouts work; cosine slightly tighter", y=1.02); fig.tight_layout()
    _save(fig, "pair_cosine")
    # probe-accessible: rotation + binding
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    c = [v for _,v in cvp["rotation 4-cls"]["cos_per_hull"]]
    p = [v for _,v in cvp["rotation 4-cls"]["probe_per_hull"]]
    _strip_panel(ax[0], c, p, chance=0.25, ceiling=pp["rotation_4cls"]["acc"], ylim=(0,1.05),
                  title="Rotation 4-cls  —  cosine 0.50 vs probe 0.76")
    cos_acc, prb_acc = 0.50, binding["binding_probe_acc"]
    ax[1].bar([0],[cos_acc], 0.55, color=COS, edgecolor="#1f6fb4", alpha=0.7, label=f"cosine NN  μ={cos_acc:.2f}")
    ax[1].bar([1],[prb_acc], 0.55, color=PRB, edgecolor="#a64500", alpha=0.7, label=f"linear probe  μ={prb_acc:.2f}")
    ax[1].scatter([1],[binding["binding_probe_perm_mean"]], color="#666", s=70, marker="x", lw=2, zorder=4,
                   label=f"perm shuffle ({binding['binding_probe_perm_mean']:.2f})")
    ax[1].axhline(0.5, color="#999", ls=":", lw=1, label="chance (0.50)")
    ax[1].set_xticks([0,1]); ax[1].set_xticklabels(["cosine NN","linear probe"])
    ax[1].set_xlim(-0.5,1.5); ax[1].set_ylim(0,1.05)
    ax[1].set_title(f"Binding — cosine 0.50 vs probe {prb_acc:.2f}\ncos(SWAP)≈cos(noise)≈0.999")
    ax[1].legend(loc="lower left", fontsize=8)
    fig.suptitle("Probe-accessible — cosine collapses; probe lifts to 0.76 / 0.87", y=1.02); fig.tight_layout()
    _save(fig, "pair_probe")
    # agent-only: position + per-instance addressing
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    c = [v for _,v in cvp["position 9-cls"]["cos_per_hull"]]
    p = [v for _,v in cvp["position 9-cls"]["probe_per_hull"]]
    _strip_panel(ax[0], c, p, chance=1/9, ceiling=pp["position_9cls_fixed_scale"]["acc"], ylim=(0,1.0),
                  title="Position 9-cls — cosine 0.27 vs probe 0.24 (both ≈ chance)")
    Ns = [r["N"] for r in spec]; pv = [r["iou_probe"] for r in spec]
    bv = [r["iou_baseline"] for r in spec]; ov = [r["iou_oracle_union"] for r in spec]
    x = np.arange(len(Ns)); w = 0.27
    ax[1].bar(x-w, pv, w, color=PRB, edgecolor="#a64500", alpha=0.85, label="probe")
    ax[1].bar(x,   bv, w, color="#bbb", edgecolor="#666", alpha=0.85, label="baseline")
    ax[1].bar(x+w, ov, w, color="#9467bd", edgecolor="#5b3088", alpha=0.85, label="oracle (1/N)")
    for xi, p_, b_, o_ in zip(x, pv, bv, ov):
        ax[1].text(xi-w, p_+0.01, f"{p_:.2f}", ha="center", fontsize=7.5)
        ax[1].text(xi,   b_+0.01, f"{b_:.2f}", ha="center", fontsize=7.5)
        ax[1].text(xi+w, o_+0.01, f"{o_:.2f}", ha="center", fontsize=7.5)
    ax[1].set_xticks(x); ax[1].set_xticklabels(Ns); ax[1].set_xlabel("N")
    ax[1].set_ylabel("soft IoU vs leftmost"); ax[1].set_ylim(0, max(0.4, max(pv+bv+ov)*1.15))
    ax[1].set_title("Per-instance addressing — caps at 1/N oracle"); ax[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Agent-only — neither readout transfers; agent has to look at patches/pixels", y=1.02); fig.tight_layout()
    _save(fig, "pair_agent")

# ── 16. ELLE 2-panel — manifold gauge + per-CHIP confidence
def fig_elle():
    """Two panels: (a) ELLE z by stratum (manifold gauge — strong);
    (b) per-chip ELLE z vs per-chip probe error (per-chip confidence signal —
    strong on rotation, moderate on scale, useless on count)."""
    import torch
    from sklearn.linear_model import Ridge as _R, LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline as _mp
    from sklearn.preprocessing import StandardScaler as _SS
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold

    df = pd.read_parquet(V/"variants_meta.parquet").reset_index(drop=True)
    emb = np.load(V/"embeddings.npy")
    pairs = HERE.parent.parent.parent / "elle/data/clay_naip_pairs.pt"
    if not pairs.exists(): pairs = Path.home()/"code/lgnd/elle/data/clay_naip_pairs.pt"
    d = torch.load(pairs, map_location="cpu", weights_only=False)
    head = _mp(_SS(), _R(alpha=1.0)).fit(d["cls_emb"].numpy().astype(np.float32),
                                          d["loss"].numpy().astype(np.float32))
    z = (head.predict(emb) - d["loss"].numpy().mean()) / d["loss"].numpy().std()
    def stratum(s):
        return ("oblique" if s.startswith("oblique_") else
                "single hull" if s in BOATS else
                "mixed cardinality" if s == "mixed_boats" else
                "adversarial" if s == "adv" else
                f"null • {s.replace('null_','').replace('_',' ')}" if s.startswith("null_") else "other")
    sx = df.source_id.map(stratum)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # ── (a) ELLE z by stratum
    order = ["null • noise","null • patch density","adversarial","mixed cardinality","single hull","oblique","null • water"]
    order = [s for s in order if (sx == s).any()]
    parts = axes[0].violinplot([z[sx==s] for s in order], positions=range(len(order)), showmedians=True, widths=0.85)
    for body, s in zip(parts["bodies"], order):
        body.set_facecolor("#d62728" if ("noise" in s or "patch" in s) else "#aaa" if ("water" in s or s=="adversarial") else "#2ca02c")
        body.set_alpha(0.55); body.set_edgecolor("#333")
    axes[0].axhline(0, ls="--", color="#444", lw=1, label="typical-NAIP (z=0)")
    axes[0].axhline(2, ls=":", color="#888", lw=1); axes[0].axhline(-2, ls=":", color="#888", lw=1)
    axes[0].set_xticks(range(len(order))); axes[0].set_xticklabels(order, rotation=20, ha="right", fontsize=8.5)
    axes[0].set_ylabel("ELLE-loss z (vs typical NAIP)")
    axes[0].set_title("(a) Manifold gauge: composites land on-manifold,\nGaussian noise sits >0.5σ above"); axes[0].legend(fontsize=8)

    # ── (b) per-chip ELLE z vs per-chip probe error
    # Use 5-fold CV predictions for each probe on its filter slice.
    def clf(): return _mp(_SS(), LinearSVC(C=1, dual=True, max_iter=3000, class_weight="balanced"))
    def reg(): return _mp(_SS(), _R(alpha=1.0))
    panels = []
    # rotation 4-cls (classification)
    f, lab, kind, _ = (lambda d: d.source_id.isin(BOATS) & d.rotation_deg.isin(ROT_CARD) & d.scale.notna() & d.n_boats.isna()), \
                       (lambda s: s.rotation_deg.astype(int).to_numpy()), "clf", 0.25
    sub = df[f(df)]; idx = sub.index.to_numpy()
    yhat = cross_val_predict(clf(), emb[sub.chip_idx.to_numpy()], lab(sub),
                              cv=StratifiedKFold(5, shuffle=True, random_state=0))
    err_rot = (yhat != lab(sub)).astype(int)
    panels.append(("rotation 4-cls", z[idx], err_rot, "clf",
                   float(np.corrcoef(z[idx], err_rot)[0,1]),
                   float(roc_auc_score(err_rot, z[idx])) if err_rot.sum() > 0 else float("nan")))
    # scale R² (regression)
    f, lab = (lambda d: d.source_id.isin(BOATS) & d.scale.notna() & d.n_boats.isna()), \
             (lambda s: np.log(s.scale.astype(float).to_numpy()))
    sub = df[f(df)]; idx = sub.index.to_numpy()
    yhat = cross_val_predict(reg(), emb[sub.chip_idx.to_numpy()], lab(sub),
                              cv=KFold(5, shuffle=True, random_state=0))
    err_sca = np.abs(yhat - lab(sub))
    panels.append(("scale R²", z[idx], err_sca, "reg",
                   float(np.corrcoef(z[idx], err_sca)[0,1]),
                   float(roc_auc_score((err_sca > np.median(err_sca)).astype(int), z[idx]))))

    ax = axes[1]
    for label, zi, ei, kind, r, auc in panels:
        kw = {"alpha":0.45, "s":18, "edgecolor":"none"}
        if kind == "clf":
            ax.scatter(zi[ei==0], np.zeros((ei==0).sum())+0.05, color="#2ca02c", label=f"{label} ✓ (correct, n={int((ei==0).sum())})", **kw)
            ax.scatter(zi[ei==1], np.zeros((ei==1).sum())+0.95, color="#d62728", label=f"{label} ✗ (wrong, n={int((ei==1).sum())})  AUROC={auc:.2f}", **kw)
        else:
            # normalize residual to [0,1] for the same axis
            en = (ei - ei.min()) / (ei.max() - ei.min() + 1e-9)
            ax.scatter(zi, en, color="#1f77b4", label=f"{label} |residual| (norm.)  Pearson r={r:+.2f}", **kw)
    ax.set_xlabel("ELLE-loss z-score (per chip)")
    ax.set_ylabel("rotation: 0=correct, 1=wrong   |   scale: |residual| normalised")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(b) Per-chip confidence: ELLE z predicts rotation mistakes\n(AUROC 0.88) and tracks scale residual (r=+0.41)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)

    fig.suptitle("ELLE — manifold gauge (a) + per-chip confidence signal (b)", y=1.02, fontsize=12)
    fig.tight_layout(); _save(fig, "elle")

# ── CLI
ALL = {"canonicals":fig_canonicals, "perturb":fig_perturb, "tldr":fig_tldr,
       "lobo_strip":fig_lobo_strip, "hullsim":fig_hullsim, "rotcont":fig_rotcont,
       "binding":fig_binding, "retrieval":fig_retrieval, "dims":fig_dims,
       "polygon":fig_polygon, "polygon_specific":fig_polygon_specific,
       "polygon_visual":fig_polygon_visual, "confidence":fig_confidence,
       "cosvsprobe":fig_cosvsprobe, "cosvsprobe_matrix":fig_cosvsprobe_matrix,
       "pairs":fig_pairs, }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", default="all", nargs="?")
    a = ap.parse_args()
    if a.which == "all":
        for fn in ALL.values(): fn()
    else: ALL[a.which]()

if __name__ == "__main__": main()
