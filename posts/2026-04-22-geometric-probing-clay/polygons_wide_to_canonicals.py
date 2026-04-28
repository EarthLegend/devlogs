"""polygons_wide_to_canonicals.py — extract N canonical boat cutouts from
hand-digitised polygons drawn on assets/anchorage_wide_4b.tif.

Workflow:
  1. Open assets/anchorage_wide_4b.tif in QGIS (4-band, EPSG:26910).
  2. Create a polygon layer in the same CRS, digitise each boat's hull.
  3. Add a 'name' or 'id' attribute to each polygon. Suggested:
        boat_a, boat_b, boat_c, ... (alphabetic, no whitespace)
     If you skip the attribute, polygons get auto-named by centroid
     (boat_01, boat_02, …, ordered north-to-south then west-to-east).
  4. Save as assets/boats_wide.geojson (right-click → Export → Save Features
     As → GeoJSON).
  5. Run: python polygons_wide_to_canonicals.py
     → for each polygon, crops a tight 4-band window around the boat,
       rasterises the polygon as the alpha mask, two-pass-PCA-rotates to
       horizontal, smooths, and writes:
         assets/canonical_<name>_4b.npz
         assets/canonical_<name>_4b_preview.png
         assets/canonical_<name>_4b_nir_preview.png

Each canonical is then a drop-in replacement for the boat1/boat2 mask
in phase_a_4b.py / binding_4b.py / widget_xprod.py — re-run those with
the broader CANONICALS dict and you get a multi-boat dataset.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from PIL import Image
from skimage.transform import rotate as sk_rotate
from scipy.ndimage import binary_fill_holes, gaussian_filter

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
PAD_PX = 60        # pixel pad around each polygon when cropping the window

# All boat polygons live in a single boats.geojson. Each feature's
# properties.source records which legacy file it was traced into:
#   "boats"     — narrow per-boat TIFFs; already extracted as canonical_boat{1,2}_4b
#   "moreboats" — west strip (anchorage_wide_4b.tif)
#   "evemore"   — covers both strips
# We don't trust the source label for raster routing — instead we pick
# whichever candidate raster's WGS84 bbox contains the polygon centroid.
RASTER_CANDIDATES = [
    "anchorage_wide_4b.tif",
    "anchorage_wide_east_4b.tif",
    "boat1_native_4b.tif",
    "boat2_native_4b.tif",
]


def _pca_angle(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return 0.0
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    pts -= pts.mean(axis=0)
    cov = np.cov(pts.T)
    _, eigvecs = np.linalg.eigh(cov)
    p = eigvecs[:, -1]
    return float(np.degrees(np.arctan2(p[1], p[0])))


def _apply_rotation(rgbn: np.ndarray, mask: np.ndarray, deg: float):
    chans = []
    for c in range(rgbn.shape[0]):
        rot = sk_rotate(rgbn[c].astype(np.float32) / 255.0, deg,
                         resize=True, mode="constant", cval=0.0, order=3)
        chans.append((rot * 255).clip(0, 255).astype(np.uint8))
    rgbn_out = np.stack(chans, axis=0)
    mask_out = sk_rotate(mask.astype(np.float32), deg, resize=True,
                          mode="constant", cval=0.0, order=0) > 0.5
    return rgbn_out, mask_out


def _straighten(rgbn, mask):
    a1 = _pca_angle(mask)
    rot1 = a1
    while rot1 > 90: rot1 -= 180
    while rot1 <= -90: rot1 += 180
    rgbn1, mask1 = _apply_rotation(rgbn, mask, rot1)
    a2 = _pca_angle(mask1)
    while a2 > 90: a2 -= 180
    while a2 <= -90: a2 += 180
    rgbn2, mask2 = _apply_rotation(rgbn1, mask1, a2)
    ys, xs = np.where(mask2)
    if len(ys) > 10 and (ys.max() - ys.min()) > (xs.max() - xs.min()):
        rgbn2, mask2 = _apply_rotation(rgbn2, mask2, 90.0)
    return rgbn2, mask2


def _tight_crop_5ch(rgbn, mask, pad: int = 4):
    ys, xs = np.where(mask)
    if not len(ys):
        return np.zeros((5, 8, 8), dtype=np.uint8)
    y0 = max(0, ys.min() - pad); y1 = min(mask.shape[0], ys.max() + 1 + pad)
    x0 = max(0, xs.min() - pad); x1 = min(mask.shape[1], xs.max() + 1 + pad)
    rgbn_c = rgbn[:, y0:y1, x0:x1]
    mask_c = mask[y0:y1, x0:x1]
    alpha = (mask_c.astype(np.uint8) * 255)
    return np.concatenate([rgbn_c, alpha[None]], axis=0)


def _smooth_alpha(cutout5, sigma=1.5):
    out = cutout5.copy()
    a = out[4] > 0
    a = binary_fill_holes(a)
    a = gaussian_filter(a.astype(np.float32), sigma=sigma) > 0.5
    a = binary_fill_holes(a)
    out[4] = a.astype(np.uint8) * 255
    return out


def _name_for(idx: int, props: dict, source_label: str) -> str:
    """Globally unique name. Polygons originally traced into boats.geojson
    (source='boats') correspond to canonical_boat{1,2}_4b.npz (the oil tanker
    and the bulk carrier, hand-named); they're already extracted from the
    narrow per-boat TIFFs and skipped here. Everything else gets sequential
    boat_03, boat_04, ... by its index in the deduped pool."""
    if source_label == "boats":
        return ""    # signal caller to skip — already extracted as canonical_boat{1,2}_4b
    return f"boat_{idx + 1:02d}"


def _polygon_pixel_bounds(geom, src_crs, src_transform):
    """Reproject geom to src CRS, project ring vertices to pixel coords,
    return (col_min, row_min, col_max, row_max) inclusive."""
    proj = transform_geom("EPSG:4326", src_crs.to_string(), geom)
    if proj["type"] == "Polygon":
        rings = proj["coordinates"]
    elif proj["type"] == "MultiPolygon":
        rings = [proj["coordinates"][0][0]]
    else:
        return None
    cols, rows = [], []
    inv = ~src_transform
    for ring in rings:
        for xy in ring:
            c, r = inv * xy
            cols.append(c); rows.append(r)
    return (int(np.floor(min(cols))), int(np.floor(min(rows))),
            int(np.ceil(max(cols))), int(np.ceil(max(rows))))


def _polygon_centroid_lonlat(g):
    if g["type"] == "MultiPolygon":
        ring = g["coordinates"][0][0]
    elif g["type"] == "Polygon":
        ring = g["coordinates"][0]
    else:
        return None
    xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _haversine_m(p1, p2):
    """Approx great-circle distance in metres for (lon, lat) pairs."""
    import math
    lon1, lat1 = p1; lon2, lat2 = p2
    R = 6371000.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def main():
    pool = []   # list of (geom, props, source_label)
    path = ASSETS / "boats.geojson"
    if not path.exists():
        sys.exit(f"missing {path}")
    gj = json.loads(path.read_text())
    feats = gj["features"] if gj["type"] == "FeatureCollection" else [gj]
    for f in feats:
        props = f.get("properties") or {}
        src = props.get("source", "boats")
        pool.append((f["geometry"], props, src))
    by_src = {}
    for _, _, s in pool: by_src[s] = by_src.get(s, 0) + 1
    print(f"  boats.geojson: {len(feats)} polygons "
          f"({', '.join(f'{k}={v}' for k,v in sorted(by_src.items()))})")
    # Dedupe by centroid distance (< 50 m apart = same boat)
    deduped = []
    for g, props, src in pool:
        c = _polygon_centroid_lonlat(g)
        if c is None: continue
        is_dup = False
        for (g2, _, _) in deduped:
            c2 = _polygon_centroid_lonlat(g2)
            if c2 is None: continue
            if _haversine_m(c, c2) < 50.0:
                is_dup = True; break
        if not is_dup:
            deduped.append((g, props, src))
    print(f"  after dedup: {len(deduped)} unique polygons (was {len(pool)})")
    # Skip polygons already extracted via the narrow per-boat TIFFs (source='boats' → boat1/boat2)
    deduped = [(g, p, s) for (g, p, s) in deduped if s != "boats"]
    print(f"  after skipping source='boats' (already extracted as canonical_boat{{1,2}}_4b): {len(deduped)} polygons")
    # Pre-open every candidate raster + cache its WGS84 bounds for fast
    # polygon-→-raster routing
    raster_info = {}
    for r in RASTER_CANDIDATES:
        rp = ASSETS / r
        if rp.exists():
            with rasterio.open(rp) as src:
                from rasterio.warp import transform_bounds
                lon_w, lat_s, lon_e, lat_n = transform_bounds(
                    src.crs, "EPSG:4326",
                    src.bounds.left, src.bounds.bottom,
                    src.bounds.right, src.bounds.top)
                raster_info[r] = (lon_w, lat_s, lon_e, lat_n)
    # Process each polygon: pick the raster whose bbox contains the centroid
    for i, (geom, props, src) in enumerate(deduped):
        name = f"boat_{i + 3:02d}"   # boat_03, boat_04, ...
        c = _polygon_centroid_lonlat(geom)
        if c is None:
            print(f"  [{name}] skip (no centroid)"); continue
        cx, cy = c
        chosen = None
        for r, (lon_w, lat_s, lon_e, lat_n) in raster_info.items():
            if lon_w <= cx <= lon_e and lat_s <= cy <= lat_n:
                chosen = r; break
        if chosen is None:
            print(f"  [{name}] skip — centroid ({cx:.4f}, {cy:.4f}) outside all candidate rasters")
            continue
        tif_path = ASSETS / chosen
        with rasterio.open(tif_path) as src:
            H, W = src.shape
            T = src.transform
            crs = src.crs
            bbox = _polygon_pixel_bounds(geom, crs, T)
            if bbox is None:
                print(f"  [{name}] skip (unsupported geom type {geom['type']})")
                continue
            c0, r0, c1, r1 = bbox
            # Add pad and clip to source extent
            c0 = max(0, c0 - PAD_PX); r0 = max(0, r0 - PAD_PX)
            c1 = min(W, c1 + PAD_PX); r1 = min(H, r1 + PAD_PX)
            if c1 <= c0 or r1 <= r0:
                print(f"  [{name}] skip (bbox empty)"); continue
            from rasterio.windows import Window, transform as window_transform
            win = Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0)
            rgbn = src.read(window=win)             # (4, h, w)
            wt = window_transform(win, T)
            proj = transform_geom("EPSG:4326", crs.to_string(), geom)
            mask = rasterize([(proj, 1)], out_shape=(r1 - r0, c1 - c0),
                              transform=wt, fill=0, dtype="uint8").astype(bool)
            if mask.sum() == 0:
                print(f"  [{name}] skip (mask empty after crop)"); continue
            rgbn_r, mask_r = _straighten(rgbn, mask)
            cutout5 = _tight_crop_5ch(rgbn_r, mask_r, pad=4)
            cutout5 = _smooth_alpha(cutout5, sigma=1.5)
            np.savez_compressed(ASSETS / f"canonical_{name}_4b.npz", img=cutout5)
            rgba = np.concatenate([cutout5[:3], cutout5[4:5]], axis=0).transpose(1, 2, 0)
            Image.fromarray(rgba, mode="RGBA").save(
                ASSETS / f"canonical_{name}_4b_preview.png")
            nir_rgba = np.stack([cutout5[3]] * 3 + [cutout5[4]], axis=-1)
            Image.fromarray(nir_rgba, mode="RGBA").save(
                ASSETS / f"canonical_{name}_4b_nir_preview.png")
            print(f"  [{name}] cutout {cutout5.shape[1]}x{cutout5.shape[2]} "
                   f"alpha={int((cutout5[4]>0).sum())} px, "
                   f"NIR-mean(hull)={cutout5[3][cutout5[4]>0].mean():.0f}")


if __name__ == "__main__":
    main()
