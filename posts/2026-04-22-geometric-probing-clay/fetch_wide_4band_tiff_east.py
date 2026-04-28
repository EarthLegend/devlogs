"""fetch_wide_4band_tiff_east.py — second strip on the east side of the source
NAIP tile, covering boat2's area + any additional vessels further east.

Output:
  assets/anchorage_wide_east_4b.tif   (~50 MB LZW, 2940 x 9000 4-band)

Usage:
  python fetch_wide_4band_tiff_east.py

Then in QGIS:
  1. Open assets/anchorage_wide_east_4b.tif
  2. Trace polygons for boats you can see (NEW boats only — skip ones already
     digitised in moreboats.geojson)
  3. Save as assets/moreboats_east.geojson
  4. Run polygons_wide_to_canonicals.py (it auto-includes moreboats_east.geojson
     if present and dedupes by centroid distance < 50 m)
"""
from __future__ import annotations
from pathlib import Path
import sys

import pystac_client
import planetary_computer
import rasterio
from rasterio.windows import Window, transform as window_transform

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
TILE_ID = "ca_m_3712222_nw_10_060_20200524"

# East side of source tile (col 7000-9940), full vertical extent
CROP_WIN = (7000, 0, 2940, 9000)


def main():
    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    items = list(cat.search(collections=["naip"], ids=[TILE_ID]).items())
    if not items:
        sys.exit(f"no item {TILE_ID}")
    href = items[0].assets["image"].href
    out = ASSETS / "anchorage_wide_east_4b.tif"
    if out.exists():
        out.unlink()
    co, ro, w, h = CROP_WIN
    win = Window(col_off=co, row_off=ro, width=w, height=h)
    print(f"streaming east strip {CROP_WIN} → {out}")
    with rasterio.open(href) as src:
        arr = src.read(window=win)
        t = window_transform(win, src.transform)
        prof = src.profile.copy()
        prof.update(width=w, height=h, transform=t,
                     compress="lzw", tiled=True,
                     blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(arr)
            for i, n in enumerate(["Red", "Green", "Blue", "NIR"], start=1):
                dst.set_band_description(i, n)
    with rasterio.open(out) as r:
        from rasterio.warp import transform_bounds
        wgs = transform_bounds(r.crs, "EPSG:4326",
                                 r.bounds.left, r.bounds.bottom,
                                 r.bounds.right, r.bounds.top)
        print(f"saved: {out.name}  shape {r.shape}  bands {r.count}")
        print(f"  WGS84: lon {wgs[0]:.4f}–{wgs[2]:.4f}  lat {wgs[1]:.4f}–{wgs[3]:.4f}")
        print(f"  size: {out.stat().st_size / 1024 / 1024:.0f} MB")


if __name__ == "__main__":
    main()
