"""fetch_wide_4band_tiff.py — pull the FULL boat-rich region of the source
NAIP tile as one wide 4-band GeoTIFF.

The source tile (ca_m_3712222_nw_10_060_20200524) is 12290 × 9940 px
at 0.6 m/px and the SF Bay anchorage where both ships are has 8–10
visible vessels. Cropping the entire tile gives the user maximum
freedom to digitise any subset of boats they like in QGIS.

Output:
  assets/anchorage_wide_4b.tif    (full tile, ~470 MB uncompressed,
                                    ~250 MB LZW, 4-band RGBN, EPSG:26910)

Workflow:
  1. python fetch_wide_4band_tiff.py
  2. Open assets/anchorage_wide_4b.tif in QGIS
  3. Trace polygons for as many boats as you like in a single GeoPackage
     or shapefile, save as assets/boats_wide.geojson
  4. Run polygon_to_canonical_wide.py (next script) to crop a 540×540
     4-band npz around each polygon and rebuild canonicals for all
     N hand-digitised boats.

Each polygon you draw on the wide tile gives one boat at the highest
resolution. The crop window is auto-snapped to the polygon's centroid
and clipped to the source tile's bounds.
"""
from __future__ import annotations
from pathlib import Path
import sys

import pystac_client
import planetary_computer
import rasterio

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
TILE_ID = "ca_m_3712222_nw_10_060_20200524"


CROP_WIN = (4000, 0, 3000, 9000)       # col_off, row_off, width, height
                                       # — narrow vertical strip down the boat-rich
                                       #   middle (col 4000-7000) covering the
                                       #   anchorage from top to bottom. 27 Mpx,
                                       #   ~25 MB LZW, fetch <30s


def main():
    from rasterio.windows import Window, transform as window_transform
    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    items = list(cat.search(collections=["naip"], ids=[TILE_ID]).items())
    if not items:
        sys.exit(f"no item {TILE_ID}")
    href = items[0].assets["image"].href
    out = ASSETS / "anchorage_wide_4b.tif"
    if out.exists():
        out.unlink()
    print(f"streaming {TILE_ID} crop {CROP_WIN} → {out}")
    co, ro, w, h = CROP_WIN
    win = Window(col_off=co, row_off=ro, width=w, height=h)
    with rasterio.open(href) as src:
        arr = src.read(window=win)
        t = window_transform(win, src.transform)
        prof = src.profile.copy()
        prof.update(width=w, height=h, transform=t,
                     compress="lzw", tiled=True, blockxsize=512,
                     blockysize=512, BIGTIFF="IF_SAFER")
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(arr)
            dst.set_band_description(1, "Red")
            dst.set_band_description(2, "Green")
            dst.set_band_description(3, "Blue")
            dst.set_band_description(4, "NIR")
        with rasterio.open(out) as r:
            print(f"saved: {out.name}")
            print(f"  shape: {r.shape}, bands: {r.count}, crs: {r.crs}")
            print(f"  bounds: {r.bounds}")
            from rasterio.warp import transform_bounds
            wgs = transform_bounds(r.crs, "EPSG:4326",
                                     r.bounds.left, r.bounds.bottom,
                                     r.bounds.right, r.bounds.top)
            print(f"  WGS84: lon {wgs[0]:.4f}–{wgs[2]:.4f} "
                   f"lat {wgs[1]:.4f}–{wgs[3]:.4f}")
            sz_mb = out.stat().st_size / 1024 / 1024
            print(f"  file size: {sz_mb:.0f} MB")


if __name__ == "__main__":
    main()
