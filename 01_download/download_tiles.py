# =============================================================================
# 01_download/download_tiles.py
#
# PURPOSE : Download orthophoto tiles from Regione Toscana WMS and
#           assemble them into a georeferenced GeoTIFF per epoch.
#
# OUTPUT  : data/raw/pianosa_YYYY_RGB.tif
#
# RUN     : python 01_download/download_tiles.py
# =============================================================================

import sys
import math
import tempfile
import os
import requests
import numpy as np
from osgeo import gdal, osr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (BASE_URL, EPOCHS, RAW_DIR,
                    XMIN, YMIN, XMAX, YMAX,
                    GSD, TILE_M, TILE_PX, OVERLAP_M,
                    TOTAL_W, TOTAL_H)


# =============================================================================
# TILE DOWNLOAD
# =============================================================================

def download_tile(wms_map, layer, tx_min, ty_min, tx_max, ty_max,
                  width_px, height_px, retries=3):
    url = (
        f"{BASE_URL}?map={wms_map}"
        f"&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
        f"&LAYERS={layer}&STYLES="
        f"&CRS=EPSG:32632"
        f"&BBOX={tx_min},{ty_min},{tx_max},{ty_max}"
        f"&WIDTH={width_px}&HEIGHT={height_px}"
        f"&FORMAT=image/png"
    )
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and len(r.content) > 1000:
                return r.content
            print(f"    Retry {attempt+1}: status={r.status_code} "
                  f"size={len(r.content)}")
        except requests.exceptions.RequestException as e:
            print(f"    Retry {attempt+1}: {e}")
    return None


# =============================================================================
# MOSAIC ASSEMBLY
# =============================================================================

def assemble_mosaic(year, wms_map, layer):
    out_path = RAW_DIR / f"pianosa_{year}_RGB.tif"
    if out_path.exists():
        print(f"[SKIP] {out_path.name} already exists")
        return

    print(f"\n── {year} {'─'*45}")
    print(f"   Full image: {TOTAL_W} × {TOTAL_H} px")

    driver  = gdal.GetDriverByName("GTiff")
    options = ["COMPRESS=LZW", "TILED=YES",
               "BLOCKXSIZE=512", "BLOCKYSIZE=512", "BIGTIFF=YES"]
    ds = driver.Create(str(out_path), TOTAL_W, TOTAL_H, 3,
                       gdal.GDT_Byte, options)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32632)
    ds.SetProjection(srs.ExportToWkt())
    ds.SetGeoTransform([XMIN, GSD, 0, YMAX, 0, -GSD])

    n_x = math.ceil(TOTAL_W / TILE_PX)
    n_y = math.ceil(TOTAL_H / TILE_PX)
    total_tiles = n_x * n_y
    done = failed = 0

    for row in range(n_y):
        for col in range(n_x):
            done += 1

            # Core pixel extent
            px_off = col * TILE_PX
            py_off = row * TILE_PX
            px_end = min(px_off + TILE_PX, TOTAL_W)
            py_end = min(py_off + TILE_PX, TOTAL_H)
            w_px   = px_end - px_off
            h_px   = py_end - py_off

            # Core geo extent
            tx_min = XMIN + px_off * GSD
            tx_max = XMIN + px_end * GSD
            ty_max = YMAX - py_off * GSD
            ty_min = YMAX - py_end * GSD

            # Expanded download extent with overlap border (clamped to AOI)
            dl_xmin = max(tx_min - OVERLAP_M, XMIN)
            dl_xmax = min(tx_max + OVERLAP_M, XMAX)
            dl_ymin = max(ty_min - OVERLAP_M, YMIN)
            dl_ymax = min(ty_max + OVERLAP_M, YMAX)

            dl_w = int(round((dl_xmax - dl_xmin) / GSD))
            dl_h = int(round((dl_ymax - dl_ymin) / GSD))

            crop_x = int(round((tx_min - dl_xmin) / GSD))
            crop_y = int(round((dl_ymax - ty_max) / GSD))

            print(f"  Tile {done:>4}/{total_tiles} "
                  f"(col={col} row={row}) {w_px}×{h_px}px ...",
                  end=" ", flush=True)

            data = download_tile(wms_map, layer,
                                 dl_xmin, dl_ymin, dl_xmax, dl_ymax,
                                 dl_w, dl_h)

            if data is None:
                print("FAILED — writing zeros")
                failed += 1
                for b in range(1, 4):
                    ds.GetRasterBand(b).WriteArray(
                        np.zeros((h_px, w_px), dtype=np.uint8), px_off, py_off)
                continue

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            tile_ds = gdal.Open(tmp_path)
            if tile_ds is None:
                print("DECODE ERROR — skipping")
                os.unlink(tmp_path)
                failed += 1
                continue

            n_bands = tile_ds.RasterCount
            if n_bands not in (1, 3, 4):
                print(f"UNEXPECTED BANDS ({n_bands}) — skipping")
                tile_ds = None
                os.unlink(tmp_path)
                failed += 1
                continue

            for b in range(1, 4):
                src_b = min(b, n_bands)
                arr = tile_ds.GetRasterBand(src_b).ReadAsArray()
                arr = arr[crop_y:crop_y + h_px, crop_x:crop_x + w_px]
                ds.GetRasterBand(b).WriteArray(arr, px_off, py_off)

            tile_ds = None
            os.unlink(tmp_path)
            print("OK")

    ds.FlushCache()
    ds = None

    status = f"✓ Saved → {out_path}"
    if failed:
        status += f"  ⚠ {failed}/{total_tiles} tiles failed"
    print(f"  {status}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"Output directory: {RAW_DIR}")
    for year, wms_map, layer in EPOCHS:
        assemble_mosaic(year, wms_map, layer)
    print("\n✓ All epochs downloaded.")
