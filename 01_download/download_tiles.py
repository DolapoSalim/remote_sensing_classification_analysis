# =============================================================================
# 01_download/download_tiles.py
#
# PURPOSE : Download orthophoto tiles from Regione Toscana WMS and
#           assemble them into a georeferenced GeoTIFF per epoch.
#
# OUTPUT FORMAT: 3-band RGB GeoTIFF with NoData=0
#   - Valid imagery pixels: RGB values from the server
#   - Background / outside survey boundary: RGB = (0,0,0), flagged as NoData
#   - QGIS and any GDAL tool will render NoData pixels as transparent,
#     so the image sits cleanly over OSM / Google Satellite Hybrid
#
# WHY NOT RGBA:
#   The server imagery is RGB. Transparency is handled via the GeoTIFF NoData
#   metadata tag — no need for a 4th alpha band. This keeps the file format
#   simple, compatible with all tools, and consistent with the preprocessing
#   scripts which expect 3 bands.
#
# BACKGROUND DETECTION:
#   Outside the survey boundary the server returns pure white (255,255,255).
#   We detect this with a threshold (R>=252, G>=252, B>=252) and write (0,0,0)
#   instead, then declare 0 as NoData. A 2-pixel morphological erosion removes
#   the antialiased fringe at the survey edge.
#
# RUN     : python 01_download/download_tiles.py
# =============================================================================

import sys, math, tempfile, os
import requests
import numpy as np
from osgeo import gdal, osr
from pathlib import Path
from scipy.ndimage import binary_erosion

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (BASE_URL, EPOCHS, RAW_DIR,
                    XMIN, YMIN, XMAX, YMAX,
                    GSD, TILE_M, TILE_PX, OVERLAP_M,
                    TOTAL_W, TOTAL_H)

WHITE_THRESHOLD = 252   # R,G,B all >= this → background pixel
EDGE_ERODE_PX   = 2     # erode valid mask by this many px to remove fringe


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
        f"&TRANSPARENT=TRUE"
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
# BACKGROUND MASK
# Returns True where pixels are valid imagery (not background).
# Uses server alpha band if present, then white-pixel detection,
# then erodes the boundary to remove antialiased fringe.
# =============================================================================

def valid_mask(r_arr, g_arr, b_arr, server_alpha=None):
    if server_alpha is not None:
        valid = server_alpha > 0
    else:
        valid = np.ones(r_arr.shape, dtype=bool)

    # Zero out white / near-white background
    white = (
        (r_arr >= WHITE_THRESHOLD) &
        (g_arr >= WHITE_THRESHOLD) &
        (b_arr >= WHITE_THRESHOLD)
    )
    valid[white] = False

    # Erode boundary to remove antialiased fringe pixels
    if EDGE_ERODE_PX > 0:
        struct = np.ones((EDGE_ERODE_PX * 2 + 1, EDGE_ERODE_PX * 2 + 1))
        valid  = binary_erosion(valid, structure=struct)

    return valid


# =============================================================================
# MOSAIC ASSEMBLY — 3-band RGB with NoData=0
# =============================================================================

def assemble_mosaic(year, wms_map, layer):
    out_path = RAW_DIR / f"pianosa_{year}_RGB.tif"
    if out_path.exists():
        print(f"[SKIP] {out_path.name} already exists")
        return

    print(f"\n── {year} {'─'*45}")
    print(f"   Full image: {TOTAL_W:,} × {TOTAL_H:,} px  |  3-band RGB  |  NoData=0")

    driver  = gdal.GetDriverByName("GTiff")
    options = ["COMPRESS=LZW", "TILED=YES",
               "BLOCKXSIZE=512", "BLOCKYSIZE=512", "BIGTIFF=YES"]

    ds = driver.Create(str(out_path), TOTAL_W, TOTAL_H, 3,
                       gdal.GDT_Byte, options)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32632)
    ds.SetProjection(srs.ExportToWkt())
    ds.SetGeoTransform([XMIN, GSD, 0, YMAX, 0, -GSD])

    # Set NoData = 0 on all three bands
    # QGIS reads this and renders those pixels as transparent
    for b in range(1, 4):
        ds.GetRasterBand(b).SetNoDataValue(0)

    n_x = math.ceil(TOTAL_W / TILE_PX)
    n_y = math.ceil(TOTAL_H / TILE_PX)
    total_tiles = n_x * n_y
    done = failed = 0

    for row in range(n_y):
        for col in range(n_x):
            done += 1

            px_off = col * TILE_PX
            py_off = row * TILE_PX
            px_end = min(px_off + TILE_PX, TOTAL_W)
            py_end = min(py_off + TILE_PX, TOTAL_H)
            w_px   = px_end - px_off
            h_px   = py_end - py_off

            tx_min = XMIN + px_off * GSD
            tx_max = XMIN + px_end * GSD
            ty_max = YMAX - py_off * GSD
            ty_min = YMAX - py_end * GSD

            dl_xmin = max(tx_min - OVERLAP_M, XMIN)
            dl_xmax = min(tx_max + OVERLAP_M, XMAX)
            dl_ymin = max(ty_min - OVERLAP_M, YMIN)
            dl_ymax = min(ty_max + OVERLAP_M, YMAX)

            dl_w   = int(round((dl_xmax - dl_xmin) / GSD))
            dl_h   = int(round((dl_ymax - dl_ymin) / GSD))
            crop_x = int(round((tx_min - dl_xmin) / GSD))
            crop_y = int(round((dl_ymax - ty_max) / GSD))

            print(f"  Tile {done:>4}/{total_tiles} "
                  f"(col={col} row={row}) {w_px}×{h_px}px ...",
                  end=" ", flush=True)

            data = download_tile(wms_map, layer,
                                 dl_xmin, dl_ymin, dl_xmax, dl_ymax,
                                 dl_w, dl_h)

            if data is None:
                print("FAILED — writing NoData zeros")
                failed += 1
                zeros = np.zeros((h_px, w_px), dtype=np.uint8)
                for b in range(1, 4):
                    ds.GetRasterBand(b).WriteArray(zeros, px_off, py_off)
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

            def read_band(b):
                src_b = min(b, n_bands)
                arr   = tile_ds.GetRasterBand(src_b).ReadAsArray()
                return arr[crop_y:crop_y + h_px, crop_x:crop_x + w_px]

            r_arr = read_band(1)
            g_arr = read_band(2)
            b_arr = read_band(3)
            server_alpha = read_band(4) if n_bands == 4 else None

            # Build valid mask and zero out background pixels
            mask = valid_mask(r_arr, g_arr, b_arr, server_alpha)
            r_arr[~mask] = 0
            g_arr[~mask] = 0
            b_arr[~mask] = 0

            ds.GetRasterBand(1).WriteArray(r_arr, px_off, py_off)
            ds.GetRasterBand(2).WriteArray(g_arr, px_off, py_off)
            ds.GetRasterBand(3).WriteArray(b_arr, px_off, py_off)

            tile_ds = None
            os.unlink(tmp_path)
            print("OK")

    ds.FlushCache()
    ds = None

    status = f" Saved → {out_path}"
    if failed:
        status += f"  {failed}/{total_tiles} tiles failed"
    print(f"  {status}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"Output: {RAW_DIR}")
    print("Format: 3-band RGB GeoTIFF  |  NoData=0 (transparent in QGIS)\n")
    for year, wms_map, layer in EPOCHS:
        assemble_mosaic(year, wms_map, layer)
    print("\n All epochs downloaded.")
    print("Run build_overviews.py next, then load the RGB.tif over your basemap.")
