import requests
import numpy as np
from osgeo import gdal, osr
import os
import math
import tempfile

# ── AOI ──────────────────────────────────────────────────────────────
XMIN, YMIN = 585183.86, 4711140.24
XMAX, YMAX = 592352.51, 4719733.64

# ── Settings ─────────────────────────────────────────────────────────
GSD        = 0.20
TILE_M     = 500
OVERLAP_M  = 10
OVERLAP_PX = int(OVERLAP_M / GSD)   # 50 pixels

EPOCHS = [
    ("2016", "owsofc",    "rt_ofc.5k16.32bit"),
    ("2019", "owsofc",    "rt_ofc.5k19.32bit"),
    ("2021", "owsofc_rt", "rt_ofc.5k21.32bit"),
    ("2022", "owsofc",    "rt_ofc.5k22.32bit"),
    ("2023", "owsofc_rt", "rt_ofc.5k23.32bit"),
    ("2024", "owsofc_rt", "rt_ofc.5k24.32bit"),
]

BASE_URL = "https://www502.regione.toscana.it/ows_ofc/com.rt.wms.RTmap/wms"
os.makedirs("pianosa", exist_ok=True)

TILE_PX  = int(TILE_M / GSD)        # 2500 pixels
TOTAL_W  = int(round((XMAX - XMIN) / GSD))
TOTAL_H  = int(round((YMAX - YMIN) / GSD))
print(f"Full image: {TOTAL_W} x {TOTAL_H} pixels")


def download_tile(wms_map, layer, tx_min, ty_min, tx_max, ty_max,
                  width_px, height_px, retries=3):
    url = (
        f"{BASE_URL}?map={wms_map}"
        f"&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
        f"&LAYERS={layer}&STYLES="
        f"&CRS=EPSG:32632"
        f"&BBOX={tx_min},{ty_min},{tx_max},{ty_max}"
        f"&WIDTH={width_px}&HEIGHT={height_px}"
        f"&FORMAT=image/png"            # ← lossless
    )
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and len(r.content) > 1000:
                return r.content
            print(f"    Retry {attempt+1}: status {r.status_code}, "
                  f"size {len(r.content)}")
        except requests.exceptions.RequestException as e:
            print(f"    Retry {attempt+1}: {e}")
    return None


def assemble_mosaic(year, wms_map, layer):
    out_path = f"pianosa/pianosa_{year}_RGB.tif"
    if os.path.exists(out_path):
        print(f"[SKIP] {out_path} already exists")
        return

    print(f"\n── {year} ({'─'*40})")

    driver  = gdal.GetDriverByName("GTiff")
    options = ["COMPRESS=LZW", "TILED=YES",
               "BLOCKXSIZE=512", "BLOCKYSIZE=512", "BIGTIFF=YES"]
    ds = driver.Create(out_path, TOTAL_W, TOTAL_H, 3,
                       gdal.GDT_Byte, options)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32632)
    ds.SetProjection(srs.ExportToWkt())
    ds.SetGeoTransform([XMIN, GSD, 0, YMAX, 0, -GSD])

    n_x = math.ceil(TOTAL_W / TILE_PX)
    n_y = math.ceil(TOTAL_H / TILE_PX)
    total_tiles = n_x * n_y
    done = 0
    failed = 0

    for row in range(n_y):
        for col in range(n_x):
            done += 1

            # ── Core pixel/geo extent ─────────────────────────────
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

            # ── Expanded download extent (with overlap) ───────────
            # Clamp to AOI so we don't request outside it
            dl_xmin = max(tx_min - OVERLAP_M, XMIN)
            dl_xmax = min(tx_max + OVERLAP_M, XMAX)
            dl_ymin = max(ty_min - OVERLAP_M, YMIN)
            dl_ymax = min(ty_max + OVERLAP_M, YMAX)

            dl_w = int(round((dl_xmax - dl_xmin) / GSD))
            dl_h = int(round((dl_ymax - dl_ymin) / GSD))

            # Offsets to crop back to core region
            crop_x = int(round((tx_min - dl_xmin) / GSD))
            crop_y = int(round((dl_ymax - ty_max) / GSD))

            print(f"  Tile {done}/{total_tiles} "
                  f"(col={col} row={row}) "
                  f"{w_px}×{h_px}px ...", end=" ", flush=True)

            data = download_tile(wms_map, layer,
                                 dl_xmin, dl_ymin, dl_xmax, dl_ymax,
                                 dl_w, dl_h)

            if data is None:
                print("FAILED — zeros")
                failed += 1
                for b in range(1, 4):
                    ds.GetRasterBand(b).WriteArray(
                        np.zeros((h_px, w_px), dtype=np.uint8),
                        px_off, py_off)
                continue

            with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            tile_ds = gdal.Open(tmp_path)
            if tile_ds is None:
                print("DECODE ERROR — zeros")
                os.unlink(tmp_path)
                failed += 1
                continue

            # ── Band count guard ──────────────────────────────────
            n_bands = tile_ds.RasterCount
            if n_bands not in (1, 3, 4):
                print(f"UNEXPECTED BANDS ({n_bands}) — skipping")
                tile_ds = None
                os.unlink(tmp_path)
                failed += 1
                continue

            for b in range(1, 4):
                src_b = min(b, n_bands)   # handles greyscale (1 band)
                arr   = tile_ds.GetRasterBand(src_b).ReadAsArray()
                # Crop overlap border back to core size
                arr   = arr[crop_y:crop_y + h_px,
                            crop_x:crop_x + w_px]
                ds.GetRasterBand(b).WriteArray(arr, px_off, py_off)

            tile_ds = None
            os.unlink(tmp_path)
            print("OK")

    ds.FlushCache()
    ds = None

    status = f"✓ Saved → {out_path}"
    if failed:
        status += f"  ⚠ {failed}/{total_tiles} tiles failed (zeros)"
    print(f"  {status}")


for year, wms_map, layer in EPOCHS:
    assemble_mosaic(year, wms_map, layer)

print("\nAll epochs done.")