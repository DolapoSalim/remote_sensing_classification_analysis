import requests
import numpy as np
from osgeo import gdal, osr
import os
import math

# ── AOI ──────────────────────────────────────────────────────────────
XMIN, YMIN = 585183.86, 4711140.24
XMAX, YMAX = 592352.51, 4719733.64

# ── Settings ─────────────────────────────────────────────────────────
GSD        = 0.20          # metres per pixel
TILE_M     = 500           # tile size in metres (500m × 500m chunks)
OVERLAP_M  = 10            # overlap between tiles to avoid edge seams

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
os.makedirs("pianosa/tiles", exist_ok=True)

# ── Tile size in pixels ───────────────────────────────────────────────
TILE_PX    = int(TILE_M / GSD)       # 2500 pixels per tile side
OVERLAP_PX = int(OVERLAP_M / GSD)    # 50 pixels overlap

# ── Full image dimensions ─────────────────────────────────────────────
TOTAL_W = int(round((XMAX - XMIN) / GSD))
TOTAL_H = int(round((YMAX - YMIN) / GSD))
print(f"Full image: {TOTAL_W} x {TOTAL_H} pixels")


def download_tile(wms_map, layer, tx_min, ty_min, tx_max, ty_max,
                  width_px, height_px, retries=3):
    """Download one WMS tile as raw bytes, with retries."""
    url = (
        f"{BASE_URL}?map={wms_map}"
        f"&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
        f"&LAYERS={layer}&STYLES="
        f"&CRS=EPSG:32632"
        f"&BBOX={tx_min},{ty_min},{tx_max},{ty_max}"
        f"&WIDTH={width_px}&HEIGHT={height_px}"
        f"&FORMAT=image/jpeg"
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

    # Create output GeoTIFF
    driver = gdal.GetDriverByName("GTiff")
    options = ["COMPRESS=LZW", "TILED=YES",
               "BLOCKXSIZE=512", "BLOCKYSIZE=512", "BIGTIFF=YES"]
    ds = driver.Create(out_path, TOTAL_W, TOTAL_H, 3,
                       gdal.GDT_Byte, options)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32632)
    ds.SetProjection(srs.ExportToWkt())
    ds.SetGeoTransform([XMIN, GSD, 0, YMAX, 0, -GSD])

    # Tile grid
    n_x = math.ceil(TOTAL_W / TILE_PX)
    n_y = math.ceil(TOTAL_H / TILE_PX)
    total_tiles = n_x * n_y
    done = 0

    for row in range(n_y):
        for col in range(n_x):
            done += 1

            # Pixel offsets in output image
            px_off = col * TILE_PX
            py_off = row * TILE_PX
            px_end = min(px_off + TILE_PX, TOTAL_W)
            py_end = min(py_off + TILE_PX, TOTAL_H)
            w_px = px_end - px_off
            h_px = py_end - py_off

            # Geographic coordinates of this tile
            tx_min = XMIN + px_off * GSD
            tx_max = XMIN + px_end * GSD
            ty_max = YMAX - py_off * GSD
            ty_min = YMAX - py_end * GSD

            print(f"  Tile {done}/{total_tiles} "
                  f"(col={col} row={row}) "
                  f"{w_px}×{h_px}px ...", end=" ")

            data = download_tile(wms_map, layer,
                                 tx_min, ty_min, tx_max, ty_max,
                                 w_px, h_px)

            if data is None:
                print("FAILED — filling with zeros")
                for b in range(1, 4):
                    band = ds.GetRasterBand(b)
                    band.WriteArray(
                        np.zeros((h_px, w_px), dtype=np.uint8),
                        px_off, py_off)
                continue

            # Decode JPEG bytes → numpy array
            import tempfile
            with tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            tile_ds = gdal.Open(tmp_path)
            if tile_ds is None:
                print("DECODE ERROR — filling with zeros")
                os.unlink(tmp_path)
                continue

            for b in range(1, 4):
                arr = tile_ds.GetRasterBand(b).ReadAsArray()
                ds.GetRasterBand(b).WriteArray(arr, px_off, py_off)

            tile_ds = None
            os.unlink(tmp_path)
            print("OK")

    ds.FlushCache()
    ds = None
    print(f"  Saved → {out_path}")


# ── Run ───────────────────────────────────────────────────────────────
for year, wms_map, layer in EPOCHS:
    assemble_mosaic(year, wms_map, layer)

print("\nAll epochs done.")