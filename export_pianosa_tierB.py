import subprocess
import os

# ── Pianosa AOI in EPSG:32632 ──────────────────────────────────────
XMIN, YMIN = 585183.86, 4711140.24
XMAX, YMAX = 592352.51, 4719733.64

# ── All Tier B epochs ──────────────────────────────────────────────
# (year, wms_map, layer_name)
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

for year, wms_map, layer in EPOCHS:
    out_file = f"pianosa/pianosa_{year}_RGB.tif"

    if os.path.exists(out_file):
        print(f"  [SKIP] {out_file} already exists")
        continue

    wms_url = (
        f"WMS:{BASE_URL}?map={wms_map}"
        f"&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
        f"&LAYERS={layer}&STYLES="
        f"&CRS=EPSG:32632&FORMAT=image/tiff"
    )

    cmd = [
        "gdal_translate",
        "-of", "GTiff",
        "-a_srs", "EPSG:32632",
        "-projwin",
            str(XMIN), str(YMAX),   # xmin ymax
            str(XMAX), str(YMIN),   # xmax ymin
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        wms_url,
        out_file
    ]

    print(f"Exporting {year}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  ✓ Saved → {out_file}")
    else:
        print(f"  ✗ ERROR on {year}:")
        print(f"    {result.stderr[:300]}")

print("\nDone. Check the 'pianosa/' folder.")