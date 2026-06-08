# =============================================================================
# config.py  —  EDIT THIS BEFORE RUNNING ANY SCRIPT
# All other scripts import from here.
# =============================================================================

from pathlib import Path

# ── Project root (all paths relative to this) ─────────────────────────────
ROOT = Path(__file__).parent

# ── Data directories ──────────────────────────────────────────────────────
RAW_DIR        = ROOT / "data" / "raw"          # downloaded GeoTIFFs
PROCESSED_DIR  = ROOT / "data" / "processed"    # feature stacks (.npy)
CLASSIFIED_DIR = ROOT / "data" / "classified"   # classified rasters
CHANGE_DIR     = ROOT / "data" / "change"       # change rasters
TRAINING_DIR   = ROOT / "training_data"
OUTPUT_DIR     = ROOT / "07_outputs" / "figures"

for d in [RAW_DIR, PROCESSED_DIR, CLASSIFIED_DIR,
          CHANGE_DIR, TRAINING_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Training / validation data ────────────────────────────────────────────
TRAINING_POLYGONS   = TRAINING_DIR / "training_polygons.geojson"
VALIDATION_POINTS   = TRAINING_DIR / "validation_points.geojson"

# ── WMS download settings ─────────────────────────────────────────────────
XMIN, YMIN = 585183.86, 4711140.24
XMAX, YMAX = 592352.51, 4719733.64

GSD       = 0.20    # ground sample distance in metres
TILE_M    = 500     # tile size in metres
OVERLAP_M = 10      # overlap border in metres

BASE_URL = "https://www502.regione.toscana.it/ows_ofc/com.rt.wms.RTmap/wms"

EPOCHS = [
    ("2016", "owsofc",    "rt_ofc.5k16.32bit"),
    ("2019", "owsofc",    "rt_ofc.5k19.32bit"),
    ("2021", "owsofc_rt", "rt_ofc.5k21.32bit"),
    ("2022", "owsofc",    "rt_ofc.5k22.32bit"),
    ("2023", "owsofc_rt", "rt_ofc.5k23.32bit"),
    ("2024", "owsofc_rt", "rt_ofc.5k24.32bit"),
]

# ── Classification settings ───────────────────────────────────────────────
TRAINING_YEAR = "2022"    # year to digitize training polygons on

# Output filename suffix for downloaded rasters
# RGBA = 4-band with alpha channel (transparent land)
RASTER_SUFFIX = "RGBA"

CLASS_MAP = {"seagrass": 1, "sand": 2, "rock": 3}
CLASS_NAMES  = {1: "Seagrass", 2: "Sand", 3: "Rock"}
CLASS_COLORS = {1: "#1a7a4a", 2: "#e8d5a3", 3: "#8c7b6b"}

RF_PARAMS = {
    "n_estimators":     300,
    "max_depth":        None,
    "min_samples_leaf": 5,
    "n_jobs":           -1,
    "random_state":     42,
    "class_weight":     "balanced",
}

FEATURE_NAMES = [
    "R", "G", "B",
    "DI_RG", "DI_GB",
    "VARI",
    "GLCM_contrast", "GLCM_homogeneity",
    "GLCM_energy",   "GLCM_correlation",
]

# ── Preprocessing settings ────────────────────────────────────────────────
NDWI_THRESHOLD = 0.0     # pixels above this are treated as water
GLCM_WINDOW    = 21      # texture window size in pixels
GLCM_LEVELS    = 64      # gray levels for GLCM

# ── Derived geometry (do not edit) ───────────────────────────────────────
import math
TILE_PX  = int(TILE_M  / GSD)
TOTAL_W  = int(round((XMAX - XMIN) / GSD))
TOTAL_H  = int(round((YMAX - YMIN) / GSD))
