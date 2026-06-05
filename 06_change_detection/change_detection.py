# =============================================================================
# 06_change_detection/change_detection.py
#
# PURPOSE : Compare classified rasters between consecutive epochs.
#           For each pair of epochs:
#             1. Pixel-by-pixel class transition map
#             2. Transition matrix (area in pixels and hectares)
#             3. Stable / changed binary raster
#           Also produces a multi-epoch seagrass area trend table.
#
# OUTPUT  : data/change/change_YYYY_YYYY.tif   (1=stable, 2=changed, 0=NoData)
#           06_change_detection/transitions_YYYY_YYYY.csv
#           06_change_detection/seagrass_trend.csv
#
# RUN     : python 06_change_detection/change_detection.py
# =============================================================================

import sys
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (EPOCHS, CLASSIFIED_DIR, CHANGE_DIR,
                    CLASS_NAMES, GSD)
from utils.io import save_raster

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1 pixel = GSD² m² → convert to hectares
PX_TO_HA = (GSD ** 2) / 10_000


# =============================================================================
# LOAD CLASSIFIED RASTER
# =============================================================================

def load_classified(year):
    path = CLASSIFIED_DIR / f"classified_{year}.tif"
    if not path.exists():
        return None, None
    with rasterio.open(path) as src:
        arr     = src.read(1)
        profile = src.profile
    return arr, profile


# =============================================================================
# PAIRWISE CHANGE
# =============================================================================

def compute_pairwise_change(year_a, year_b):
    arr_a, prof_a = load_classified(year_a)
    arr_b, _      = load_classified(year_b)

    if arr_a is None or arr_b is None:
        print(f"  [SKIP] {year_a}→{year_b}: missing classified raster")
        return

    print(f"\n  {year_a} → {year_b}")

    # Valid pixels = water (non-zero) in both epochs
    valid = (arr_a > 0) & (arr_b > 0)

    # Binary change raster
    change = np.zeros_like(arr_a, dtype=np.uint8)
    change[valid & (arr_a == arr_b)] = 1   # stable
    change[valid & (arr_a != arr_b)] = 2   # changed

    stable_pct = (change == 1).sum() / valid.sum() * 100 if valid.sum() > 0 else 0
    print(f"    Stable pixels  : {stable_pct:.1f}%")
    print(f"    Changed pixels : {100 - stable_pct:.1f}%")

    # Save change raster
    out_tif = CHANGE_DIR / f"change_{year_a}_{year_b}.tif"
    save_raster(change, prof_a, out_tif, dtype=rasterio.uint8, nodata=0)

    # Transition matrix
    rows = []
    for from_id, from_name in CLASS_NAMES.items():
        for to_id, to_name in CLASS_NAMES.items():
            mask  = valid & (arr_a == from_id) & (arr_b == to_id)
            n_px  = int(mask.sum())
            n_ha  = round(n_px * PX_TO_HA, 2)
            rows.append({
                "from_class":  from_name,
                "to_class":    to_name,
                "pixels":      n_px,
                "hectares":    n_ha,
            })

    tm = pd.DataFrame(rows)
    tm_pivot = tm.pivot(index="from_class", columns="to_class", values="hectares")
    print(f"\n    Transition matrix (hectares):\n{tm_pivot.round(2)}\n")
    tm.to_csv(OUT_DIR / f"transitions_{year_a}_{year_b}.csv", index=False)


# =============================================================================
# SEAGRASS AREA TREND (all epochs)
# =============================================================================

def seagrass_trend():
    print("\n  Seagrass area trend across all epochs:")
    years  = [year for year, _, _ in EPOCHS]
    rows   = []

    for year in years:
        arr, _ = load_classified(year)
        if arr is None:
            continue
        total_water = (arr > 0).sum()
        sg_px       = (arr == 1).sum()
        sand_px     = (arr == 2).sum()
        rock_px     = (arr == 3).sum()

        rows.append({
            "year":            year,
            "seagrass_ha":     round(sg_px   * PX_TO_HA, 2),
            "sand_ha":         round(sand_px  * PX_TO_HA, 2),
            "rock_ha":         round(rock_px  * PX_TO_HA, 2),
            "seagrass_pct":    round(sg_px / total_water * 100, 2) if total_water > 0 else 0,
            "total_water_ha":  round(total_water * PX_TO_HA, 2),
        })

    if not rows:
        print("  No classified rasters found.")
        return

    df = pd.DataFrame(rows).set_index("year")
    print(df.to_string())
    df.to_csv(OUT_DIR / "seagrass_trend.csv")
    print(f"\n  Trend table saved → {OUT_DIR / 'seagrass_trend.csv'}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=== Step 06: Change detection ===\n")

    years = [year for year, _, _ in EPOCHS]

    # Pairwise consecutive comparisons
    for i in range(len(years) - 1):
        compute_pairwise_change(years[i], years[i+1])

    # Overall seagrass trend
    seagrass_trend()

    print("\n✓ Change detection complete — run step 07 for figures.")
