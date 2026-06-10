"""
06_transition_matrix.py
───────────────────────
Cross-tabulates consecutive epoch pairs to produce gain/loss/stable
transition matrices. Shows what each class changed TO between epochs.
Key for understanding whether lost Posidonia became sand (degradation)
or rock (physical disturbance).

Input:  02_classified_maps/from_gee/pianosa_{year}_classified.tif
        03_masks/pianosa_study_zone.tif
Output: 04_extent_analysis/outputs/transition_matrix_{y1}_{y2}.csv
        04_extent_analysis/outputs/gain_loss_all_epochs.csv
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

CLASS_DIR = "02_classified_maps/from_gee"
MASK_TIF  = "03_masks/pianosa_study_zone.tif"
OUT_DIR   = "04_extent_analysis/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_MAP   = {1: "posidonia", 2: "rock", 3: "sand"}
CLASS_NAMES = list(CLASS_MAP.values())
CLASS_IDS   = list(CLASS_MAP.keys())

EPOCHS = ["2016", "2019", "2021", "2022", "2023", "2024"]

YEAR_NUMERIC = {
    "2016": 2016.5, "2019": 2019.5, "2021": 2021.5,
    "2022": 2022.5, "2023": 2023.5, "2024": 2024.5,
}


def load_classified(year):
    path = os.path.join(CLASS_DIR, f"pianosa_{year}_classified.tif")
    if not os.path.exists(path):
        return None, None
    with rasterio.open(path) as src:
        arr     = src.read(1)
        nodata  = src.nodata
        profile = src.profile
    if nodata is not None:
        arr = np.where(arr == nodata, 0, arr)
    return arr, profile


def load_mask(shape):
    if not os.path.exists(MASK_TIF):
        return np.ones(shape, dtype=bool)
    with rasterio.open(MASK_TIF) as src:
        m = src.read(1).astype(bool)
    if m.shape != shape:
        print(f"  WARNING: Mask shape mismatch — no mask applied")
        return np.ones(shape, dtype=bool)
    return m


def transition_matrix(arr1, arr2, mask, px_ha, year1, year2):
    """
    Build a class × class transition matrix in hectares.
    Rows = class at t1, columns = class at t2.
    """
    rows = []
    for from_id, from_name in CLASS_MAP.items():
        for to_id, to_name in CLASS_MAP.items():
            px = ((arr1 == from_id) & (arr2 == to_id) & mask).sum()
            rows.append({
                "from_class": from_name,
                "to_class":   to_name,
                "pixels":     int(px),
                "hectares":   round(float(px * px_ha), 4),
            })
    df = pd.DataFrame(rows)

    # Pivot to matrix form
    matrix = df.pivot(
        index="from_class",
        columns="to_class",
        values="hectares"
    ).reindex(index=CLASS_NAMES, columns=CLASS_NAMES).fillna(0)

    return df, matrix


def gain_loss_summary(matrix, year1, year2):
    """Derive gain, loss, stable, net change per class."""
    rows = []
    for cls in CLASS_NAMES:
        stable = matrix.loc[cls, cls]
        loss   = matrix.loc[cls].sum() - stable    # left this class
        gain   = matrix[cls].sum() - stable        # entered this class
        net    = gain - loss
        rows.append({
            "year1":    year1,
            "year2":    year2,
            "class":    cls,
            "stable_ha": round(stable, 3),
            "gain_ha":   round(gain, 3),
            "loss_ha":   round(loss, 3),
            "net_ha":    round(net, 3),
            "interval_yr": YEAR_NUMERIC[year2] - YEAR_NUMERIC[year1],
        })
    df = pd.DataFrame(rows)
    df["annual_change_ha"] = (df["net_ha"] /
                               df["interval_yr"]).round(4)
    return df


def main():
    # Find available epochs
    available = [y for y in EPOCHS
                 if os.path.exists(
                     os.path.join(CLASS_DIR,
                                  f"pianosa_{y}_classified.tif"))]

    if len(available) < 2:
        print("Need at least 2 classified maps.")
        print("Add GEE exports to 02_classified_maps/from_gee/")
        return

    print(f"Available epochs: {available}")
    pairs = [(available[i], available[i+1])
             for i in range(len(available)-1)]

    all_gain_loss = []

    for year1, year2 in pairs:
        print(f"\n── Transition {year1} → {year2} ──────────────────────")

        arr1, prof1 = load_classified(year1)
        arr2, prof2 = load_classified(year2)

        if arr1 is None or arr2 is None:
            print("  SKIP — missing classified map")
            continue

        mask = load_mask(arr1.shape)

        px_size = abs(prof1["transform"].a)
        px_ha   = (px_size ** 2) / 10000

        df_long, matrix = transition_matrix(
            arr1, arr2, mask, px_ha, year1, year2)

        # Save transition matrix
        out_path = os.path.join(
            OUT_DIR, f"transition_matrix_{year1}_{year2}.csv")
        matrix.to_csv(out_path)
        print(f"  Transition matrix (ha):")
        print(matrix.round(2).to_string())
        print(f"  ✓ Saved → {out_path}")

        # Gain/loss summary
        gl = gain_loss_summary(matrix, year1, year2)
        all_gain_loss.append(gl)
        print(f"\n  Gain / Loss summary:")
        print(gl[["class", "stable_ha", "gain_ha",
                   "loss_ha", "net_ha",
                   "annual_change_ha"]].to_string(index=False))

    if all_gain_loss:
        gl_all = pd.concat(all_gain_loss, ignore_index=True)
        out_gl  = os.path.join(OUT_DIR, "gain_loss_all_epochs.csv")
        gl_all.to_csv(out_gl, index=False)
        print(f"\n✓ Full gain/loss table → {out_gl}")

        # Highlight Posidonia specifically
        pos = gl_all[gl_all["class"] == "posidonia"]
        if not pos.empty:
            print("\nPosidonia change summary:")
            print(pos[["year1", "year2", "gain_ha",
                        "loss_ha", "net_ha",
                        "annual_change_ha"]].to_string(index=False))


if __name__ == "__main__":
    main()
