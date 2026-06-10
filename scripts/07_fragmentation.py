"""
07_fragmentation.py
───────────────────
Computes landscape fragmentation metrics for each class (primarily
Posidonia) per epoch using pylandstats. Key metrics for the paper:
- NP   : number of patches
- AREA_MN : mean patch area (ha)
- LPI  : largest patch index (%)
- ED   : edge density (m/ha)
- MESH : effective mesh size (ha) — best single fragmentation metric

Input:  02_classified_maps/from_gee/pianosa_{year}_classified.tif
        03_masks/pianosa_study_zone.tif
Output: 05_fragmentation_analysis/outputs/fragmentation_metrics_per_year.csv
        05_fragmentation_analysis/outputs/fragmentation_trends.csv
"""

import os
import numpy as np
import pandas as pd
import rasterio
import pylandstats as pls

CLASS_DIR = "02_classified_maps/from_gee"
MASK_TIF  = "03_masks/pianosa_study_zone.tif"
OUT_DIR   = "05_fragmentation_analysis/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV    = os.path.join(OUT_DIR, "fragmentation_metrics_per_year.csv")
TRENDS_CSV = os.path.join(OUT_DIR, "fragmentation_trends.csv")

CLASS_MAP = {1: "posidonia", 2: "rock", 3: "sand"}

EPOCHS = ["2016", "2019", "2021", "2022", "2023", "2024"]
YEAR_NUMERIC = {
    "2016": 2016.5, "2019": 2019.5, "2021": 2021.5,
    "2022": 2022.5, "2023": 2023.5, "2024": 2024.5,
}

# pylandstats metrics to compute
METRICS = [
    "number_of_patches",
    "largest_patch_index",
    "total_area",
    "patch_density",
    "edge_density",
    "effective_mesh_size",
    "area_mn",
    "area_sd",
]


def apply_mask(arr, mask_path):
    """Set pixels outside study zone to nodata (0)."""
    if not os.path.exists(mask_path):
        return arr
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(bool)
    if mask.shape != arr.shape:
        return arr
    return np.where(mask, arr, 0)


def compute_fragmentation(year):
    path = os.path.join(CLASS_DIR, f"pianosa_{year}_classified.tif")
    if not os.path.exists(path):
        return None

    with rasterio.open(path) as src:
        arr     = src.read(1)
        nodata  = src.nodata
        res     = abs(src.transform.a)   # metres per pixel

    # Set nodata to 0 (background)
    if nodata is not None:
        arr = np.where(arr == nodata, 0, arr)

    # Apply study zone mask
    arr = apply_mask(arr, MASK_TIF)

    # pylandstats expects: nodata=0, integer class values
    # Pixel resolution in metres
    try:
        ls = pls.Landscape(arr, res=(res, res), nodata=0)
        metrics_df = ls.compute_class_metrics_df(metrics=METRICS)
        metrics_df = metrics_df.reset_index()
        metrics_df.columns = metrics_df.columns.str.lower()

        # Map class IDs to names
        metrics_df["class_name"] = metrics_df["class_val"].map(CLASS_MAP)
        metrics_df["year"]       = year
        metrics_df["year_num"]   = YEAR_NUMERIC[year]

        return metrics_df

    except Exception as e:
        print(f"  ERROR computing metrics for {year}: {e}")
        return None


def main():
    all_metrics = []

    print("Computing fragmentation metrics ...\n")

    for year in EPOCHS:
        path = os.path.join(CLASS_DIR, f"pianosa_{year}_classified.tif")
        if not os.path.exists(path):
            print(f"[SKIP] {year} — classified map not found")
            continue

        print(f"── {year} ──────────────────────────────────")
        df = compute_fragmentation(year)

        if df is None:
            continue

        # Print posidonia row
        pos = df[df["class_name"] == "posidonia"]
        if not pos.empty:
            row = pos.iloc[0]
            print(f"  Posidonia:")
            for m in METRICS:
                col = m.lower()
                if col in row.index:
                    print(f"    {m:<28}: {row[col]:.4f}")

        all_metrics.append(df)
        print()

    if not all_metrics:
        print("No classified maps found.")
        return

    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"✓ Fragmentation metrics → {OUT_CSV}")

    # ── Trend table: one row per class per year ────────────────────
    trend_rows = []
    for year in combined["year"].unique():
        for cls_id, cls_name in CLASS_MAP.items():
            row = combined[
                (combined["year"] == year) &
                (combined["class_val"] == cls_id)
            ]
            if row.empty:
                continue
            r = row.iloc[0]
            trend_rows.append({
                "year":          year,
                "year_num":      YEAR_NUMERIC[year],
                "class":         cls_name,
                "n_patches":     r.get("number_of_patches", np.nan),
                "total_area_ha": round(
                    r.get("total_area", 0) / 10000, 4),
                "mean_area_ha":  round(
                    r.get("area_mn", 0) / 10000, 6),
                "lpi_pct":       r.get("largest_patch_index", np.nan),
                "patch_density": r.get("patch_density", np.nan),
                "edge_density":  r.get("edge_density", np.nan),
                "mesh_ha":       round(
                    r.get("effective_mesh_size", 0) / 10000, 4),
            })

    trends = pd.DataFrame(trend_rows)
    trends.to_csv(TRENDS_CSV, index=False)
    print(f"✓ Trend table       → {TRENDS_CSV}")

    # Print Posidonia fragmentation trend
    pos_trend = trends[trends["class"] == "posidonia"].sort_values(
        "year_num")
    if not pos_trend.empty:
        print("\nPosidonia fragmentation trend:")
        print(pos_trend[[
            "year", "n_patches", "total_area_ha",
            "mean_area_ha", "mesh_ha", "edge_density"
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
