"""
05_extent_calculation.py
────────────────────────
Core contribution: calculates habitat extent (hectares) per class
per epoch from GEE-classified maps, masked to the study zone.
Exports a tidy CSV ready for plotting and statistical analysis.

Input:  02_classified_maps/from_gee/pianosa_{year}_classified.tif
        03_masks/pianosa_study_zone.tif
Output: 04_extent_analysis/outputs/extent_per_class_per_year.csv
        04_extent_analysis/outputs/extent_summary_stats.csv
"""

import os
import numpy as np
import pandas as pd
import rasterio

# ── Paths ─────────────────────────────────────────────────────────────
CLASS_DIR = "02_classified_maps/from_gee"
MASK_TIF  = "03_masks/pianosa_study_zone.tif"
OUT_DIR   = "04_extent_analysis/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV   = os.path.join(OUT_DIR, "extent_per_class_per_year.csv")
STATS_CSV = os.path.join(OUT_DIR, "extent_summary_stats.csv")

# ── Class mapping ─────────────────────────────────────────────────────
CLASS_MAP = {1: "posidonia", 2: "rock", 3: "sand"}

EPOCHS = ["2016", "2019", "2021", "2022", "2023", "2024"]

# Approximate years for temporal analysis (2024 flight = mid-2024)
YEAR_NUMERIC = {
    "2016": 2016.5,
    "2019": 2019.5,
    "2021": 2021.5,
    "2022": 2022.5,
    "2023": 2023.5,
    "2024": 2024.5,
}


def load_mask():
    """Load study zone mask as boolean array."""
    if not os.path.exists(MASK_TIF):
        print("WARNING: No mask found — analysing full raster extent.")
        print("Run 03_mask_creation.py first for accurate results.")
        return None
    with rasterio.open(MASK_TIF) as src:
        return src.read(1).astype(bool)


def calculate_extent(classified_path, mask=None):
    """
    Count pixels per class, apply mask, convert to hectares.
    Returns dict: {class_name: hectares}
    """
    with rasterio.open(classified_path) as src:
        arr      = src.read(1)
        nodata   = src.nodata
        # Pixel area in hectares (GSD=0.2m → 0.04m² → 0.000004 ha)
        px_size  = abs(src.transform.a)   # metres per pixel
        px_ha    = (px_size ** 2) / 10000

    # Apply nodata
    valid = np.ones(arr.shape, dtype=bool)
    if nodata is not None:
        valid = arr != nodata

    # Apply study zone mask
    if mask is not None:
        # Resize mask if dimensions differ slightly
        if mask.shape != arr.shape:
            print(f"  WARNING: Mask shape {mask.shape} != "
                  f"raster shape {arr.shape} — skipping mask")
        else:
            valid = valid & mask

    results = {}
    total_valid_px = valid.sum()

    for class_id, class_name in CLASS_MAP.items():
        px_count = ((arr == class_id) & valid).sum()
        hectares = px_count * px_ha
        pct      = (px_count / total_valid_px * 100
                    if total_valid_px > 0 else 0)
        results[class_name] = {
            "pixels":   int(px_count),
            "hectares": round(float(hectares), 4),
            "pct":      round(float(pct), 2),
        }

    results["_total_valid_px"] = int(total_valid_px)
    results["_px_size_m"]      = float(px_size)
    results["_px_ha"]          = float(px_ha)

    return results


def main():
    mask = load_mask()
    rows = []

    print("Calculating habitat extent per epoch ...\n")
    print(f"{'Year':<6} {'Class':<12} {'Hectares':>10} {'Pixels':>12} {'%':>7}")
    print("─" * 52)

    found_any = False

    for year in EPOCHS:
        classified_path = os.path.join(
            CLASS_DIR, f"pianosa_{year}_classified.tif")

        if not os.path.exists(classified_path):
            print(f"[SKIP] {year} — not found in {CLASS_DIR}")
            continue

        found_any = True
        extent    = calculate_extent(classified_path, mask)
        total_ha  = sum(
            v["hectares"] for k, v in extent.items()
            if not k.startswith("_"))

        for class_name in CLASS_MAP.values():
            d = extent[class_name]
            print(f"{year:<6} {class_name:<12} "
                  f"{d['hectares']:>10.2f} "
                  f"{d['pixels']:>12,} "
                  f"{d['pct']:>6.1f}%")
            rows.append({
                "year":       year,
                "year_num":   YEAR_NUMERIC[year],
                "class":      class_name,
                "hectares":   d["hectares"],
                "pixels":     d["pixels"],
                "pct_of_zone": d["pct"],
                "total_zone_ha": round(total_ha, 2),
            })
        print()

    if not found_any:
        print("\nNo classified maps found in 02_classified_maps/from_gee/")
        print("Waiting for GEE exports.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"✓ Extent table → {OUT_CSV}")

    # Summary statistics per class across years
    stats = df.groupby("class")["hectares"].agg(
        ["mean", "std", "min", "max"]).round(3)
    stats.columns = ["mean_ha", "std_ha", "min_ha", "max_ha"]

    # Add total change and rate
    for cls in CLASS_MAP.values():
        sub = df[df["class"] == cls].sort_values("year_num")
        if len(sub) >= 2:
            first = sub.iloc[0]["hectares"]
            last  = sub.iloc[-1]["hectares"]
            yrs   = sub.iloc[-1]["year_num"] - sub.iloc[0]["year_num"]
            stats.loc[cls, "total_change_ha"] = round(last - first, 3)
            stats.loc[cls, "annual_rate_ha"]  = round(
                (last - first) / yrs, 3) if yrs > 0 else 0
            stats.loc[cls, "pct_change"]      = round(
                (last - first) / first * 100, 2) if first > 0 else 0

    stats.to_csv(STATS_CSV)
    print(f"✓ Summary stats → {STATS_CSV}")
    print("\nSummary by class:")
    print(stats.to_string())


if __name__ == "__main__":
    main()
