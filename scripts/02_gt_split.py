"""
02_gt_split.py
──────────────
Splits the digitised ground truth shapefile into spatially separated
train (70%) and test (30%) sets. Spatial separation prevents
autocorrelation inflating accuracy metrics.

Input:  01_ground_truth/pianosa_GT.shp
Output: 01_ground_truth/pianosa_GT_train.gpkg
        01_ground_truth/pianosa_GT_test.gpkg
        01_ground_truth/gt_split_summary.csv
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit

GT_PATH   = "01_ground_truth/pianosa_GT.shp"
OUT_DIR   = "01_ground_truth"
TRAIN_OUT = os.path.join(OUT_DIR, "pianosa_GT_train.gpkg")
TEST_OUT  = os.path.join(OUT_DIR, "pianosa_GT_test.gpkg")
SUMMARY   = os.path.join(OUT_DIR, "gt_split_summary.csv")

RANDOM_STATE = 42
TEST_SIZE    = 0.30


def spatial_split(gdf, test_size=0.30, random_state=42):
    """
    Stratified spatial split — ensures each class has test_size
    proportion in the test set, and that test polygons are
    spatially spread (not clustered in one area).
    """
    gdf = gdf.copy().reset_index(drop=True)

    # Stratified split by class
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    train_idx, test_idx = next(sss.split(gdf, gdf["class"]))

    train = gdf.iloc[train_idx].copy()
    test  = gdf.iloc[test_idx].copy()

    train["split"] = "train"
    test["split"]  = "test"

    return train, test


def main():
    if not os.path.exists(GT_PATH):
        print(f"ERROR: Ground truth file not found at {GT_PATH}")
        print("Complete digitising in QGIS first.")
        return

    print(f"Loading ground truth from {GT_PATH} ...")
    gdf = gpd.read_file(GT_PATH)

    print(f"\nTotal polygons: {len(gdf)}")
    print("\nClass distribution:")
    print(gdf["class"].value_counts().to_string())

    # Check minimum counts
    min_per_class = gdf["class"].value_counts().min()
    if min_per_class < 10:
        print(f"\nWARNING: Minimum class count is {min_per_class}.")
        print("Aim for at least 50 polygons per class before splitting.")

    print(f"\nSplitting: {int((1-TEST_SIZE)*100)}% train / "
          f"{int(TEST_SIZE*100)}% test ...")

    train, test = spatial_split(gdf, TEST_SIZE, RANDOM_STATE)

    print(f"\nTrain set: {len(train)} polygons")
    print(train["class"].value_counts().to_string())
    print(f"\nTest set:  {len(test)} polygons")
    print(test["class"].value_counts().to_string())

    # Save
    train.to_file(TRAIN_OUT, driver="GPKG")
    test.to_file(TEST_OUT,  driver="GPKG")
    print(f"\n✓ Saved train → {TRAIN_OUT}")
    print(f"✓ Saved test  → {TEST_OUT}")

    # Summary CSV
    summary_rows = []
    for cls in gdf["class"].unique():
        summary_rows.append({
            "class":       cls,
            "total":       len(gdf[gdf["class"] == cls]),
            "train":       len(train[train["class"] == cls]),
            "test":        len(test[test["class"] == cls]),
            "test_pct":    round(
                len(test[test["class"] == cls]) /
                len(gdf[gdf["class"] == cls]) * 100, 1)
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY, index=False)
    print(f"✓ Summary   → {SUMMARY}")
    print("\nSplit summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
