# =============================================================================
# 03_training/extract_samples.py
#
# PURPOSE : Sample pixel feature vectors from within the training polygons
#           you digitized in QGIS, and save them as X_train.npy / y_train.npy.
#
# REQUIRES: training_data/training_polygons.geojson
#             — a GeoJSON with a "class" attribute: "seagrass" | "sand" | "rock"
#           data/processed/features_YYYY.npy  (from step 02)
#           data/processed/watermask_YYYY.npy (from step 02)
#
# OUTPUT  : training_data/X_train.npy   (n_samples × n_features)
#           training_data/y_train.npy   (n_samples,)
#
# RUN     : python 03_training/extract_samples.py
# =============================================================================

import sys
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (TRAINING_YEAR, PROCESSED_DIR, TRAINING_DIR, RAW_DIR,
                    TRAINING_POLYGONS, CLASS_MAP, CLASS_NAMES, FEATURE_NAMES)
from utils.io import load_feature_stack


# =============================================================================
# SAMPLE EXTRACTION
# =============================================================================

def extract_samples(feature_stack, water_mask, profile, polygons_path):
    """
    Rasterize each training polygon and collect pixel feature vectors.
    Returns X (n_samples, n_features) and y (n_samples,) integer class IDs.
    """
    gdf = gpd.read_file(polygons_path)
    print(f"  Loaded {len(gdf)} polygons from {polygons_path.name}")

    # Validate that the 'class' column exists and has expected values
    if "class" not in gdf.columns:
        raise ValueError("GeoJSON must have a 'class' column. "
                         "Check your QGIS digitizing attribute.")
    unknown = set(gdf["class"].unique()) - set(CLASS_MAP.keys())
    if unknown:
        raise ValueError(f"Unknown class labels in GeoJSON: {unknown}. "
                         f"Expected: {list(CLASS_MAP.keys())}")

    gdf["class_id"] = gdf["class"].map(CLASS_MAP)

    h, w    = water_mask.shape
    transform = profile["transform"]

    X_list, y_list = [], []

    for class_name, class_id in CLASS_MAP.items():
        subset = gdf[gdf["class"] == class_name]
        if subset.empty:
            print(f" No polygons for class '{class_name}'")
            continue

        # Rasterize all polygons for this class at once
        mask = geometry_mask(
            subset.geometry,
            out_shape=(h, w),
            transform=transform,
            invert=True       # True inside polygons
        )
        mask = mask & water_mask  # restrict to water pixels only

        pixels = feature_stack[mask]   # (n_pixels, n_features)
        labels = np.full(len(pixels), class_id, dtype=np.uint8)

        X_list.append(pixels)
        y_list.append(labels)
        print(f"  {CLASS_NAMES[class_id]:>10}: {len(pixels):>8,} pixels  "
              f"({subset.shape[0]} polygons)")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # Clean up any remaining NaN / inf from log/texture operations
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=== Step 03: Extract training samples ===\n")

    # Check inputs
    if not TRAINING_POLYGONS.exists():
        print(f"ERROR: {TRAINING_POLYGONS} not found.")
        print("  Complete the QGIS digitizing step first.")
        print("  See 03_training/QGIS_GUIDE.md for instructions.")
        sys.exit(1)

    feat_path = PROCESSED_DIR / f"features_{TRAINING_YEAR}.npy"
    mask_path = PROCESSED_DIR / f"watermask_{TRAINING_YEAR}.npy"

    if not feat_path.exists():
        print(f"ERROR: {feat_path} not found — run step 02 first.")
        sys.exit(1)

    # Load feature stack for training year
    print(f"Loading features for training year: {TRAINING_YEAR}")
    stack      = load_feature_stack(feat_path)
    water_mask = np.load(mask_path)

    # Load rasterio profile for the transform
    with rasterio.open(RAW_DIR / f"pianosa_{TRAINING_YEAR}_RGB.tif") as src:
        profile = src.profile

    # Extract samples
    print(f"\nExtracting pixel samples from training polygons...")
    X, y = extract_samples(stack, water_mask, profile, TRAINING_POLYGONS)

    # Summary
    print(f"\n  Total samples : {len(y):,}")
    print(f"  Features      : {X.shape[1]}  → {FEATURE_NAMES}")
    counts = Counter(y)
    for cid, n in sorted(counts.items()):
        print(f"    Class {cid} ({CLASS_NAMES[cid]:>10}): {n:,} ({n/len(y)*100:.1f}%)")

    # Save
    out_X = TRAINING_DIR / "X_train.npy"
    out_y = TRAINING_DIR / "y_train.npy"
    np.save(out_X, X.astype(np.float32))
    np.save(out_y, y.astype(np.uint8))
    print(f"\n  X_train saved → {out_X}")
    print(f"  y_train saved → {out_y}")
    print("\n✓ Training samples ready — run step 04 next.")
