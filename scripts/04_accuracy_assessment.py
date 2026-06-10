"""
04_accuracy_assessment.py
─────────────────────────
Validates GEE-exported classified maps against the held-out test GT
polygons. Produces per-epoch confusion matrices, OA, Kappa, and F1
per class. Exports a single accuracy summary CSV for the paper.

Input:  02_classified_maps/from_gee/pianosa_{year}_classified.tif
        01_ground_truth/pianosa_GT_test.gpkg
        03_masks/pianosa_study_zone.tif
Output: 06_accuracy_assessment/confusion_matrices/confusion_{year}.csv
        06_accuracy_assessment/accuracy_summary.csv
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from sklearn.metrics import (confusion_matrix, cohen_kappa_score,
                              f1_score, accuracy_score,
                              classification_report)

# ── Paths ─────────────────────────────────────────────────────────────
GT_TEST    = "01_ground_truth/pianosa_GT_test.gpkg"
MASK_TIF   = "03_masks/pianosa_study_zone.tif"
CLASS_DIR  = "02_classified_maps/from_gee"
OUT_DIR    = "06_accuracy_assessment"
CM_DIR     = os.path.join(OUT_DIR, "confusion_matrices")
SUMMARY    = os.path.join(OUT_DIR, "accuracy_summary.csv")

os.makedirs(CM_DIR, exist_ok=True)

# ── Class mapping ─────────────────────────────────────────────────────
# Must match the class values used in GEE classifier output
CLASS_MAP = {1: "posidonia", 2: "rock", 3: "sand"}
CLASS_IDS = list(CLASS_MAP.keys())
CLASS_NAMES = list(CLASS_MAP.values())

EPOCHS = ["2016", "2019", "2021", "2022", "2023", "2024"]


def sample_classified_at_polygons(classified_path, gt_gdf):
    """
    For each GT polygon, sample the majority class value from the
    classified raster. Returns arrays of (true_label, predicted_label).
    """
    y_true, y_pred = [], []

    with rasterio.open(classified_path) as src:
        for _, row in gt_gdf.iterrows():
            geom = [row.geometry.__geo_interface__]
            try:
                out_image, _ = rio_mask(src, geom, crop=True,
                                        nodata=255)
                arr = out_image[0]
                valid = arr[arr != 255]
                if len(valid) == 0:
                    continue
                # Majority vote within polygon
                values, counts = np.unique(valid, return_counts=True)
                pred_class = values[np.argmax(counts)]
                if pred_class not in CLASS_IDS:
                    continue
                y_true.append(row["class_id"])
                y_pred.append(int(pred_class))
            except Exception:
                continue

    return np.array(y_true), np.array(y_pred)


def class_name_to_id(name):
    inv = {v: k for k, v in CLASS_MAP.items()}
    return inv.get(name, -1)


def compute_metrics(y_true, y_pred, year):
    """Compute full accuracy metrics and return as dict."""
    oa    = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, average=None,
                     labels=CLASS_IDS, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro",
                        labels=CLASS_IDS, zero_division=0)

    metrics = {
        "year":          year,
        "n_samples":     len(y_true),
        "overall_acc":   round(oa, 4),
        "kappa":         round(kappa, 4),
        "f1_macro":      round(f1_macro, 4),
    }
    for i, cls in enumerate(CLASS_NAMES):
        metrics[f"f1_{cls}"] = round(f1[i], 4)

    return metrics


def save_confusion_matrix(y_true, y_pred, year):
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
    df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    df.index.name   = "actual \\ predicted"
    out_path = os.path.join(CM_DIR, f"confusion_{year}.csv")
    df.to_csv(out_path)
    return df


def main():
    if not os.path.exists(GT_TEST):
        print(f"ERROR: Test GT not found at {GT_TEST}")
        print("Run 02_gt_split.py first.")
        return

    print("Loading test GT ...")
    gt = gpd.read_file(GT_TEST)
    gt["class_id"] = gt["class"].apply(class_name_to_id)
    gt = gt[gt["class_id"] != -1]
    print(f"  {len(gt)} test polygons loaded")

    all_metrics = []

    for year in EPOCHS:
        classified_path = os.path.join(
            CLASS_DIR, f"pianosa_{year}_classified.tif")

        if not os.path.exists(classified_path):
            print(f"\n[SKIP] {year} — classified map not found "
                  f"({classified_path})")
            print("        Drop GEE export into 02_classified_maps/from_gee/")
            continue

        print(f"\n── {year} ──────────────────────────────────")
        print(f"  Classified map: {classified_path}")

        y_true, y_pred = sample_classified_at_polygons(
            classified_path, gt)

        if len(y_true) == 0:
            print(f"  WARNING: No valid samples found for {year}")
            continue

        print(f"  Sampled {len(y_true)} polygons")

        metrics = compute_metrics(y_true, y_pred, year)
        all_metrics.append(metrics)

        cm_df = save_confusion_matrix(y_true, y_pred, year)
        print(f"\n  Confusion matrix:")
        print(cm_df.to_string())
        print(f"\n  Overall accuracy : {metrics['overall_acc']:.4f}")
        print(f"  Kappa            : {metrics['kappa']:.4f}")
        print(f"  F1 macro         : {metrics['f1_macro']:.4f}")
        for cls in CLASS_NAMES:
            print(f"  F1 {cls:<12}: {metrics[f'f1_{cls}']:.4f}")

        print(f"\n  ✓ Confusion matrix → {CM_DIR}/confusion_{year}.csv")

    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        summary_df.to_csv(SUMMARY, index=False)
        print(f"\n✓ Accuracy summary → {SUMMARY}")
        print("\nSummary:")
        print(summary_df.to_string(index=False))
    else:
        print("\nNo classified maps found yet.")
        print("Add GEE exports to 02_classified_maps/from_gee/ and re-run.")


if __name__ == "__main__":
    main()
