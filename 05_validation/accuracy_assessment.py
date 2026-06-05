# =============================================================================
# 05_validation/accuracy_assessment.py
#
# PURPOSE : Evaluate classified maps against an INDEPENDENT validation set
#           (separate from training polygons — digitize these in QGIS too).
#           Outputs confusion matrices, OA, Kappa, and per-class F1 scores
#           ready for publication.
#
# REQUIRES: training_data/validation_points.geojson
#             — point GeoJSON with a "class" attribute
#           data/classified/classified_YYYY.tif  (from step 04)
#
# OUTPUT  : 05_validation/accuracy_YYYY.csv
#           05_validation/confusion_YYYY.csv
#           05_validation/accuracy_summary.csv  (all years combined)
#
# RUN     : python 05_validation/accuracy_assessment.py
# =============================================================================

import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
from sklearn.metrics import (classification_report, confusion_matrix,
                             cohen_kappa_score, accuracy_score)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (EPOCHS, CLASSIFIED_DIR, VALIDATION_POINTS,
                    CLASS_MAP, CLASS_NAMES)

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ASSESS ONE EPOCH
# =============================================================================

def assess_epoch(year):
    classified_path = CLASSIFIED_DIR / f"classified_{year}.tif"
    if not classified_path.exists():
        print(f"  [SKIP] {year}: classified raster not found")
        return None

    gdf = gpd.read_file(VALIDATION_POINTS)
    gdf["class_id"] = gdf["class"].map(CLASS_MAP)

    y_true, y_pred = [], []

    with rasterio.open(classified_path) as src:
        profile   = src.profile
        classified = src.read(1)
        transform  = src.transform
        h, w       = classified.shape

    for _, row in gdf.iterrows():
        # Convert point coordinates to pixel indices
        col_f, row_f = ~transform * (row.geometry.x, row.geometry.y)
        c, r_idx = int(col_f), int(row_f)
        if 0 <= r_idx < h and 0 <= c < w:
            pred = classified[r_idx, c]
            if pred > 0:   # skip NoData (land) pixels
                y_true.append(int(row["class_id"]))
                y_pred.append(int(pred))

    if len(y_true) < 10:
        print(f"  ⚠ {year}: only {len(y_true)} valid validation points — skipping")
        return None

    oa    = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm    = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
    report = classification_report(
        y_true, y_pred,
        labels=[1, 2, 3],
        target_names=list(CLASS_NAMES.values()),
        output_dict=True
    )

    print(f"\n  {year}  (n={len(y_true)} points)")
    print(f"    Overall Accuracy : {oa:.4f}")
    print(f"    Cohen's Kappa    : {kappa:.4f}")

    # Confusion matrix (rows=actual, cols=predicted)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {n}" for n in CLASS_NAMES.values()],
        columns=[f"Pred {n}"  for n in CLASS_NAMES.values()],
    )
    print(f"\n  Confusion matrix:\n{cm_df}\n")

    # Per-class metrics
    report_df = pd.DataFrame(report).T
    print(report_df.round(3))

    # Save
    cm_df.to_csv(OUT_DIR / f"confusion_{year}.csv")
    report_df.to_csv(OUT_DIR / f"accuracy_{year}.csv")

    return {"year": year, "OA": oa, "Kappa": kappa, "n_points": len(y_true)}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=== Step 05: Accuracy assessment ===\n")

    if not VALIDATION_POINTS.exists():
        print(f"ERROR: {VALIDATION_POINTS} not found.")
        print("  → Digitize validation POINTS in QGIS (separate from training).")
        print("  → Use the same 'class' attribute: seagrass | sand | rock")
        print("  → Aim for ≥50 points per class, spread across the study area.")
        sys.exit(1)

    years   = [year for year, _, _ in EPOCHS]
    summary = []

    for year in years:
        result = assess_epoch(year)
        if result:
            summary.append(result)

    if summary:
        summary_df = pd.DataFrame(summary).set_index("year")
        print(f"\n{'='*50}")
        print("  ACCURACY SUMMARY — ALL EPOCHS")
        print(f"{'='*50}")
        print(summary_df.round(4))
        summary_df.to_csv(OUT_DIR / "accuracy_summary.csv")
        print(f"\n  Summary saved → {OUT_DIR / 'accuracy_summary.csv'}")

    print("\n✓ Accuracy assessment complete — run step 06 for change detection.")
