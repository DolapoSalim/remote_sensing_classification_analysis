# =============================================================================
# 04_classification/train_and_classify.py
#
# PURPOSE : 1. Load training samples (from step 03)
#           2. Train a Random Forest classifier with cross-validation
#           3. Apply the trained classifier to every epoch
#           4. Save classified GeoTIFFs
#
# OUTPUT  : data/classified/classified_YYYY.tif
#           04_classification/rf_model.pkl
#           04_classification/scaler.pkl
#           04_classification/feature_importances.csv
#           04_classification/cv_results.txt
#
# RUN     : python 04_classification/train_and_classify.py
# =============================================================================

import sys
import numpy as np
import pandas as pd
import joblib
import rasterio
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, cohen_kappa_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (EPOCHS, TRAINING_DIR, PROCESSED_DIR, CLASSIFIED_DIR,
                    RAW_DIR, CLASS_MAP, CLASS_NAMES, FEATURE_NAMES, RF_PARAMS)
from utils.io import load_feature_stack, save_raster

MODEL_DIR = Path(__file__).parent


# =============================================================================
# TRAIN
# =============================================================================

def train_classifier(X, y):
    print("\n--- Training Random Forest ---")
    print(f"  Samples   : {len(y):,}")
    print(f"  Features  : {X.shape[1]}")
    print(f"  RF params : {RF_PARAMS}")

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    rf = RandomForestClassifier(**RF_PARAMS)

    # 5-fold stratified cross-validation reporting OA and Kappa
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "kappa":    make_scorer(cohen_kappa_score),
    }
    cv = cross_validate(rf, X_sc, y, cv=skf, scoring=scoring,
                        return_train_score=False)

    oa_mean    = cv["test_accuracy"].mean()
    oa_std     = cv["test_accuracy"].std()
    kap_mean   = cv["test_kappa"].mean()
    kap_std    = cv["test_kappa"].std()

    print(f"\n  Cross-validation (5-fold stratified):")
    print(f"    Overall Accuracy : {oa_mean:.4f} ± {oa_std:.4f}")
    print(f"    Cohen's Kappa    : {kap_mean:.4f} ± {kap_std:.4f}")

    # Save CV results for the paper
    cv_text = (
        f"5-fold stratified cross-validation\n"
        f"Overall Accuracy : {oa_mean:.4f} ± {oa_std:.4f}\n"
        f"Cohen's Kappa    : {kap_mean:.4f} ± {kap_std:.4f}\n"
        f"Per-fold OA      : {cv['test_accuracy'].round(4).tolist()}\n"
        f"Per-fold Kappa   : {cv['test_kappa'].round(4).tolist()}\n"
    )
    (MODEL_DIR / "cv_results.txt").write_text(cv_text)
    print(f"  CV results → {MODEL_DIR / 'cv_results.txt'}")

    # Final fit on all training data
    rf.fit(X_sc, y)

    # Feature importance
    fi = pd.Series(rf.feature_importances_, index=FEATURE_NAMES
                   ).sort_values(ascending=False)
    print("\n  Feature importances:")
    print(fi.to_string())
    fi.to_csv(MODEL_DIR / "feature_importances.csv", header=["importance"])

    return rf, scaler


# =============================================================================
# CLASSIFY ONE EPOCH
# =============================================================================

def classify_epoch(year, rf, scaler):
    feat_path = PROCESSED_DIR / f"features_{year}.npy"
    mask_path = PROCESSED_DIR / f"watermask_{year}.npy"
    out_path  = CLASSIFIED_DIR / f"classified_{year}.tif"

    if out_path.exists():
        print(f"  [SKIP] {year}: classified raster already exists")
        return

    if not feat_path.exists():
        print(f"  [SKIP] {year}: features not found — run step 02 first")
        return

    print(f"\n  Classifying {year}...")
    stack      = load_feature_stack(feat_path)
    water_mask = np.load(mask_path)
    h, w, n_f  = stack.shape

    # Flatten to pixel matrix, scale, predict
    X_all = stack.reshape(-1, n_f)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    X_sc  = scaler.transform(X_all)

    print(f"    Predicting {h*w:,} pixels...")
    pred = rf.predict(X_sc)
    classified = pred.reshape(h, w).astype(np.uint8)

    # Mask non-water pixels → class 0 (NoData)
    classified[~water_mask] = 0

    # Load profile from original TIF for georeferencing
    with rasterio.open(RAW_DIR / f"pianosa_{year}_RGB.tif") as src:
        profile = src.profile

    save_raster(classified, profile, out_path, dtype=rasterio.uint8, nodata=0)

    # Print class area summary
    total_water = water_mask.sum()
    print(f"    Class distribution (water pixels only):")
    for cid, name in CLASS_NAMES.items():
        n = (classified == cid).sum()
        print(f"      {name:>10}: {n:>8,} px  ({n/total_water*100:.1f}%)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=== Step 04: Train RF and classify all epochs ===\n")

    # Load training data
    X_path = TRAINING_DIR / "X_train.npy"
    y_path = TRAINING_DIR / "y_train.npy"

    if not X_path.exists():
        print("ERROR: X_train.npy not found — run step 03 first.")
        sys.exit(1)

    X = np.load(X_path)
    y = np.load(y_path)

    # Train
    rf, scaler = train_classifier(X, y)

    # Save model
    joblib.dump(rf,     MODEL_DIR / "rf_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    print(f"\n  Model saved → {MODEL_DIR / 'rf_model.pkl'}")

    # Classify all epochs
    print("\n--- Classifying all epochs ---")
    years = [year for year, _, _ in EPOCHS]
    for year in years:
        classify_epoch(year, rf, scaler)

    print("\n✓ Classification complete — run step 05 for accuracy assessment.")
