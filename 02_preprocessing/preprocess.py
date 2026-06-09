# =============================================================================
# 02_preprocessing/preprocess.py
#
# PURPOSE : For each epoch GeoTIFF:
#             1. Compute water mask (NDWI proxy)
#             2. Apply sun-glint correction (Hedley 2005, RGB adaptation)
#             3. Compute Lyzenga depth-invariant indices (DI_RG, DI_GB)
#             4. Compute VARI spectral index
#             5. Compute GLCM texture features on green band
#             6. Stack all features and save as .npy for fast reloading
#
# OUTPUT  : data/processed/features_YYYY.npy   (float32, shape H×W×10)
#           data/processed/watermask_YYYY.npy  (bool,    shape H×W)
#
# RUN     : python 02_preprocessing/preprocess.py
# =============================================================================

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (EPOCHS, RAW_DIR, PROCESSED_DIR,
                    NDWI_THRESHOLD, GLCM_WINDOW, GLCM_LEVELS, FEATURE_NAMES)
from utils.io import read_rgb, read_alpha_mask, save_feature_stack
from utils.corrections import (compute_water_mask, sunglint_correction,
                                lyzenga_indices, compute_glcm_features)


# =============================================================================
# FEATURE STACK BUILDER
# =============================================================================

def build_feature_stack(year):
    tif_path = RAW_DIR / f"pianosa_{year}_RGB.tif"
    feat_out  = PROCESSED_DIR / f"features_{year}.npy"
    mask_out  = PROCESSED_DIR / f"watermask_{year}.npy"

    if feat_out.exists() and mask_out.exists():
        print(f"  [SKIP] {year} — already preprocessed")
        return

    print(f"\n── Preprocessing {year} ──")

    # 1. Load raw RGB (alpha handled separately)
    r_raw, g_raw, b_raw, profile = read_rgb(tif_path)
    print(f"  Image size: {r_raw.shape[1]} × {r_raw.shape[0]} px")

    # 2. Alpha mask from the RGBA GeoTIFF (land = False, transparent pixels)
    #    This is the primary land mask — it's exactly what the server sent.
    alpha_mask = read_alpha_mask(tif_path)
    print(f"  Alpha mask (non-transparent): {alpha_mask.mean()*100:.1f}% of pixels")

    # 3. NDWI water mask as a secondary filter within the valid (non-land) area.
    #    Combines with alpha: a pixel is "water" only if it is both
    #    non-transparent AND has a positive NDWI value.
    ndwi_mask  = compute_water_mask(r_raw, g_raw, threshold=NDWI_THRESHOLD)
    water_mask = alpha_mask & ndwi_mask
    water_pct  = water_mask.mean() * 100
    print(f"  Water mask (alpha + NDWI):    {water_pct:.1f}% of pixels")

    # 3. Sun-glint correction
    print("  Applying sun-glint correction...")
    r, g, b = sunglint_correction(r_raw, g_raw, b_raw)

    # 4. Lyzenga depth-invariant indices
    print("  Computing Lyzenga depth-invariant indices...")
    di_rg, di_gb = lyzenga_indices(r, g, b, water_mask)

    # 5. VARI — Visible Atmospherically Resistant Index
    #    VARI = (G - R) / (G + R - B)
    print("  Computing VARI...")
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = g + r - b
        vari  = np.where(denom != 0, (g - r) / denom, 0.0)
    vari = np.clip(vari, -1, 1)

    # 6. GLCM texture (on green band, most informative for seagrass)
    print(f"  Computing GLCM texture (window={GLCM_WINDOW}px) — may take ~2 min...")
    g_water = g.copy()
    g_water[~water_mask] = 0   # restrict texture to water pixels
    contrast, homogeneity, energy, correlation = compute_glcm_features(
        g_water, window=GLCM_WINDOW, levels=GLCM_LEVELS
    )

    # 7. Stack all features — shape (H, W, 10)
    stack = np.stack([
        r, g, b,
        di_rg, di_gb,
        vari,
        contrast, homogeneity, energy, correlation,
    ], axis=-1).astype(np.float32)

    # Replace any NaN/inf from log transforms
    stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Feature names: {FEATURE_NAMES}")

    # 8. Save
    save_feature_stack(stack, feat_out)
    np.save(mask_out, water_mask)
    print(f"  Water mask saved → {mask_out}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    years = [year for year, _, _ in EPOCHS]
    print(f"Preprocessing {len(years)} epochs: {years}")
    print(f"Output: {PROCESSED_DIR}\n")

    for year in years:
        tif_path = RAW_DIR / f"pianosa_{year}_RGB.tif"
        if not tif_path.exists():
            print(f"  [SKIP] {year}: {tif_path.name} not found — run step 01 first")
            continue
        build_feature_stack(year)

    print("\n✓ Preprocessing complete.")
