# utils/corrections.py  —  glint correction, water mask, Lyzenga indices, GLCM

import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import zoom


# =============================================================================
# WATER MASK
# Uses NDWI proxy on RGB: (G - R) / (G + R)
# Positive values = water surface
# =============================================================================

def compute_water_mask(r, g, threshold=0.0):
    """
    Returns a boolean array: True = water pixel.
    NDWI_rgb = (G - R) / (G + R)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = np.where((g + r) > 0, (g - r) / (g + r), 0.0)
    return ndwi > threshold


# =============================================================================
# SUN-GLINT CORRECTION
# Reference: Hedley et al. (2005) RSE 102:237–244
# Full method requires NIR; with RGB only we apply minimum-subtraction
# as a first-order correction (removes additive glint offset per band).
# =============================================================================

def sunglint_correction(r, g, b):
    """
    Minimum-subtraction glint correction (RGB adaptation of Hedley 2005).
    Subtracts the per-band minimum from each band.
    Returns corrected float arrays.
    """
    r_c = r.astype(np.float64) - r.min()
    g_c = g.astype(np.float64) - g.min()
    b_c = b.astype(np.float64) - b.min()
    return r_c, g_c, b_c


# =============================================================================
# LYZENGA DEPTH-INVARIANT INDICES
# Reference: Lyzenga (1978) RSE 7:1–20
# Removes depth-dependent variation by computing log band ratios.
# Attenuation ratio k_ij estimated from variance in deep-water pixels.
# Pairs: (R, G) → DI_RG    (G, B) → DI_GB
# =============================================================================

def lyzenga_indices(r, g, b, water_mask, epsilon=1e-6):
    """
    Compute two depth-invariant spectral indices.
    Returns DI_RG and DI_GB arrays.
    """
    r = np.where(r > 0, r, epsilon)
    g = np.where(g > 0, g, epsilon)
    b = np.where(b > 0, b, epsilon)

    ln_r = np.log(r)
    ln_g = np.log(g)
    ln_b = np.log(b)

    deep = water_mask.flatten()

    def attenuation_ratio(band_a, band_b):
        a_deep = band_a.flatten()[deep]
        b_deep = band_b.flatten()[deep]
        if len(a_deep) < 100:
            return 1.0
        std_b = np.std(b_deep)
        return np.std(a_deep) / std_b if std_b > 0 else 1.0

    k_rg = attenuation_ratio(ln_r, ln_g)
    k_gb = attenuation_ratio(ln_g, ln_b)

    di_rg = ln_r - k_rg * ln_g
    di_gb = ln_g - k_gb * ln_b

    print(f"    Lyzenga k_RG={k_rg:.3f}  k_GB={k_gb:.3f}")
    return di_rg, di_gb


# =============================================================================
# GLCM TEXTURE FEATURES
# Reference: Haralick et al. (1973) IEEE Trans. SMC 3:610–621
# Computed on the green band using a sliding window.
# Subsampled 2× and interpolated back for speed.
# Features: contrast, homogeneity, energy, correlation
# =============================================================================

def compute_glcm_features(band, window=21, levels=64):
    """
    Compute GLCM texture features on a single-band image.
    Returns four arrays: contrast, homogeneity, energy, correlation.
    """
    b_scaled = (band / (band.max() + 1e-6) * (levels - 1)).astype(np.uint8)
    h, w = b_scaled.shape
    half = window // 2

    contrast    = np.zeros((h, w), dtype=np.float32)
    homogeneity = np.zeros((h, w), dtype=np.float32)
    energy      = np.zeros((h, w), dtype=np.float32)
    correlation = np.zeros((h, w), dtype=np.float32)

    # Subsample every 2 pixels for speed; interpolate back
    for i in range(half, h - half, 2):
        for j in range(half, w - half, 2):
            patch = b_scaled[i-half:i+half+1, j-half:j+half+1]
            glcm  = graycomatrix(patch, distances=[1], angles=[0],
                                 levels=levels, symmetric=True, normed=True)
            contrast[i, j]    = graycoprops(glcm, "contrast")[0, 0]
            homogeneity[i, j] = graycoprops(glcm, "homogeneity")[0, 0]
            energy[i, j]      = graycoprops(glcm, "energy")[0, 0]
            correlation[i, j] = graycoprops(glcm, "correlation")[0, 0]

    # Interpolate subsampled grid back to full resolution
    sub_h = len(range(half, h - half, 2))
    sub_w = len(range(half, w - half, 2))

    def interp(arr):
        small = arr[half:half + sub_h*2:2, half:half + sub_w*2:2]
        factor_h = h / small.shape[0]
        factor_w = w / small.shape[1]
        return zoom(small, (factor_h, factor_w), order=1)[:h, :w]

    return (interp(contrast), interp(homogeneity),
            interp(energy),   interp(correlation))
