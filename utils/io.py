# utils/io.py  —  shared raster read / write helpers

import numpy as np
import rasterio
from pathlib import Path


def read_rgb(tif_path):
    """
    Read RGB bands from a GeoTIFF (works for both 3-band RGB and 4-band RGBA).
    Always returns only R, G, B as float32 — alpha is ignored for analysis.
    """
    with rasterio.open(tif_path) as src:
        r = src.read(1).astype(np.float32)
        g = src.read(2).astype(np.float32)
        b = src.read(3).astype(np.float32)
        profile = src.profile
    return r, g, b, profile


def read_alpha_mask(tif_path):
    """
    Read the alpha channel from a 4-band RGBA GeoTIFF as a boolean mask.
    True = valid/opaque pixel.  False = transparent (land/background).
    Falls back to all-True if the file has only 3 bands.
    """
    with rasterio.open(tif_path) as src:
        if src.count >= 4:
            return src.read(4) > 0
        h, w = src.height, src.width
        return np.ones((h, w), dtype=bool)


def save_raster(array, profile, out_path, dtype=rasterio.uint8, nodata=0):
    """
    Save a 2-D numpy array as a single-band GeoTIFF.
    """
    out_profile = profile.copy()
    out_profile.update(
        dtype=dtype, count=1,
        compress="lzw", nodata=nodata,
        tiled=True, blockxsize=512, blockysize=512,
    )
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(array.astype(dtype), 1)
    print(f"  Saved → {out_path}")


def save_feature_stack(stack, path):
    """Save feature stack (H, W, n_features) as .npy for fast reloading."""
    np.save(path, stack.astype(np.float32))
    print(f"  Feature stack saved → {path}  shape={stack.shape}")


def load_feature_stack(path):
    """Load a .npy feature stack."""
    stack = np.load(path)
    print(f"  Feature stack loaded ← {path}  shape={stack.shape}")
    return stack
