# utils/io.py  —  shared raster read / write helpers

import numpy as np
import rasterio
from pathlib import Path


def read_rgb(tif_path):
    """
    Read a 3-band RGB GeoTIFF.
    Returns r, g, b arrays (float64) and the rasterio profile.
    """
    with rasterio.open(tif_path) as src:
        r = src.read(1).astype(np.float64)
        g = src.read(2).astype(np.float64)
        b = src.read(3).astype(np.float64)
        profile = src.profile
    return r, g, b, profile


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
