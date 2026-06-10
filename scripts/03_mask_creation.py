"""
03_mask_creation.py
───────────────────
Rasterises the shallow water study zone shapefile (digitised in QGIS)
to match the exact grid of the classified GeoTIFFs from GEE.
The mask is applied to all classified maps before extent/fragmentation
analysis to restrict analysis to the valid study zone only.

Input:  03_masks/pianosa_shallow_water.shp   (digitised in QGIS)
        02_classified_maps/from_gee/pianosa_2021_classified.tif  (reference grid)
Output: 03_masks/pianosa_study_zone.tif
        03_masks/pianosa_study_zone.gpkg     (clean vector version)
"""

import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import geopandas as gpd
import numpy as np
import os

MASK_SHP   = "03_masks/pianosa_shallow_water.shp"
REF_TIF    = "02_classified_maps/from_gee/pianosa_2021_classified.tif"
OUT_RASTER = "03_masks/pianosa_study_zone.tif"
OUT_VECTOR = "03_masks/pianosa_study_zone.gpkg"

# Fallback: if no reference classified TIF yet, use raw imagery
FALLBACK_REF = "00_raw_imagery/pianosa_2021_RGB.tif"


def get_reference_profile():
    """Load spatial profile from reference GeoTIFF."""
    ref = REF_TIF if os.path.exists(REF_TIF) else FALLBACK_REF
    if not os.path.exists(ref):
        raise FileNotFoundError(
            f"No reference raster found. Run 01_download_wms.py first."
        )
    with rasterio.open(ref) as src:
        profile   = src.profile.copy()
        transform = src.transform
        shape     = (src.height, src.width)
        crs       = src.crs
    print(f"Reference grid from: {ref}")
    print(f"  Shape: {shape[1]} x {shape[0]} pixels")
    print(f"  CRS:   {crs}")
    return profile, transform, shape, crs


def main():
    if not os.path.exists(MASK_SHP):
        print(f"ERROR: Shallow water shapefile not found at {MASK_SHP}")
        print("Digitise the study zone in QGIS first (Option A mask).")
        print("\nInstructions:")
        print("  1. In QGIS, create a new polygon layer (EPSG:32632)")
        print("  2. Draw a polygon covering the island + shallow shelf")
        print("     (roughly 0–20m depth around the island perimeter)")
        print("  3. Save as 03_masks/pianosa_shallow_water.shp")
        print("  4. Re-run this script")
        return

    print("Loading shallow water mask ...")
    mask_gdf = gpd.read_file(MASK_SHP)
    print(f"  {len(mask_gdf)} polygon(s) loaded")
    print(f"  CRS: {mask_gdf.crs}")

    profile, transform, shape, crs = get_reference_profile()

    # Reproject mask to match reference if needed
    if mask_gdf.crs != crs:
        print(f"  Reprojecting mask from {mask_gdf.crs} to {crs} ...")
        mask_gdf = mask_gdf.to_crs(crs)

    # Save clean vector version
    mask_gdf.to_file(OUT_VECTOR, driver="GPKG")
    print(f"  ✓ Vector mask → {OUT_VECTOR}")

    # Rasterise
    print("Rasterising mask ...")
    shapes = [(geom, 1) for geom in mask_gdf.geometry]
    mask_arr = rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,          # 0 = outside study zone
        dtype=np.uint8
    )

    valid_px = mask_arr.sum()
    total_px = mask_arr.size
    print(f"  Valid pixels: {valid_px:,} / {total_px:,} "
          f"({100*valid_px/total_px:.1f}%)")

    # Write output
    out_profile = {
        "driver":    "GTiff",
        "dtype":     "uint8",
        "width":     shape[1],
        "height":    shape[0],
        "count":     1,
        "crs":       crs,
        "transform": transform,
        "compress":  "LZW",
        "nodata":    0,
    }
    with rasterio.open(OUT_RASTER, "w", **out_profile) as dst:
        dst.write(mask_arr, 1)

    print(f"  ✓ Raster mask → {OUT_RASTER}")
    print("\nMask creation complete.")
    print("Apply this mask in scripts 04–07 before any analysis.")


if __name__ == "__main__":
    main()
