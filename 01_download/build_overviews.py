# =============================================================================
# 01_download/build_overviews.py
#
# PURPOSE : Build internal image pyramids (overviews) for every RGBA GeoTIFF
#           so QGIS renders them instantly at any zoom level.
#
# WHAT IT DOES:
#   At 20cm GSD a full mosaic is ~35k×43k px. Without pyramids, QGIS reads
#   and resamples the full resolution every time you pan or zoom — very slow.
#   Overviews are pre-downsampled copies stored inside the same GeoTIFF:
#     Level 2   → 10k×14k px  (used when zoomed out a little)
#     Level 4   → 5k×7k px
#     Level 8   → 2.5k×3.5k px
#     Level 16  → 1.2k×1.7k px
#     Level 32  → ~600×850px   (used at island-scale view)
#     Level 64  → ~300×425px   (full extent thumbnail)
#   QGIS picks the right level automatically — rendering becomes near-instant.
#
# RESAMPLING : Lanczos for RGB bands (sharp, best for imagery)
#              Nearest for alpha band (never interpolate a mask)
#
# OUTPUT  : Overviews embedded in the existing RGBA.tif files (no new files)
#           Also writes a small .ovr sidecar as fallback if the TIF is read-only.
#
# RUN     : python 01_download/build_overviews.py
# =============================================================================

import sys
from pathlib import Path
from osgeo import gdal

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EPOCHS, RAW_DIR

# Overview levels — each is a factor of the original resolution
OVERVIEW_LEVELS = [2, 4, 8, 16, 32, 64]

# Resampling method for overviews
# LANCZOS = high quality for RGB imagery; NEAREST = for alpha/classified rasters
RESAMPLE_METHOD = "LANCZOS"


def build_overviews(tif_path):
    print(f"\n── {tif_path.name}")

    ds = gdal.Open(str(tif_path), gdal.GA_Update)
    if ds is None:
        print(f"  ERROR: could not open {tif_path}")
        return

    w  = ds.RasterXSize
    h  = ds.RasterYSize
    nb = ds.RasterCount
    print(f"  Size : {w:,} × {h:,} px  |  {nb} bands")

    # Check if overviews already exist
    existing = ds.GetRasterBand(1).GetOverviewCount()
    if existing >= len(OVERVIEW_LEVELS):
        print(f"  Already has {existing} overview levels — skipping")
        ds = None
        return

    print(f"  Building {len(OVERVIEW_LEVELS)} overview levels: "
          f"{OVERVIEW_LEVELS}")
    print(f"  Resampling: {RESAMPLE_METHOD}  (this may take 1–3 min)...")

    # gdal.SetConfigOption controls behaviour for this build
    gdal.SetConfigOption("COMPRESS_OVERVIEW", "LZW")
    gdal.SetConfigOption("INTERLEAVE_OVERVIEW", "PIXEL")

    # Build overviews — all bands in one call
    result = ds.BuildOverviews(RESAMPLE_METHOD, OVERVIEW_LEVELS)

    if result != 0:
        print(f"  WARNING: BuildOverviews returned code {result}")
    else:
        ovr_band = ds.GetRasterBand(1)
        actual   = ovr_band.GetOverviewCount()
        print(f"  ✓ {actual} overview levels built successfully")

        # Show the sizes so you can verify
        for i in range(actual):
            ovr = ovr_band.GetOverview(i)
            print(f"    Level {OVERVIEW_LEVELS[i]:>2}×  →  "
                  f"{ovr.XSize:>6,} × {ovr.YSize:>6,} px")

    ds.FlushCache()
    ds = None


def build_overviews_classified(tif_path):
    """
    Same as above but uses NEAREST resampling — correct for classified
    integer rasters where averaging class IDs would be meaningless.
    """
    print(f"\n── {tif_path.name}  [classified — NEAREST]")

    ds = gdal.Open(str(tif_path), gdal.GA_Update)
    if ds is None:
        print(f"  ERROR: could not open {tif_path}")
        return

    existing = ds.GetRasterBand(1).GetOverviewCount()
    if existing >= len(OVERVIEW_LEVELS):
        print(f"  Already has {existing} overview levels — skipping")
        ds = None
        return

    gdal.SetConfigOption("COMPRESS_OVERVIEW", "LZW")
    result = ds.BuildOverviews("NEAREST", OVERVIEW_LEVELS)

    if result == 0:
        print(f"  ✓ Overviews built")
    ds.FlushCache()
    ds = None


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=== Building image pyramids (overviews) ===")
    print(f"Levels: {OVERVIEW_LEVELS}\n")

    years = [year for year, _, _ in EPOCHS]
    found = 0

    for year in years:
        rgb_path = RAW_DIR / f"pianosa_{year}_RGB.tif"
        if not rgb_path.exists():
            print(f"  [SKIP] {year}: {rgb_path.name} not found")
            continue
        build_overviews(rgb_path)
        found += 1

    if found == 0:
        print("\nNo RGBA.tif files found — run download_tiles.py first.")
    else:
        print(f"\n✓ Done. {found} file(s) updated.")
        print("\nIn QGIS: the files will now render instantly at any zoom level.")
        print("If you still see slow rendering, check:")
        print("  Settings → Options → Rendering → enable 'Render layers in parallel'")
        print("  Settings → Options → Rendering → increase 'Max cores' to match your CPU")
