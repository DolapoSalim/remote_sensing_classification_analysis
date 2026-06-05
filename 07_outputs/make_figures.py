# =============================================================================
# 07_outputs/make_figures.py
#
# PURPOSE : Generate publication-quality figures:
#           Fig 1  — Classified habitat maps for all epochs (grid)
#           Fig 2  — Seagrass area trend bar chart
#           Fig 3  — Change maps for each consecutive pair
#           Fig 4  — Feature importance bar chart
#
# OUTPUT  : 07_outputs/figures/fig1_classified_maps.png
#                              fig2_seagrass_trend.png
#                              fig3_change_maps.png
#                              fig4_feature_importance.png
#
# RUN     : python 07_outputs/make_figures.py
# =============================================================================

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import rasterio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (EPOCHS, CLASSIFIED_DIR, CHANGE_DIR,
                    CLASS_NAMES, CLASS_COLORS, FEATURE_NAMES)

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Colormap for classified maps
CMAP = mcolors.ListedColormap(
    ["white", CLASS_COLORS[1], CLASS_COLORS[2], CLASS_COLORS[3]]
)
BOUNDS = [0, 0.5, 1.5, 2.5, 3.5]
NORM   = mcolors.BoundaryNorm(BOUNDS, CMAP.N)

# Colormap for change maps
CHANGE_CMAP = mcolors.ListedColormap(["white", "#4CAF50", "#E53935"])
CHANGE_BOUNDS = [0, 0.5, 1.5, 2.5]
CHANGE_NORM   = mcolors.BoundaryNorm(CHANGE_BOUNDS, CHANGE_CMAP.N)


# =============================================================================
# FIGURE 1 — Classified maps grid
# =============================================================================

def fig_classified_maps():
    years = [year for year, _, _ in EPOCHS]
    n     = len(years)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                              dpi=150)
    axes = axes.flatten()

    for i, year in enumerate(years):
        path = CLASSIFIED_DIR / f"classified_{year}.tif"
        ax   = axes[i]

        if not path.exists():
            ax.text(0.5, 0.5, f"{year}\n(not found)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        with rasterio.open(path) as src:
            arr = src.read(1)

        ax.imshow(arr, cmap=CMAP, norm=NORM, interpolation="none")
        ax.set_title(year, fontsize=12, fontweight="bold")
        ax.axis("off")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color="white",            label="Land / No data"),
        mpatches.Patch(color=CLASS_COLORS[1],    label="Seagrass"),
        mpatches.Patch(color=CLASS_COLORS[2],    label="Sand"),
        mpatches.Patch(color=CLASS_COLORS[3],    label="Rock"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle("Benthic habitat classification — Pianosa",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = OUT_DIR / "fig1_classified_maps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Fig 1 → {out}")


# =============================================================================
# FIGURE 2 — Seagrass area trend
# =============================================================================

def fig_seagrass_trend():
    trend_csv = Path("06_change_detection") / "seagrass_trend.csv"
    if not trend_csv.exists():
        print("  Fig 2: seagrass_trend.csv not found — run step 06 first")
        return

    df = pd.read_csv(trend_csv, index_col="year")
    years = df.index.astype(str).tolist()
    x     = np.arange(len(years))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.bar(x - width,    df["seagrass_ha"], width, color=CLASS_COLORS[1],
           label="Seagrass", zorder=3)
    ax.bar(x,            df["sand_ha"],     width, color=CLASS_COLORS[2],
           label="Sand",     zorder=3)
    ax.bar(x + width,    df["rock_ha"],     width, color=CLASS_COLORS[3],
           label="Rock",     zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Area (hectares)", fontsize=11)
    ax.set_title("Benthic habitat area — Pianosa 2016–2024",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "fig2_seagrass_trend.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Fig 2 → {out}")


# =============================================================================
# FIGURE 3 — Change maps
# =============================================================================

def fig_change_maps():
    years = [year for year, _, _ in EPOCHS]
    pairs = [(years[i], years[i+1]) for i in range(len(years) - 1)]
    n     = len(pairs)

    if n == 0:
        print("  Fig 3: no change rasters found")
        return

    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 5 * nrows), dpi=150)
    if n == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for i, (ya, yb) in enumerate(pairs):
        path = CHANGE_DIR / f"change_{ya}_{yb}.tif"
        ax   = axes[i]

        if not path.exists():
            ax.text(0.5, 0.5, f"{ya}→{yb}\n(not found)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        with rasterio.open(path) as src:
            arr = src.read(1)

        ax.imshow(arr, cmap=CHANGE_CMAP, norm=CHANGE_NORM,
                  interpolation="none")
        ax.set_title(f"{ya} → {yb}", fontsize=11, fontweight="bold")
        ax.axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    patches = [
        mpatches.Patch(color="white",   label="Land / No data"),
        mpatches.Patch(color="#4CAF50", label="Stable"),
        mpatches.Patch(color="#E53935", label="Changed"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Habitat change detection — Pianosa",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "fig3_change_maps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Fig 3 → {out}")


# =============================================================================
# FIGURE 4 — Feature importance
# =============================================================================

def fig_feature_importance():
    fi_csv = Path("04_classification") / "feature_importances.csv"
    if not fi_csv.exists():
        print("  Fig 4: feature_importances.csv not found — run step 04 first")
        return

    fi = pd.read_csv(fi_csv, index_col=0).squeeze().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    colors  = ["#1a7a4a" if "DI" in f or "VARI" in f
               else "#5b8dd9" if "GLCM" in f
               else "#888" for f in fi.index]
    fi.plot.barh(ax=ax, color=colors, edgecolor="none")
    ax.set_xlabel("Mean decrease in impurity", fontsize=11)
    ax.set_title("Random Forest — feature importance",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # Legend
    patches = [
        mpatches.Patch(color="#888",    label="RGB bands"),
        mpatches.Patch(color="#1a7a4a", label="Spectral indices"),
        mpatches.Patch(color="#5b8dd9", label="GLCM texture"),
    ]
    ax.legend(handles=patches, frameon=False, fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / "fig4_feature_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Fig 4 → {out}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=== Step 07: Generate publication figures ===\n")
    fig_classified_maps()
    fig_seagrass_trend()
    fig_change_maps()
    fig_feature_importance()
    print(f"\n✓ Figures saved to {OUT_DIR}")
