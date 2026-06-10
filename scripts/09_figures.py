"""
09_figures.py
─────────────
Generates all publication-quality figures for the paper.

Fig 1 — Study area map (Pianosa location in Tuscan Archipelago)
Fig 2 — Classified habitat maps grid (6 epochs side by side)
Fig 3 — Habitat extent time series (area per class per year)
Fig 4 — Gain / loss bar chart per epoch transition
Fig 5 — Fragmentation metrics time series (NP, MESH, ED)
Fig 6 — COVID period highlight (2019–2021 vs overall trend)

Input:  04_extent_analysis/outputs/extent_per_class_per_year.csv
        04_extent_analysis/outputs/gain_loss_all_epochs.csv
        05_fragmentation_analysis/outputs/fragmentation_trends.csv
        02_classified_maps/from_gee/pianosa_*_classified.tif
Output: 07_paper_figures/fig{1-6}_*.png  (300 dpi, ready for submission)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import rasterio

# ── Paths ─────────────────────────────────────────────────────────────
EXTENT_CSV  = "04_extent_analysis/outputs/extent_per_class_per_year.csv"
GL_CSV      = "04_extent_analysis/outputs/gain_loss_all_epochs.csv"
FRAG_CSV    = "05_fragmentation_analysis/outputs/fragmentation_trends.csv"
CLASS_DIR   = "02_classified_maps/from_gee"
OUT_DIR     = "07_paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

EPOCHS = ["2016", "2019", "2021", "2022", "2023", "2024"]

# ── Style ─────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "posidonia": "#2d6a4f",
    "rock":      "#6b6b6b",
    "sand":      "#e9c46a",
}
CLASS_LABELS = {
    "posidonia": "Posidonia oceanica",
    "rock":      "Rock",
    "sand":      "Sand",
}

# Classified map palette (matches GEE output class IDs)
CMAP_CLASSIFIED = mcolors.ListedColormap(
    ["#000000",           # 0 = nodata/background
     "#2d6a4f",           # 1 = posidonia
     "#6b6b6b",           # 2 = rock
     "#e9c46a"])          # 3 = sand

plt.rcParams.update({
    "font.family":  "sans-serif",
    "font.size":    10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi":   150,
})

DPI_OUT = 300   # output resolution for submission


# ── Figure 2: Classified maps grid ────────────────────────────────────
def fig2_classified_maps():
    available = [y for y in EPOCHS
                 if os.path.exists(
                     os.path.join(CLASS_DIR,
                                  f"pianosa_{y}_classified.tif"))]
    if not available:
        print("  [SKIP] No classified maps yet")
        return

    n = len(available)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    bounds = [0, 0.5, 1.5, 2.5, 3.5]
    norm   = mcolors.BoundaryNorm(bounds, CMAP_CLASSIFIED.N)

    for i, year in enumerate(available):
        ax = axes[i]
        path = os.path.join(CLASS_DIR, f"pianosa_{year}_classified.tif")
        with rasterio.open(path) as src:
            arr = src.read(1).astype(float)
            nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = 0

        ax.imshow(arr, cmap=CMAP_CLASSIFIED, norm=norm,
                  interpolation="nearest")
        ax.set_title(year, fontweight="bold")
        ax.axis("off")

    # Hide unused axes
    for j in range(len(available), len(axes)):
        axes[j].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=CLASS_COLORS["posidonia"],
                       label=CLASS_LABELS["posidonia"]),
        mpatches.Patch(color=CLASS_COLORS["rock"],
                       label=CLASS_LABELS["rock"]),
        mpatches.Patch(color=CLASS_COLORS["sand"],
                       label=CLASS_LABELS["sand"]),
    ]
    fig.legend(handles=patches, loc="lower center",
               ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle("Habitat classification — Pianosa (2016–2024)",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])

    out = os.path.join(OUT_DIR, "fig2_classified_maps_grid.png")
    fig.savefig(out, dpi=DPI_OUT, bbox_inches="tight")
    plt.close()
    print(f"  ✓ fig2 → {out}")


# ── Figure 3: Extent time series ──────────────────────────────────────
def fig3_extent_timeseries():
    if not os.path.exists(EXTENT_CSV):
        print(f"  [SKIP] {EXTENT_CSV} not found")
        return

    df  = pd.read_csv(EXTENT_CSV)
    fig, ax = plt.subplots(figsize=(8, 5))

    for cls in ["posidonia", "rock", "sand"]:
        sub = df[df["class"] == cls].sort_values("year_num")
        if sub.empty:
            continue
        ax.plot(sub["year_num"], sub["hectares"],
                marker="o", linewidth=2, markersize=6,
                color=CLASS_COLORS[cls],
                label=CLASS_LABELS[cls])
        ax.fill_between(sub["year_num"], sub["hectares"],
                        alpha=0.08, color=CLASS_COLORS[cls])

    # COVID shading
    ax.axvspan(2019.5, 2021.5, alpha=0.10, color="steelblue",
               label="COVID period (2019–2021)")

    ax.set_xlabel("Year")
    ax.set_ylabel("Area (hectares)")
    ax.set_title("Habitat extent over time — Pianosa island",
                 fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "fig3_extent_timeseries.png")
    fig.savefig(out, dpi=DPI_OUT, bbox_inches="tight")
    plt.close()
    print(f"  ✓ fig3 → {out}")


# ── Figure 4: Gain / loss bars ────────────────────────────────────────
def fig4_gain_loss():
    if not os.path.exists(GL_CSV):
        print(f"  [SKIP] {GL_CSV} not found")
        return

    df = pd.read_csv(GL_CSV)
    pos = df[df["class"] == "posidonia"].copy()
    pos["period"] = pos["year1"] + "→" + pos["year2"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x      = np.arange(len(pos))
    width  = 0.25

    ax.bar(x - width, pos["gain_ha"],  width, label="Gain",
           color="#52b788", alpha=0.85)
    ax.bar(x,          pos["loss_ha"],  width, label="Loss",
           color="#e63946", alpha=0.85)
    ax.bar(x + width,  pos["net_ha"],   width, label="Net change",
           color=["#52b788" if v >= 0 else "#e63946"
                  for v in pos["net_ha"]], alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pos["period"], rotation=30, ha="right")
    ax.set_ylabel("Area (hectares)")
    ax.set_title("Posidonia oceanica gain / loss per interval",
                 fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "fig4_gain_loss_bar.png")
    fig.savefig(out, dpi=DPI_OUT, bbox_inches="tight")
    plt.close()
    print(f"  ✓ fig4 → {out}")


# ── Figure 5: Fragmentation metrics ───────────────────────────────────
def fig5_fragmentation():
    if not os.path.exists(FRAG_CSV):
        print(f"  [SKIP] {FRAG_CSV} not found")
        return

    df  = pd.read_csv(FRAG_CSV)
    pos = df[df["class"] == "posidonia"].sort_values("year_num")
    if pos.empty:
        print("  [SKIP] No Posidonia fragmentation data")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    metrics = [
        ("n_patches",    "Number of patches",        "NP"),
        ("mesh_ha",      "Effective mesh size (ha)", "MESH"),
        ("edge_density", "Edge density (m/ha)",      "ED"),
    ]

    for ax, (col, label, short) in zip(axes, metrics):
        if col not in pos.columns:
            ax.set_visible(False)
            continue
        ax.plot(pos["year_num"], pos[col],
                marker="o", linewidth=2, markersize=7,
                color=CLASS_COLORS["posidonia"])
        ax.axvspan(2019.5, 2021.5, alpha=0.10, color="steelblue")
        ax.set_xlabel("Year")
        ax.set_ylabel(label)
        ax.set_title(f"{short} — Posidonia", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Fragmentation metrics — Pianosa island",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "fig5_fragmentation_timeseries.png")
    fig.savefig(out, dpi=DPI_OUT, bbox_inches="tight")
    plt.close()
    print(f"  ✓ fig5 → {out}")


# ── Figure 6: COVID period highlight ──────────────────────────────────
def fig6_covid_highlight():
    if not os.path.exists(EXTENT_CSV):
        print(f"  [SKIP] {EXTENT_CSV} not found")
        return

    df  = pd.read_csv(EXTENT_CSV)
    pos = df[df["class"] == "posidonia"].sort_values("year_num")
    if pos.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(pos["year_num"], pos["hectares"],
            marker="o", linewidth=2.5, markersize=8,
            color=CLASS_COLORS["posidonia"],
            label=CLASS_LABELS["posidonia"])

    # COVID band
    ax.axvspan(2019.5, 2021.5, alpha=0.15, color="steelblue",
               label="COVID maritime\nshutdown")
    ax.axvline(2019.5, color="steelblue", linestyle="--",
               linewidth=1, alpha=0.6)
    ax.axvline(2021.5, color="steelblue", linestyle="--",
               linewidth=1, alpha=0.6)

    # Annotate
    pre  = pos[pos["year"] == "2019"]["hectares"].values
    post = pos[pos["year"] == "2021"]["hectares"].values
    if len(pre) and len(post):
        delta = post[0] - pre[0]
        sign  = "+" if delta >= 0 else ""
        ax.annotate(f"{sign}{delta:.1f} ha",
                    xy=(2020.5, (pre[0] + post[0]) / 2),
                    ha="center", fontsize=9,
                    color="steelblue", fontweight="bold")

    ax.set_xlabel("Year")
    ax.set_ylabel("Posidonia area (hectares)")
    ax.set_title("Posidonia extent — COVID recovery signal",
                 fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "fig6_covid_period.png")
    fig.savefig(out, dpi=DPI_OUT, bbox_inches="tight")
    plt.close()
    print(f"  ✓ fig6 → {out}")


def main():
    print("Generating paper figures ...\n")
    fig2_classified_maps()
    fig3_extent_timeseries()
    fig4_gain_loss()
    fig5_fragmentation()
    fig6_covid_highlight()
    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
