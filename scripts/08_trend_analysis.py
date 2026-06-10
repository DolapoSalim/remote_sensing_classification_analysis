"""
08_trend_analysis.py
────────────────────
Statistical trend analysis over the 2016–2024 time series.
- Linear regression (OLS) of extent and fragmentation over time
- COVID period test: compares 2019→2021 rate vs overall trend
- Outputs a stats table for the paper methods/results section

Input:  04_extent_analysis/outputs/extent_per_class_per_year.csv
        05_fragmentation_analysis/outputs/fragmentation_trends.csv
Output: 04_extent_analysis/outputs/trend_regression_results.csv
        05_fragmentation_analysis/outputs/fragmentation_regression_results.csv
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

EXTENT_CSV   = "04_extent_analysis/outputs/extent_per_class_per_year.csv"
FRAG_CSV     = "05_fragmentation_analysis/outputs/fragmentation_trends.csv"
EXTENT_OUT   = "04_extent_analysis/outputs/trend_regression_results.csv"
FRAG_OUT     = "05_fragmentation_analysis/outputs/fragmentation_regression_results.csv"

CLASS_NAMES  = ["posidonia", "rock", "sand"]

# COVID period: between 2019 and 2021 flights
COVID_YEARS  = ("2019", "2021")


def linear_regression(x, y):
    """OLS regression. Returns slope, intercept, r², p-value, se."""
    if len(x) < 3:
        return None
    slope, intercept, r, p, se = stats.linregress(x, y)
    return {
        "slope":     round(slope, 6),
        "intercept": round(intercept, 4),
        "r2":        round(r**2, 4),
        "p_value":   round(p, 4),
        "std_err":   round(se, 6),
        "n":         len(x),
        "significant": p < 0.05,
    }


def covid_test(df_class, metric_col):
    """
    Compare the 2019→2021 rate of change to the overall trend rate.
    A significant difference suggests a COVID effect.
    """
    sub = df_class.sort_values("year_num")

    # Overall annual rate (slope from regression)
    reg = linear_regression(
        sub["year_num"].values,
        sub[metric_col].values
    )
    if reg is None:
        return None

    overall_slope = reg["slope"]

    # COVID interval rate
    pre  = sub[sub["year"] == COVID_YEARS[0]]
    post = sub[sub["year"] == COVID_YEARS[1]]

    if pre.empty or post.empty:
        return None

    covid_delta = (post[metric_col].values[0] -
                   pre[metric_col].values[0])
    covid_yrs   = (post["year_num"].values[0] -
                   pre["year_num"].values[0])
    covid_rate  = covid_delta / covid_yrs if covid_yrs > 0 else np.nan

    return {
        "overall_annual_rate": round(overall_slope, 6),
        "covid_annual_rate":   round(covid_rate, 6),
        "covid_vs_trend":      round(covid_rate - overall_slope, 6),
        "covid_pct_diff":      round(
            (covid_rate - overall_slope) / abs(overall_slope) * 100
            if overall_slope != 0 else np.nan, 2),
    }


def analyse_extent():
    if not os.path.exists(EXTENT_CSV):
        print(f"No extent data at {EXTENT_CSV} — run 05 first.")
        return

    df = pd.read_csv(EXTENT_CSV)
    rows = []

    print("── Extent trend analysis ─────────────────────────────────")
    for cls in CLASS_NAMES:
        sub = df[df["class"] == cls].sort_values("year_num")
        if sub.empty or len(sub) < 3:
            continue

        x = sub["year_num"].values
        y = sub["hectares"].values

        reg = linear_regression(x, y)
        if reg is None:
            continue

        covid = covid_test(sub, "hectares")

        sig = "✓" if reg["significant"] else "✗"
        print(f"\n  {cls.upper()}")
        print(f"    Slope      : {reg['slope']:+.4f} ha/yr")
        print(f"    R²         : {reg['r2']:.4f}")
        print(f"    p-value    : {reg['p_value']:.4f}  {sig} significant")
        if covid:
            print(f"    Overall rate    : {covid['overall_annual_rate']:+.4f} ha/yr")
            print(f"    COVID rate      : {covid['covid_annual_rate']:+.4f} ha/yr")
            print(f"    COVID vs trend  : {covid['covid_vs_trend']:+.4f} ha/yr "
                  f"({covid['covid_pct_diff']:+.1f}%)")

        row = {"class": cls, "variable": "hectares", **reg}
        if covid:
            row.update({f"covid_{k}": v for k, v in covid.items()})
        rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(EXTENT_OUT, index=False)
        print(f"\n✓ Extent regression results → {EXTENT_OUT}")


def analyse_fragmentation():
    if not os.path.exists(FRAG_CSV):
        print(f"No fragmentation data at {FRAG_CSV} — run 07 first.")
        return

    df   = pd.read_csv(FRAG_CSV)
    rows = []

    frag_metrics = [
        ("n_patches",     "Number of patches"),
        ("mesh_ha",       "Effective mesh size (ha)"),
        ("edge_density",  "Edge density (m/ha)"),
        ("mean_area_ha",  "Mean patch area (ha)"),
    ]

    print("\n── Fragmentation trend analysis ──────────────────────────")
    for cls in CLASS_NAMES:
        sub = df[df["class"] == cls].sort_values("year_num")
        if sub.empty or len(sub) < 3:
            continue

        print(f"\n  {cls.upper()}")
        for metric_col, metric_label in frag_metrics:
            if metric_col not in sub.columns:
                continue

            y = sub[metric_col].dropna().values
            x = sub.loc[sub[metric_col].notna(), "year_num"].values

            if len(x) < 3:
                continue

            reg   = linear_regression(x, y)
            covid = covid_test(sub, metric_col)

            if reg is None:
                continue

            sig = "✓" if reg["significant"] else "✗"
            print(f"    {metric_label:<30}: "
                  f"slope={reg['slope']:+.6f}  "
                  f"R²={reg['r2']:.3f}  "
                  f"p={reg['p_value']:.4f} {sig}")

            row = {
                "class":    cls,
                "variable": metric_col,
                "label":    metric_label,
                **reg
            }
            if covid:
                row.update(
                    {f"covid_{k}": v for k, v in covid.items()})
            rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(FRAG_OUT, index=False)
        print(f"\n✓ Fragmentation regression results → {FRAG_OUT}")


def main():
    analyse_extent()
    analyse_fragmentation()
    print("\nTrend analysis complete.")


if __name__ == "__main__":
    main()
