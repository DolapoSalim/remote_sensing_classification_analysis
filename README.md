# Pianosa Habitat Change Analysis

## Overview
This pipeline quantifies Posidonia oceanica habitat extent and fragmentation
change over Pianosa island (Tuscan Archipelago) from 2016 to 2024, using
high-resolution aerial imagery (20cm GSD) from Regione Toscana WMS.

## Data sources
- Aerial imagery: Regione Toscana WMS (owsofc / owsofc_rt), 20cm GSD
- Classification: GEE RF classifier (aerial + Sentinel-2, parallel pipelines)
- Ground truth: Field-digitised polygons in QGIS (pianosa_GT.shp)

## Epochs
2016, 2019, 2021, 2022, 2023, 2024

## Classes
1 = posidonia
2 = rock
3 = sand

## Pipeline order
01_download_wms.py         → 00_raw_imagery/
02_gt_split.py             → 01_ground_truth/
03_mask_creation.py        → 03_masks/
04_accuracy_assessment.py  → 06_accuracy_assessment/
05_extent_calculation.py   → 04_extent_analysis/outputs/
06_transition_matrix.py    → 04_extent_analysis/outputs/
07_fragmentation.py        → 05_fragmentation_analysis/outputs/
08_trend_analysis.py       → 04_extent_analysis/ + 05_fragmentation_analysis/
09_figures.py              → 07_paper_figures/

## Setup
conda env create -f environment.yml
conda activate pianosa_habitat

## Citation
Aerial imagery: 'ortofoto 20cm copyright Regione Toscana / AGEA / Consorzio TeA'
