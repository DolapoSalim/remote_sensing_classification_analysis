# Pianosa Benthic Habitat Mapping
## Seagrass · Sand · Rock — Multi-epoch change detection (2016–2024)

---

## Directory structure

```
pianosa_project/
│
├── README.md                   ← you are here
├── config.py                   ← shared settings (edit this first)
├── requirements.txt
│
├── 01_download/
│   └── download_tiles.py       ← WMS tile download & mosaic assembly
│
├── 02_preprocessing/
│   └── preprocess.py           ← glint correction, water mask, Lyzenga indices, GLCM
│
├── 03_training/
│   └── extract_samples.py      ← sample pixels from QGIS polygons
│   └── QGIS_GUIDE.md           ← step-by-step digitizing guide
│
├── 04_classification/
│   └── train_and_classify.py   ← train RF, classify all epochs
│
├── 05_validation/
│   └── accuracy_assessment.py  ← confusion matrix, OA, Kappa
│
├── 06_change_detection/
│   └── change_detection.py     ← per-epoch diff, transition matrix
│
├── 07_outputs/
│   └── make_figures.py         ← publication-quality maps & charts
│
├── utils/
│   ├── __init__.py
│   ├── io.py                   ← shared raster read/write helpers
│   └── corrections.py          ← glint, Lyzenga, water mask functions
│
├── data/
│   ├── raw/                    ← pianosa_YYYY_RGB.tif (from step 01)
│   ├── processed/              ← feature stacks (.npy) from step 02
│   ├── classified/             ← classified GeoTIFFs from step 04
│   └── change/                 ← change rasters from step 06
│
└── training_data/
    ├── training_polygons.geojson   ← digitized in QGIS (step 03)
    └── validation_points.geojson  ← separate validation set (step 05)
```

---

## Run order

```
1.  Edit config.py with your paths and settings
2.  pip install -r requirements.txt
3.  python 01_download/download_tiles.py
4.  python 02_preprocessing/preprocess.py
5.  [QGIS] Digitize training polygons → see 03_training/QGIS_GUIDE.md
6.  python 03_training/extract_samples.py
7.  python 04_classification/train_and_classify.py
8.  [QGIS] Digitize validation points (separate from training)
9.  python 05_validation/accuracy_assessment.py
10. python 06_change_detection/change_detection.py
11. python 07_outputs/make_figures.py
```

---

## Classes

| ID | Name      | QGIS colour  | Notes                          |
|----|-----------|--------------|--------------------------------|
|  1 | Seagrass  | Dark green   | Posidonia oceanica meadows     |
|  2 | Sand      | Tan/yellow   | Unconsolidated sediment        |
|  3 | Rock      | Grey-brown   | Consolidated substrate         |
|  0 | No data   | Transparent  | Land / masked pixels           |
