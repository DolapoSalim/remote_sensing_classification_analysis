# QGIS Digitizing Guide
## Delineating seagrass, sand, and rock training polygons

---

## Overview

You will do this **twice**:
1. **Training polygons** — drawn generously around confident areas → `training_polygons.geojson`
2. **Validation points** — separate points you do NOT use for training → `validation_points.geojson`

Both files must have a column called exactly `class` with values: `seagrass`, `sand`, or `rock`.

---

## Part 1 — Load the imagery in QGIS

### 1.1 Add your GeoTIFF

1. Open QGIS
2. **Layer → Add Layer → Add Raster Layer**
3. Navigate to `data/raw/pianosa_2022_RGB.tif` (use 2022 — this is your training year)
4. Click **Add**, then **Close**
5. The image should appear. If it looks grey, right-click the layer → **Properties → Symbology**:
   - Render type: **Multiband color**
   - Red band: Band 1, Green band: Band 2, Blue band: Band 3
   - Click **OK**

### 1.2 Zoom to the island

- Use the scroll wheel or **View → Zoom to Layer** to centre on Pianosa
- The island is surrounded by clear shallow water — you should see:
  - Dark green/brown patches = seagrass (*Posidonia oceanica*)
  - Bright white/cream areas = sandy seabed
  - Darker grey/brown rocky seabed = rock
  - The island itself (land) = you will NOT digitize this

### 1.3 Adjust contrast for better visibility

Right-click the raster layer → **Properties → Symbology**:
- Under **Min / Max Value Settings**, choose **Mean +/- standard deviations: 2.0**
- Click **Apply** — this stretches contrast and makes underwater features much clearer

---

## Part 2 — Create the training polygon layer

### 2.1 Create a new GeoJSON vector layer

1. **Layer → Create Layer → New GeoPackage Layer** (or GeoJSON — see 2.2)
2. Better: use **Layer → Create Layer → New Temporary Scratch Layer**:
   - Geometry type: **Polygon**
   - CRS: **EPSG:32632** (same as your raster)
   - Click **OK**

Or directly create a GeoJSON:
1. **Layer → Create Layer → New GeoJSON Layer**
2. Set:
   - **File name**: save as `training_data/training_polygons.geojson`
   - **CRS**: EPSG:32632
   - **Geometry type**: Polygon
3. Add a field:
   - Field name: `class`
   - Type: **Text (string)**
   - Length: 20
4. Click **OK**

### 2.2 Start editing

1. Select the `training_polygons` layer in the Layers panel
2. Click the **pencil icon** (Toggle Editing) in the toolbar, or press **Ctrl+E**
3. The layer should highlight to show it is in edit mode

---

## Part 3 — Draw training polygons

### 3.1 Select the digitizing tool

- In the toolbar, click **Add Polygon Feature** (the polygon icon, or press **A**)
- Your cursor will become a crosshair

### 3.2 How to draw a polygon

1. **Left-click** to place each vertex
2. Trace around the habitat patch — stay within the area where you are confident of the class
3. **Right-click** to close and finish the polygon
4. A dialog will appear asking for attributes — type the class name:
   - `seagrass`
   - `sand`
   - `rock`
5. Click **OK**

### 3.3 Visual keys — what to look for

| Class    | Appearance in true-colour RGB at 20 cm GSD |
|----------|---------------------------------------------|
| Seagrass | Dark green to dark olive-brown. Texture: mottled, fine. May have visible blade structure. Depth: typically 0–15 m. Often with clear boundaries against sand. |
| Sand     | Bright white to pale beige/cream. Texture: smooth or rippled patterns. Very high reflectance. |
| Rock     | Dark grey to dark brown. Texture: irregular, often sharp edges. May have coralline algae (red/purple tinge). Darker than seagrass but less regular. |

> **Key distinction — seagrass vs rock**: Both are dark. Seagrass has a smoother, more homogeneous texture and a greenish hue. Rock has more angular edges and irregular tone. When in doubt, **do not digitize** — only draw where you are confident.

### 3.4 Guidelines for good training data

- **Draw at least 10–15 polygons per class**, spread across different depths and locations
- **Polygons should be compact and confident** — draw them smaller and more certain rather than large and uncertain. A 20×20 pixel polygon of pure seagrass is far better than a 200×200 polygon with mixed pixels.
- **Avoid edges** — do not draw polygons that straddle the boundary between two classes. Stay well inside each habitat type.
- **Cover depth variation** — draw polygons at both shallow and deeper areas for each class, since depth affects reflectance significantly.
- **Do not use the same area for training and validation**. Leave one side of the island for validation only.

### 3.5 Recommended polygon count

| Class    | Minimum | Target |
|----------|---------|--------|
| Seagrass | 10      | 20+    |
| Sand     | 10      | 15+    |
| Rock     | 10      | 15+    |

More polygons = more training pixels = better model. But quality beats quantity — 15 clean polygons outperforms 30 mixed ones.

### 3.6 Save

- Press **Ctrl+S** or click **Save Layer Edits** after every few polygons
- When finished: **Layer → Stop Editing** → **Save**

---

## Part 4 — Check your polygons

### 4.1 Colour your polygons by class

1. Right-click `training_polygons` → **Properties → Symbology**
2. Choose **Categorized** from the dropdown
3. Column: `class`
4. Click **Classify**
5. Set colours:
   - seagrass → dark green (`#1a7a4a`)
   - sand → tan (`#e8d5a3`)
   - rock → grey-brown (`#8c7b6b`)
6. Set opacity to ~50% so you can see the raster underneath
7. Click **OK**

You should now see your polygons overlaid on the raster. Visually check that each polygon sits clearly on the right habitat type.

### 4.2 Open the attribute table

- Right-click layer → **Open Attribute Table**
- Check that the `class` column is correctly filled with `seagrass`, `sand`, or `rock` (all lowercase, no spaces or typos)
- The script will fail with a clear error if any value is wrong

---

## Part 5 — Export as GeoJSON

If you created the layer as a GeoPackage or temporary layer, export it:

1. Right-click the layer → **Export → Save Features As**
2. Format: **GeoJSON**
3. File name: `training_data/training_polygons.geojson`
4. CRS: **EPSG:32632**
5. Click **OK**

---

## Part 6 — Digitize validation POINTS (separate step)

**Do this after you finish training polygons, in a fresh area of the image.**

Validation points must be:
- **Separate from training polygons** — do not use the same areas
- **Points** (not polygons) — one pixel per point
- Same `class` attribute (`seagrass`, `sand`, `rock`)
- Target: ≥ 50 points per class

### How to create validation points

1. **Layer → Create Layer → New GeoJSON Layer**
   - File: `training_data/validation_points.geojson`
   - Geometry type: **Point**
   - CRS: EPSG:32632
   - Field: `class` (text, length 20)
2. Toggle editing (Ctrl+E)
3. **Add Point Feature** tool (or press **B**)
4. Click on a confident pixel, enter the class name
5. Spread points across different depths and areas
6. Save and export as GeoJSON

---

## Part 7 — Check the projection

Make sure your GeoJSON is in **EPSG:32632** (UTM Zone 32N).

In QGIS, check: right-click layer → **Properties → Information** → look for CRS.

If it shows EPSG:4326 (WGS84 degrees), reproject:
1. Right-click → **Export → Save Features As**
2. CRS: select **EPSG:32632**
3. Save as a new file

The Python script will fail if the projection does not match the raster.

---

## Common mistakes to avoid

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Polygons on land | Land pixels contaminate training data | Use the water mask — stay clearly in water |
| Mixed-class polygons straddling boundaries | Noisy training labels, lower accuracy | Draw smaller, stay inside one habitat type |
| Typo in `class` attribute (e.g. `Seagrass`) | Script exits with an error | Use all-lowercase exactly: `seagrass` |
| Same polygons used for training and validation | Artificially inflated accuracy — unpublishable | Keep them spatially separated |
| Too few polygons at depth | Classifier fails on deep pixels | Draw polygons across the full depth range |
| Wrong CRS | Polygons miss the raster entirely | Always use EPSG:32632 |

---

## When you are done

You should have:
```
training_data/
├── training_polygons.geojson   ← polygons, 'class' column, EPSG:32632
└── validation_points.geojson  ← points,   'class' column, EPSG:32632
```

Then run:
```
python 03_training/extract_samples.py
```
