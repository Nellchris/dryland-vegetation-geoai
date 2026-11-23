# Explainable GeoAI for Vegetation Stress Mapping in the Sokoto Basin, Nigeria

Drought-related vegetation stress across the Sokoto Basin, Northwestern Nigeria,
modelled from climate drivers on a 5 km grid and explained with SHAP.

**▶ [Open the interactive map](https://nellchris.github.io/dryland-vegetation-geoai/)**  ·  **[Read the technical report](technical_report.md)**

The map carries nine layers — observed and predicted vegetation anomalies, model
agreement, dominant climate driver, driver influence strength, land cover and
residual hotspots — with a click-through detail card for every one of the
5,191 grid cells.

---

## Method

Two machine-learning models are run side by side: a **Random Forest (RF)** and a
**Multilayer Perceptron (MLP)** neural network. Both predict the monthly NDVI
anomaly from rainfall, soil moisture, temperature and evaporative stress
anomalies, each at lags of 0–2 months plus a 3-month accumulation, on a 5 km grid
covering 2001–2023 (1,419,806 cell-months, 22 features).

Models were trained on **2001–2019** using 25 km **spatial-block five-fold
cross-validation** — folds assigned by a periodic king-graph colouring so no two
adjacent blocks share a fold — and evaluated on an untouched **2020–2023**
temporal holdout. **SHapley Additive exPlanations (SHAP)** attribute each
prediction to its drivers: TreeSHAP for the forest (51,910 rows, all cells
covered) and KernelSHAP for the network.

| | Cross-validation R² | Holdout monthly R² | Holdout spatial *r* |
|---|---|---|---|
| Random Forest | 0.226 | **+0.007** | **+0.287** |
| Neural network | **0.334** | −0.028 | +0.196 |

Target anomalies use a per-cell, per-calendar-month climatology computed from the
**training period only**. A full-record (2001–2023) baseline was tested and
rejected: it places the holdout inside its own baseline, making train and holdout
cell-mean anomalies near-exact negatives (r = −0.998) and inverting the sign of
any learned spatial pattern.

## Key findings

- **Accumulated rainfall dominates.** The three-month rainfall anomaly is the
  strongest driver in 98% of cells (18.3% of total SHAP attribution), and
  deficits depress vegetation roughly three times more than equivalent surpluses
  lift it — an asymmetric threshold response.
- **The models agree on *where*, not *why*.** Their predicted spatial patterns
  correlate at r = 0.84, yet they share **no top-three SHAP driver**: the forest
  attributes to accumulated moisture, the network to time of year.
- **Cross-validation picked the wrong model.** The network led in
  cross-validation (R² 0.334 vs 0.226) but lost on every unseen-year metric.
  Spatial cross-validation alone would have selected it — a temporal holdout was
  required to see the reversal.
- **Disagreement did not indicate unreliability.** The gap between models barely
  correlates with actual error (r = 0.04), contradicting a common assumption in
  ensemble work.
- **Shrubland is hardest to predict, cropland easiest** (20.8% vs 4.2% of cells
  failing), though normalising by each class's own variability largely flattens
  that ranking.

Monthly skill is low in absolute terms: within any calendar month, model RMSE
approximately equals the observed standard deviation, so the models resolve
*where* stress occurs better than *how much*. Predicted layers span only ~6% of
the observed range. These limits are stated on the map itself.

## Data sources

| Source | Variable |
|---|---|
| MODIS MOD13Q1 / MYD13Q1 | Monthly NDVI, 2001–2023 |
| CHIRPS | Monthly rainfall totals |
| ERA5-Land | 2 m temperature, soil moisture, actual and potential evaporation |
| ESA WorldCover | Land cover classes |
| FAO GAUL | Sokoto, Kebbi, Katsina and Zamfara State boundaries |

All climate and vegetation data were extracted via Google Earth Engine. The
basemap is CARTO Positron (© OpenStreetMap contributors, © CARTO).

> **Note on ERA5-Land PET.** ERA5-Land potential evaporation overestimates
> FAO-56 reference ET by roughly 1.9× in semi-arid zones. Values are used as
> published, winsorised at the 99.9th percentile; the ET/PET ratio is therefore a
> valid *relative* predictor rather than a calibrated aridity index.

## Repository contents

```
index.html        interactive map (self-contained, no server or API key)
data.js           per-cell model output, SHAP attribution and boundary
scripts/          analysis pipeline, steps 0–8
figures/          cross-validation design, SHAP and residual diagnostics
results/          metrics, SHAP importances, consensus and residual tables
technical_report.md   full write-up: method, results, discussion, references
```

Raw satellite and reanalysis extracts (~1 GB) are not tracked; `scripts/`
regenerates every derived file from them.

## Attribution

**Nelson Christopher, 2026**
