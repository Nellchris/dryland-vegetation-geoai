# Explainable GeoAI for Drought-Related Vegetation Stress in the Sokoto Basin, Northwestern Nigeria

**Nelson Christopher, 2026**

Interactive map: <https://nellchris.github.io/dryland-vegetation-geoai/>
Code and results: <https://github.com/Nellchris/dryland-vegetation-geoai>

---

## 1. Introduction and study area

Vegetation in the Sudano-Sahelian drylands of Northwestern Nigeria is governed by
a short, intense monsoon and is among the most climatically exposed in West
Africa. Monitoring where and why vegetation departs from its seasonal norm
matters for drought early warning, rangeland management and agricultural
planning. Satellite vegetation indices make the *where* observable; what they do
not supply is the *why*.

This study pairs two machine-learning models with model-agnostic attribution to
address both questions together. The aim is not only to predict monthly
vegetation anomalies from climate drivers, but to quantify which drivers carry
the signal, where the models fail, and whether two different learners tell the
same causal story.

### Study area

The analysis covers the Sokoto Basin across four Nigerian states — **Sokoto,
Kebbi, Katsina and Zamfara** (92 Local Government Areas, FAO GAUL boundaries
dissolved to a single outline).

| Property | Value |
|---|---|
| Extent | 3.526–8.961 °E, 10.129–13.857 °N |
| Analysis grid | 5,191 cells at 0.044916° ≈ **4.89 × 4.97 km** |
| Period | 2001–2023, monthly (276 time steps) |
| Mean annual rainfall | 832 mm, Jul–Sep dominated |
| Mean temperature | 27.9 °C (range 18.7–35.0 °C) |
| Dominant land cover | Cropland 3,482 cells (67%), Grassland 993, Shrubland 664 |

The basin shows textbook monsoon phenology: rainfall rises from ~0 mm in January
to 251 mm in August; NDVI follows with a lag, troughing at 0.208 in March and
peaking at 0.524 in September.

---

## 2. Data sources and preprocessing

### Sources

| Source | Variables | Native resolution |
|---|---|---|
| MODIS MOD13Q1 / MYD13Q1 | NDVI | 250 m, 16-day |
| CHIRPS | Precipitation | 0.05°, monthly |
| ERA5-Land | 2 m temperature, volumetric soil water (layer 1), actual and potential evaporation | 0.1°, monthly |
| ESA WorldCover | Land cover class, cropland fraction | 10 m |
| FAO GAUL | Administrative boundaries | vector |

All raster extraction was performed in Google Earth Engine and exported as per-year
CSV tables on the common 5 km grid.

### Verification

All 70 expected files were present with no missing years. Coordinate joins across
the three sources matched losslessly: **zero** yearly rows carried coordinates
absent from the static grid, and there were no duplicate `(cell, year, month)`
records. CHIRPS and ERA5-Land were complete at 62,292 rows per year
(5,191 × 12); MODIS was short by 82–159 rows per year (0.13–0.26%) from cloud and
quality masking, giving 1,430,153 of a possible 1,432,716 cell-months (99.82%).

Source files carried no `point_id`; cells are keyed on floating-point coordinate
pairs. A canonical `point_id` (0–5190) was assigned from the static grid sorted
by (latitude, longitude) and propagated by a six-decimal coordinate join.

### Target construction

The target is the **monthly NDVI anomaly** against a per-cell, per-calendar-month
climatology:

```
anomaly(cell, year, month) = NDVI(cell, year, month) − mean_over_years NDVI(cell, ·, month)
```

The climatology is computed from the **2001–2019 training period only**. A
full-record (2001–2023) baseline was tested first and rejected — see §3.5, which
documents why, since the failure mode is not obvious and is easy to reproduce
accidentally.

### Predictor construction

Each of the four climate drivers was converted to anomaly space against its own
train-period climatology, then expanded into four terms: the current month,
lags of one and two months, and a three-month accumulation. This captures the
antecedent-moisture response characteristic of dryland vegetation.

Critically, **lags were computed on the complete CHIRPS/ERA5-Land panel before
joining MODIS**. Taken after the join, the 0.18% of MODIS gaps would have caused
`shift(1)` to span a two-month interval for affected cells, silently corrupting
exactly the lag structure under investigation. The pipeline asserts that all
5,191 cells have all 276 months before any lag is taken.

ERA5-Land potential evaporation was winsorised at the 99.9th percentile
(674 mm/month; 1,433 rows clipped, 0.100%) before forming the ET/PET ratio.

**Final feature set — 22 predictors:**

| Group | Features |
|---|---|
| Rainfall | `precip_mm_an`, `_l1`, `_l2`, `_a3` |
| Temperature | `temp_c_an`, `_l1`, `_l2`, `_a3` |
| Soil moisture | `soil_moisture_an`, `_l1`, `_l2`, `_a3` |
| Evaporative stress | `et_pet_ratio_an`, `_l1`, `_l2`, `_a3` |
| Static / temporal | `landcover_class`, `cropland_fraction`, `month_sin`, `month_cos`, `lat`, `lon` |

After dropping 10,347 lag warm-up rows (January–February 2001), the matrix holds
**1,419,806 rows × 22 features** with zero missing values: 1,171,172 training
rows (2001–2019) and 248,634 holdout rows (2020–2023).

### A rejected predictor: EVI anomaly

An EVI anomaly term was initially specified. It was removed after it proved to be
target leakage rather than a driver: EVI and NDVI derive from the *same* MODIS
red/NIR/blue reflectances for the *same* cell in the *same* month, and the two
anomalies correlate at **r = 0.909** (r² = 0.827). Including it produced a holdout
R² of 0.818 that was almost entirely a restatement of NDVI as EVI, with the
climate drivers reduced to marginal contributors — a tautology, not an
explanation. All EVI terms were dropped.

---

## 3. Methodology

### 3.1 Spatial block cross-validation

Because neighbouring 5 km cells share weather, random k-fold splitting would
place near-duplicate samples on both sides of the split and inflate scores. The
basin was therefore tiled into **242 blocks of ~25 km** (0.2261° × 0.2298°,
median 25 cells per block) and blocks assigned to five folds.

A load-balancing greedy colouring was tried first and failed, leaving 24
adjacent block pairs sharing a fold after 19 dead ends. The block lattice under
8-connected adjacency is a *king graph*, which admits an exact periodic colouring:

```
fold(bi, bj) = (2·bi + bj) mod 5
```

For every neighbour offset `(di, dj) ∈ {−1,0,1}² \ {(0,0)}`, `(2·di + dj) mod 5 ≠ 0`,
so **no two adjacent blocks can share a fold** by construction. The implementation
asserts zero violations. Folds hold 18.7–20.8% of cells each and their centroids
agree to within 0.05°, confirming no fold is spatially biased.

### 3.2 Temporal holdout

**2020–2023 was withheld entirely** from training, scaler fitting, climatology
construction and hyperparameter selection. Every cell appears in both splits, so
the holdout tests temporal extrapolation, not spatial extrapolation. Reporting
both the cross-validation and the holdout turned out to be essential (§4.1).

### 3.3 Random Forest

`RandomForestRegressor`, 500 trees, `max_depth=10`, `max_features=1/3`,
`random_state=42`.

`max_features` was set to Breiman's regression default of *p*/3 after
benchmarking against scikit-learn's current default of 1.0. The latter evaluates
all 22 features at every split — bagging rather than a random forest — and was
both **2.8× slower and marginally less accurate** (fold-0 R² 0.2378 versus
0.2409). Depth 10 was chosen over the initially specified 20 because depth 20
with coordinate features memorised locations that did not transfer across years
(holdout R² −0.215 versus −0.033 at depth 10).

### 3.4 Multilayer Perceptron

Architecture 22 → 128 → 64 → 32 → 1, ReLU activations, batch normalisation and
dropout 0.2 at each hidden layer (12,097 parameters). Adam, learning rate 1e-3,
batch size 4,096, early stopping with patience 15 and a 100-epoch cap.
`StandardScaler` fitted on the training portion only.

Early stopping requires a validation signal, and using the outer cross-validation
fold for it would select the stopping epoch on the same data the fold score
reports. Each fold therefore holds out **15% of its training *blocks*** as an
inner validation set used only for early stopping; the outer fold is untouched
until scoring. Blocks rather than rows keep the inner split spatially separated
too.

The epoch cap bound in four of five folds (best epochs 87–97). Raising it would
likely have improved cross-validation while worsening the holdout, since the gap
between them is a temporal-overfitting gap (§4.1); the cap is reported rather
than tuned away.

### 3.5 Anomaly baseline — a reproducible pitfall

The target was first built against the conventional full-record (2001–2023)
climatology. The resulting holdout spatial correlation was **r = −0.461**: the
model placed stress exactly where vegetation had greened.

The cause is arithmetic, not statistical. When the holdout lies *inside* its own
baseline, a cell's mean anomaly over 2020–2023 and its mean anomaly over
2001–2019 are near-exact negatives:

| Diagnostic | Value |
|---|---|
| corr(train cell-mean anomaly, holdout cell-mean anomaly) | **−0.9984** |
| sd(holdout) / sd(train), observed | **4.774** |
| sd ratio predicted by algebra (19/4) | **4.750** |
| corr(model prediction, **train** cell-mean) | +0.4633 |
| corr(model prediction, **holdout** cell-mean) | −0.4609 |

The sd ratio matching 19/4 to three significant figures confirms the identity.
The model learned each cell's training offset correctly (+0.463); the target
definition then inverted it on the holdout (−0.461). Rebuilding the target on a
train-only climatology restored the correct sign (§4.1). No climate predictor
carried holdout information in either variant — the leak is in the target
transform alone, which is what makes it easy to miss.

### 3.6 SHAP attribution

- **TreeSHAP** on the Random Forest, `tree_path_dependent`, over a
  **cell-stratified 51,910-row** subsample (10 months per cell, all 5,191 cells
  covered, balanced across the four holdout years). Additivity error
  **9.0 × 10⁻¹⁶** — exact to machine precision. The full 248,634-row holdout was
  estimated at 16.7 hours at 242 ms/row; stratified sampling preserves complete
  per-cell spatial coverage at a fifth of the cost, and global importances are
  statistically indistinguishable at this sample size.
- **KernelSHAP** on the MLP, 2,000 points stratified across holdout years,
  k-means background of 100. Additivity error **3.2 × 10⁻⁸**.

KernelSHAP covers 1,649 of 5,191 cells (31.8%, median one sample per cell), so
**all spatial SHAP map layers use the Random Forest**; MLP SHAP is reported as
global importance only.

---

## 4. Results

### 4.1 Predictive accuracy

**Spatial block cross-validation, 2001–2019**

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Random Forest | 0.2261 ± 0.0057 | 0.03900 ± 0.00098 | 0.02518 ± 0.00062 |
| MLP | **0.3337 ± 0.0088** | **0.03619 ± 0.00099** | **0.02306 ± 0.00060** |

Pooled out-of-fold R² was 0.2259 (RF) and 0.3334 (MLP). In-sample scores exceeded
cross-validation by only 0.024 and 0.028 respectively, indicating no material
overfitting within the training period.

**Temporal holdout, 2020–2023**

| Metric | Random Forest | MLP | Climatology baseline |
|---|---|---|---|
| Monthly R² | **+0.0066** | −0.0283 | −0.0142 |
| Monthly RMSE | **0.05954** | 0.06058 | 0.06016 |
| Monthly MAE | **0.03899** | 0.03976 | 0.03951 |
| Monthly *r* | **+0.1709** | +0.1257 | — |
| **Spatial cell-mean *r*** | **+0.2874** | +0.1963 | — |
| Growing-season (Jun–Oct) spatial *r* | **+0.2416** | −0.0226 | — |
| Mean bias | −0.0089 | −0.0097 | — |

**The cross-validation and holdout rankings are opposite.** The MLP wins
cross-validation by 0.108 R² and loses every holdout metric. This is not a defect
but a distinction between two generalisation tasks: spatial block
cross-validation withholds *locations within years the model has seen*, so a
flexible learner can exploit the shared monthly weather field across blocks; the
holdout withholds *years*, where that shortcut is unavailable. Spatial
cross-validation alone would have selected the weaker extrapolator.

Absolute skill is low. The Random Forest improves on climatology by 1.03% in
RMSE and explains 8.3% of between-cell variance in the 2020–2023 mean anomaly
(*r* = 0.287). Both models are severely under-dispersed: predicted spatial
standard deviation is 5.9% (RF) and 7.7% (MLP) of the observed 0.0319. They
recover the *pattern* of stress but not its *amplitude*.

### 4.2 SHAP attribution

**Random Forest (TreeSHAP), top features by mean |SHAP|**

| Feature | mean \|SHAP\| | Share |
|---|---|---|
| 3-month rainfall anomaly | 0.002962 | **18.3%** |
| Rainfall anomaly, lag 1 | 0.001725 | 10.7% |
| 3-month soil moisture anomaly | 0.001538 | 9.5% |
| Evaporative stress anomaly | 0.001277 | 7.9% |
| 3-month evaporative stress | 0.000860 | 5.3% |
| … | | |
| Cropland fraction | 0.000043 | 0.27% |
| Land cover class | 0.000025 | 0.16% |

Rainfall terms account for 38.1% of total attribution; all moisture-related terms
together reach 64.5%. The beeswarm confirms physically correct directionality
throughout: low accumulated rainfall anomaly drives strongly negative SHAP, high
soil moisture positive, high temperature anomalies negative.

The response is **asymmetric**. The negative tail of `precip_mm_an_a3` reaches
−0.03 while its positive tail reaches only +0.01: rainfall deficits depress
vegetation roughly three times more than equivalent surpluses lift it,
consistent with a threshold-limited water response.

**MLP (KernelSHAP)** ranks very differently:

| Feature | Share |
|---|---|
| Time of year (`month_sin`) | **12.3%** |
| Time of year (`month_cos`) | 7.8% |
| Temperature anomaly | 7.5% |
| Soil moisture anomaly | 7.3% |
| 3-month rainfall anomaly | 7.2% |

**Calendar month takes 20.0% of the network's attribution against 6.1% of the
forest's.** Since the target is already deseasonalised, month carries no driver
information; the network is modulating output *amplitude* by season, because
anomaly variance is larger in the growing season. That is a variance effect, not
a mechanism, and it is the most plausible explanation for the network's weaker
temporal generalisation.

**Agreement between the two attributions is poor:** Spearman rank correlation
**+0.371**, with **zero features shared in the top three**.

```
RF  top 3: 3-month rainfall · rainfall lag 1 · 3-month soil moisture
MLP top 3: time of year (sin) · time of year (cos) · temperature anomaly
```

Spatially, 98.4% of cells have a rainfall term as their single strongest driver;
soil moisture leads in 223 cells and evaporative stress in 94.

### 4.3 Consensus and disagreement

| Metric | Value |
|---|---|
| Monthly corr(RF, MLP) | +0.6476 |
| Monthly same-sign rate | 76.1% |
| **Cell-level corr(RF, MLP)** | **+0.8370** |
| Mean \|RF − MLP\| | 0.001275 |

The models agree strongly on spatial pattern despite sharing no top driver —
they reach similar answers by different routes.

**Categories, absolute sign** — 4,034 cells "both agree stressed" (77.7%), 573
"both agree healthy" (11.0%), 584 "models disagree" (11.3%). This distribution is
an artefact: both models carry a dry bias (−0.0018, −0.0026) while the holdout
was net-greening (+0.0071), so a sign split labels most of the basin stressed.
Validated against observation, the "stressed" label is only **42.3% accurate
against a 37.7% base rate** (lift 1.12×), while "healthy" reaches **84.5% against
62.3%** (lift 1.36×). The asymmetry is blunt: trust green, distrust red.

**Categories, bias-corrected** — centring each model on its own basin mean
isolates the relative pattern, which is where the skill lies. This yields a
balanced 2,084 / 2,147 / 960 split (40.1% / 41.4% / 18.5%) with correct lift in
both directions (stressed 60.7% below basin mean against a 48.1% base rate, lift
1.26×). This is the map's default view.

**Disagreement does not indicate unreliability.**

| Disagreement quartile | Mean \|consensus error\| |
|---|---|
| Q1 (low) | 0.02476 |
| Q2 | 0.02452 |
| Q3 | 0.02500 |
| Q4 (high) | 0.02537 |

`corr(disagreement, |consensus error|) = +0.038`. Where the models disagree says
essentially nothing about where they are wrong.

**The consensus is worse than the Random Forest alone** (*r* 0.246 versus 0.287),
as expected when model quality is unequal.

### 4.4 Residual analysis

Residual = observed − predicted; positive means under-predicted greenness.

| Metric | Random Forest | MLP |
|---|---|---|
| Holdout RMSE | 0.05954 | 0.06058 |
| Threshold (1.5 × RMSE) | 0.08931 | 0.09087 |
| Monthly rows beyond threshold | 10.69% | 10.71% |
| Poorly predicted cells | 459 (8.8%) | 447 (8.6%) |

**The two models fail in the same places**: monthly residual correlation
**+0.9802**, cell-RMSE correlation +0.9953, and 88.5% of RF-poor rows are also
MLP-poor. **421 cells (8.1%) fail for both**; only 64 fail for one and not the
other. These are shared blind spots, not complementary errors — which
independently explains why ensembling did not help.

**By land cover**

| Class | Cells | RF cell RMSE | % failing (both) |
|---|---|---|---|
| Wetland | 2 | 0.1773 | 100% |
| Water | 33 | 0.1103 | 60.6% |
| Tree cover | 2 | 0.0996 | 100% |
| **Shrubland** | 664 | 0.0727 | **20.8%** |
| Grassland | 993 | 0.0551 | 11.4% |
| **Cropland** | 3,482 | 0.0500 | **4.2%** |
| Built-up | 10 | 0.0450 | 0% |
| Bare / sparse | 5 | 0.0399 | 0% |

**By season** — model RMSE tracks observed standard deviation almost exactly in
every calendar month (ratio 0.99–1.06; July 0.1048 against 0.1063, February
0.0261 against 0.0246).

**By year** — bias grows toward the end of the record: 2020 −0.0011, 2021 +0.0095,
2022 +0.0059, 2023 **+0.0214**. 2022 carries the worst RMSE (0.0663) and
near-zero spatial correlation. 65.7% of cells are under-predicted overall.

Spatially, residuals form a **northeast–southwest dipole**: systematic
under-prediction through 6–7.5 °E, 12.5–13.5 °N and over-prediction in the
southwest near 3.5–5 °E, 10.2–11.5 °N. The 421 both-fail cells form distinctly
**linear clusters** tracing drainage networks.

---

## 5. Discussion

### Which climate drivers dominate

Accumulated antecedent moisture, unambiguously. The three-month rainfall anomaly
is the single strongest predictor (18.3% of attribution, strongest driver in 98%
of cells), followed by three-month and one-month-lagged soil moisture. The
progression in marginal correlation is itself instructive: current-month rainfall
anomaly correlates with the NDVI anomaly at r = 0.089, one-month lag at 0.132 and
the three-month accumulation at 0.167. Vegetation integrates water supply over a
season rather than responding to the month in hand — which is why the originally
specified contemporaneous feature set had no predictive skill at all (holdout
R² −0.215), and why adding lag structure was necessary rather than optional.

Temperature enters negatively and with a lag, consistent with heat stress
compounding water limitation rather than acting independently. The asymmetric
rainfall response — deficits roughly three times as consequential as surpluses —
is the most policy-relevant single finding here: in this system a dry season
inflicts more damage than a comparably wet season repairs, so drought impacts
accumulate rather than average out.

### Where the models fail, and why

Three distinct failure modes emerged, and they are worth separating.

**Temporal extrapolation.** Within any calendar month, model RMSE approximately
equals the observed standard deviation. The models have essentially no
within-month skill; their slight positive holdout R² comes from capturing
seasonal *amplitude*, not spatial pattern within a month. Monthly 5 km NDVI
anomaly is dominated by compositing noise and by land management, grazing, fire
and planting decisions that no climate feature observes. The climate-forced
component is real, physically coherent and small.

**Progressive greening.** Bias grows monotonically in magnitude to +0.0214 by
2023. The basin greened over the holdout in a way the climate drivers do not
anticipate — plausibly reflecting land-use change, cropland expansion or CO₂
fertilisation, none of which are in the feature set. A model trained on
2001–2019 cannot see a trend that emerges afterwards, and no reweighting of these
features would fix it.

**Hydrologically controlled pixels.** The 421 both-fail cells cluster along
linear drainage features and the Water and Wetland classes top the failure table
(60.6% and 100% of cells failing). These are almost certainly *fadama*
floodplain pixels whose NDVI is driven by inundation and irrigation rather than
by rainfall anomalies — the predictors are simply the wrong ones for those cells.
The map de-emphasises water cells for this reason.

That both models fail in the *same* cells (residual r = 0.98) is the clearest
evidence that the limitation is in the information available, not in model
capacity. No ensemble of learners over this feature set will recover those cells.

### Land cover effects

Among classes with adequate sample size, shrubland is hardest (20.8% of cells
failing) and cropland easiest (4.2%) — a factor of five. A plausible mechanism is
that cropland is managed and responds fairly directly to seasonal rainfall, while
Sahelian shrubland mixes woody and herbaceous phenology with grazing and fire
effects invisible to the predictors.

This ranking should not be over-read, however. Normalising each class's error by
its own variability largely flattens it (shrubland 1.92, water 1.90, cropland
1.80, grassland 1.61): cropland looks predictable in part because its anomalies
are small, not only because the model understands it better. Land cover itself
contributes almost nothing marginally (0.16% of SHAP attribution) — it matters
through interactions with moisture, not as a direct effect.

### Methodological implications

Two results generalise beyond this basin.

First, **spatial cross-validation is not a proxy for temporal generalisation**.
The MLP led cross-validation by 0.108 R² and lost every holdout metric. Studies
reporting only spatial-block cross-validation may be selecting models that
interpolate well within observed years and extrapolate poorly beyond them.

Second, **disagreement-as-uncertainty failed here**. The common assumption that
ensemble spread flags unreliable predictions did not hold at all
(r = 0.038 against actual error). Disagreement is mapped in this study as a
finding, not as a confidence layer.

A third, more cautionary result is the anomaly-baseline inversion of §3.5. Using
a climatology that spans the holdout is standard practice for *describing*
anomalies and is actively harmful for *predicting* them, in a way that produces a
confidently wrong map rather than an obviously broken one. The signature — a
train/holdout cell-mean correlation near −1 and an sd ratio equal to the ratio of
record lengths — is easy to test for and worth checking in any anomaly-based
machine-learning study.

### Limitations

- Monthly skill is low in absolute terms; these maps support relative spatial
  comparison, not quantitative anomaly estimation.
- Predicted layers span ~6% of the observed range; the models are heavily
  under-dispersed and will understate extremes.
- ERA5-Land potential evaporation overestimates FAO-56 reference ET by roughly
  1.9× in semi-arid zones (median 3,898 mm/yr against an expected 1,800–2,400).
  Values are used as published and winsorised; the ET/PET ratio is therefore a
  valid *relative* predictor, not a calibrated aridity index.
- MLP SHAP covers 31.8% of cells, so no spatial SHAP layer uses the network.
- Four land cover classes (tree cover, built-up, bare/sparse, wetland; 19 cells
  combined) are too rare for reliable stratification and were pooled as "Other"
  in that analysis.
- Attribution is associational. SHAP explains the model, not the environment; it
  identifies which inputs the model used, not which physical processes operate.

### Further work

Irrigation and inundation predictors would directly target the dominant failure
mode. A land-surface trend or time term would absorb the progressive greening.
Predicting seasonal (June–October) rather than monthly aggregates would raise the
signal-to-noise ratio substantially, at the cost of temporal detail.

---

## 6. Conclusions

1. Drought-related vegetation stress in the Sokoto Basin is driven principally by
   **accumulated antecedent moisture**: the three-month rainfall anomaly is the
   strongest predictor in 98% of cells, with lagged and accumulated soil moisture
   next. Contemporaneous climate alone has no predictive skill.

2. The water response is **asymmetric** — deficits depress vegetation roughly
   three times more than equivalent surpluses lift it, so drought impacts
   accumulate rather than average out.

3. **Spatial cross-validation selected the wrong model.** The neural network led
   cross-validation (R² 0.334 versus 0.226) and lost every unseen-year metric.
   A temporal holdout was required to detect the reversal.

4. The two models **agree on where (cell-level r = 0.84) but not on why** (SHAP
   rank correlation 0.371, no shared top-three driver). Dual-model attribution
   exposed a divergence a single-model study would have reported as fact.

5. **Model disagreement did not indicate unreliability** (r = 0.038 against actual
   error), contradicting a common ensemble assumption.

6. Both models **fail in the same 421 cells**, concentrated on drainage networks
   and water bodies where vegetation is hydrologically rather than
   rainfall-controlled. The constraint is missing information, not model capacity.

7. Absolute skill is low: monthly R² ≈ 0.01, spatial *r* = 0.287, predicted range
   ~6% of observed. The models usefully indicate **where** stress is more or less
   likely and **which drivers** govern each location; they do not quantify
   anomaly magnitude.

---

## 7. References

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

Didan, K. (2015). *MOD13Q1 MODIS/Terra Vegetation Indices 16-Day L3 Global 250 m
SIN Grid V006*. NASA EOSDIS Land Processes DAAC.

Funk, C., Peterson, P., Landsfeld, M., Pedreros, D., Verdin, J., Shukla, S.,
Husak, G., Rowland, J., Harrison, L., Hoell, A., & Michaelsen, J. (2015). The
climate hazards infrared precipitation with stations — a new environmental record
for monitoring extremes. *Scientific Data*, 2, 150066.

Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., & Moore, R.
(2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone.
*Remote Sensing of Environment*, 202, 18–27.

Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B.,
Katz, R., Himmelfarb, J., Bansal, N., & Lee, S.-I. (2020). From local
explanations to global understanding with explainable AI for trees. *Nature
Machine Intelligence*, 2(1), 56–67.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model
predictions. *Advances in Neural Information Processing Systems*, 30, 4765–4774.

Muñoz-Sabater, J., Dutra, E., Agustí-Panareda, A., Albergel, C., Arduini, G.,
Balsamo, G., Boussetta, S., Choulga, M., Harrigan, S., Hersbach, H., Martens, B.,
Miralles, D. G., Piles, M., Rodríguez-Fernández, N. J., Zsoter, E., Buontempo, C.,
& Thépaut, J.-N. (2021). ERA5-Land: a state-of-the-art global reanalysis dataset
for land applications. *Earth System Science Data*, 13(9), 4349–4383.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O.,
Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A.,
Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn:
Machine learning in Python. *Journal of Machine Learning Research*, 12,
2825–2830.

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J.,
Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B.,
Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. (2017).
Cross-validation strategies for data with temporal, spatial, hierarchical, or
phylogenetic structure. *Ecography*, 40(8), 913–929.

Tucker, C. J. (1979). Red and photographic infrared linear combinations for
monitoring vegetation. *Remote Sensing of Environment*, 8(2), 127–150.

Valavi, R., Elith, J., Lahoz-Monfort, J. J., & Guillera-Arroita, G. (2019).
blockCV: An R package for generating spatially or environmentally separated folds
for k-fold cross-validation of species distribution models. *Methods in Ecology
and Evolution*, 10(2), 225–232.

Zanaga, D., Van De Kerchove, R., Daems, D., De Keersmaecker, W., Brockmann, C.,
Kirches, G., Wevers, J., Cartus, O., Santoro, M., Fritz, S., Lesiv, M.,
Herold, M., Tsendbazar, N.-E., Xu, P., Ramoino, F., & Arino, O. (2022).
*ESA WorldCover 10 m 2021 v200*. Zenodo.
