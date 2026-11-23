"""Step 1 (rebuild): climate-only feature matrix with anomaly-space lag features.

Design notes
------------
* Driver anomalies and their lags are computed on the COMPLETE CHIRPS/ERA5
  panel (5191 cells x 276 months, no gaps) BEFORE joining MODIS. If lags were
  taken after the MODIS inner join, a missing month would make shift(1) span a
  two-month jump and silently corrupt the lag structure.
* Driver climatologies use the TRAIN period only (2001-2019) so no holdout
  information leaks into any predictor.
* The target keeps the specified full-record (2001-2023) climatology. A
  train-only variant is also stored for sensitivity analysis.
* EVI is dropped entirely (same-sensor circularity with the target).
"""
import os
import numpy as np
import pandas as pd

D = r"C:\Users\USER\Documents\GeoAI_dryland"
YEARS = list(range(2001, 2024))
TRAIN_MAX = 2019
DRIVERS = ["precip_mm", "temp_c", "soil_moisture", "et_pet_ratio"]
pd.set_option("display.width", 220)


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


banner("STEP 1 (REBUILD) - CLIMATE-ONLY FEATURE MATRIX WITH LAGS")

# ---------------------------------------------------------------- static grid
print("\n[1] STATIC GRID")
static = pd.read_csv(os.path.join(D, "grid_static_features.csv"))
static["lon_r"] = static.longitude.round(6)
static["lat_r"] = static.latitude.round(6)
static = static.sort_values(["lat_r", "lon_r"], kind="mergesort").reset_index(drop=True)
static["point_id"] = np.arange(len(static), dtype=np.int32)
static["landcover_class"] = static.landcover_class.round().astype(np.int16)
WC = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
      50: "Built-up", 60: "Bare/sparse", 80: "Water", 90: "Wetland"}
RARE = {10, 50, 60, 90}
static["landcover_name"] = static.landcover_class.map(WC)
static["landcover_grouped"] = np.where(
    static.landcover_class.isin(RARE), "Other", static.landcover_name)
lut = static[["lon_r", "lat_r", "point_id"]]
print(f"  {len(static)} cells, point_id 0..{static.point_id.max()}")


def load(group, cols):
    parts = []
    for y in YEARS:
        df = pd.read_csv(os.path.join(D, f"{group}_{y}.csv"),
                         usecols=["longitude", "latitude", "year", "month"] + cols)
        df["lon_r"] = df.longitude.round(6)
        df["lat_r"] = df.latitude.round(6)
        df = df.merge(lut, on=["lon_r", "lat_r"], how="left")
        assert df.point_id.notna().all(), f"{group}_{y}: unmatched coords"
        parts.append(df[["point_id", "year", "month"] + cols])
    out = pd.concat(parts, ignore_index=True)
    out["point_id"] = out.point_id.astype(np.int32)
    out["year"] = out.year.astype(np.int16)
    out["month"] = out.month.astype(np.int8)
    print(f"  {group:<7}: {len(out):>9,} rows")
    return out


# -------------------------------------------------- complete climate panel
print("\n[2] BUILD COMPLETE CLIMATE PANEL (no MODIS yet)")
chirps = load("chirps", ["precip_mm"])
era5 = load("era5", ["temp_c", "soil_moisture", "et_mm", "pet_mm"])
clim_panel = chirps.merge(era5, on=["point_id", "year", "month"], how="inner")
n_exp = len(static) * 12 * len(YEARS)
print(f"  merged panel: {len(clim_panel):,} rows (expected {n_exp:,})")
assert len(clim_panel) == n_exp, "climate panel is NOT complete - lags unsafe"
assert not clim_panel.duplicated(["point_id", "year", "month"]).any()
cnt = clim_panel.groupby("point_id").size()
assert (cnt == 276).all(), "some cell lacks all 276 months"
print(f"  every one of {len(cnt):,} cells has all 276 months -> lags are safe")

# ---------------------------------------------------- PET clip + ET/PET ratio
print("\n[3] PET WINSORISATION + ET/PET RATIO")
p999 = clim_panel.pet_mm.quantile(0.999)
n_clip = int((clim_panel.pet_mm > p999).sum())
print(f"  99.9th pct = {p999:.2f} mm/month ({p999/30.44:.2f} mm/day); "
      f"raw max {clim_panel.pet_mm.max():.1f}")
print(f"  clipped {n_clip:,} rows ({100*n_clip/len(clim_panel):.3f}%)")
clim_panel["pet_mm_raw"] = clim_panel.pet_mm
clim_panel["pet_mm"] = clim_panel.pet_mm.clip(upper=p999)
clim_panel["et_pet_ratio"] = clim_panel.et_mm / clim_panel.pet_mm

# -------------------------------------- driver anomalies + lags (train clim)
print(f"\n[4] DRIVER ANOMALIES + LAGS  (climatology from 2001-{TRAIN_MAX} only)")
clim_panel = clim_panel.sort_values(["point_id", "year", "month"],
                                    kind="mergesort").reset_index(drop=True)
is_tr = clim_panel.year <= TRAIN_MAX
FEAT_CLIM = []
for v in DRIVERS:
    base = (clim_panel[is_tr].groupby(["point_id", "month"])[v].mean()
            .rename(f"{v}_cl").reset_index())
    clim_panel = clim_panel.merge(base, on=["point_id", "month"], how="left")
    clim_panel[f"{v}_an"] = clim_panel[v] - clim_panel[f"{v}_cl"]
    g = clim_panel.groupby("point_id")[f"{v}_an"]
    clim_panel[f"{v}_an_l1"] = g.shift(1)
    clim_panel[f"{v}_an_l2"] = g.shift(2)
    clim_panel[f"{v}_an_a3"] = (g.rolling(3, min_periods=3).sum()
                                .reset_index(level=0, drop=True))
    FEAT_CLIM += [f"{v}_an", f"{v}_an_l1", f"{v}_an_l2", f"{v}_an_a3"]
    print(f"  {v:<14} anom sd={clim_panel[f'{v}_an'].std():.4f}  "
          f"acc3 sd={clim_panel[f'{v}_an_a3'].std():.4f}  "
          f"(train-period anom mean {clim_panel.loc[is_tr, f'{v}_an'].mean():+.2e})")
print(f"  -> {len(FEAT_CLIM)} climate features")
warm = int(clim_panel[FEAT_CLIM].isna().any(axis=1).sum())
print(f"  warm-up rows with NaN lags: {warm:,} "
      f"({warm/len(static):.0f} months x {len(static):,} cells, all in Jan-Feb 2001)")
assert clim_panel.loc[clim_panel.year >= 2020, FEAT_CLIM].notna().all().all(), \
    "holdout must have no NaN lag features"

# ------------------------------------------------------- MODIS target + join
print("\n[5] MODIS TARGET + JOIN")
modis = load("modis", ["ndvi"])
g = modis.groupby(["point_id", "month"]).ndvi
modis["ndvi_clim"] = g.transform("mean")                      # spec: 2001-2023
modis["ndvi_anomaly"] = modis.ndvi - modis.ndvi_clim
tr_cl = (modis[modis.year <= TRAIN_MAX].groupby(["point_id", "month"]).ndvi.mean()
         .rename("ndvi_clim_tr").reset_index())
modis = modis.merge(tr_cl, on=["point_id", "month"], how="left")
modis["ndvi_anomaly_trclim"] = modis.ndvi - modis.ndvi_clim_tr
print(f"  ndvi_anomaly         (2001-2023 clim, PRIMARY TARGET): "
      f"sd={modis.ndvi_anomaly.std():.4f}")
print(f"  ndvi_anomaly_trclim  (2001-2019 clim, sensitivity)   : "
      f"sd={modis.ndvi_anomaly_trclim.std():.4f}")

df = modis.merge(clim_panel.drop(columns=[f"{v}_cl" for v in DRIVERS]),
                 on=["point_id", "year", "month"], how="inner")
print(f"  joined: {len(df):,} rows")
before = len(df)
df = df.dropna(subset=FEAT_CLIM).reset_index(drop=True)
print(f"  dropped {before-len(df):,} lag warm-up rows -> {len(df):,}")

# ------------------------------------------------------------- static + time
print("\n[6] STATIC FEATURES + CYCLICAL MONTH")
df = df.merge(static[["point_id", "longitude", "latitude", "landcover_class",
                      "landcover_name", "landcover_grouped", "cropland_fraction"]],
              on="point_id", how="left").rename(
    columns={"longitude": "lon", "latitude": "lat"})
df["month_sin"] = np.sin(2 * np.pi * df.month / 12.0)
df["month_cos"] = np.cos(2 * np.pi * df.month / 12.0)
FEAT_STATIC = ["landcover_class", "cropland_fraction", "month_sin", "month_cos",
               "lat", "lon"]
FEATURES = FEAT_CLIM + FEAT_STATIC
TARGET = "ndvi_anomaly"
df["split"] = np.where(df.year >= 2020, "holdout", "train")

OBS = ["ndvi", "ndvi_clim", "ndvi_clim_tr", "ndvi_anomaly_trclim",
       "precip_mm", "temp_c", "soil_moisture", "et_mm", "pet_mm", "pet_mm_raw",
       "et_pet_ratio", "landcover_name", "landcover_grouped"]
COLS, _seen = [], set()
for c in (["point_id", "lon", "lat", "year", "month", "split", TARGET]
          + FEATURES + OBS):
    if c not in _seen:
        COLS.append(c)
        _seen.add(c)
df = df[COLS].sort_values(["point_id", "year", "month"],
                          kind="mergesort").reset_index(drop=True)
assert df.columns.is_unique, "duplicate columns"

# ------------------------------------------------------------------ report
banner("FEATURE MATRIX SUMMARY")
print(f"\nSHAPE: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"\nTARGET  : {TARGET}")
print(f"FEATURES: {len(FEATURES)} = {len(FEAT_CLIM)} climate + "
      f"{len(FEAT_STATIC)} static/temporal")
print("\n  climate (anomaly space, train-period baseline):")
for i in range(0, len(FEAT_CLIM), 4):
    print("    " + "  ".join(f"{c:<22}" for c in FEAT_CLIM[i:i + 4]))
print(f"\n  static/temporal: {FEAT_STATIC}")
print("\n  EVI: dropped entirely (same-sensor circularity with target).")

mv = df.isna().sum()
print(f"\nMISSING VALUES: total = {int(mv.sum())}")
print(mv[mv > 0].to_string() if mv.sum() else "  (none in any column)")

print("\nSUMMARY STATS (target + features):")
print(df[[TARGET] + FEATURES].describe(percentiles=[.01, .5, .99])
      .T.round(4).to_string())

print(f"\nTRAIN 2001-2019  : {int((df.split=='train').sum()):>9,} rows  "
      f"mean target {df.loc[df.split=='train', TARGET].mean():+.5f}")
print(f"HOLDOUT 2020-2023: {int((df.split=='holdout').sum()):>9,} rows  "
      f"mean target {df.loc[df.split=='holdout', TARGET].mean():+.5f}")
print(f"CELLS            : {df.point_id.nunique():,}   YEARS: "
      f"{df.year.min()}-{df.year.max()}")

print(f"\nCORRELATION of each feature with {TARGET}:")
cr = df[FEATURES + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET)
print(cr.sort_values(key=abs, ascending=False).round(4).to_string())

out = os.path.join(D, "feature_matrix.csv")
df.to_parquet(os.path.join(D, "feature_matrix.parquet"), index=False)
df.to_csv(out, index=False, float_format="%.6g")
static.drop(columns=["lon_r", "lat_r"]).to_csv(
    os.path.join(D, "grid_points.csv"), index=False)
import json
with open(os.path.join(D, "feature_config.json"), "w") as fh:
    json.dump(dict(target=TARGET, features=FEATURES, climate_features=FEAT_CLIM,
                   static_features=FEAT_STATIC, drivers=DRIVERS,
                   driver_clim_years=[2001, TRAIN_MAX],
                   target_clim_years=[2001, 2023], rf_max_depth=10,
                   pet_clip_mm=float(p999), n_rows=int(len(df)),
                   n_cells=int(df.point_id.nunique())), fh, indent=2)
print("\nSAVED:")
print(f"  feature_matrix.csv     ({os.path.getsize(out)/1e6:.1f} MB)")
print("  feature_matrix.parquet")
print("  feature_config.json    (target + 22-feature list)")
print("  grid_points.csv")
banner("STEP 1 REBUILD COMPLETE")
