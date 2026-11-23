"""Step 1: Load all sources, compute anomalies, build feature matrix."""
import os
import numpy as np
import pandas as pd

D = r"C:\Users\USER\Documents\GeoAI_dryland"
YEARS = list(range(2001, 2024))
pd.set_option("display.width", 200)


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


banner("STEP 1 - DATA LOADING + ANOMALY COMPUTATION")

# ---------------------------------------------------------------- static grid
print("\n[1] STATIC GRID / point_id assignment")
static = pd.read_csv(os.path.join(D, "grid_static_features.csv"))
static["lon_r"] = static.longitude.round(6)
static["lat_r"] = static.latitude.round(6)
static = static.sort_values(["lat_r", "lon_r"], kind="mergesort").reset_index(drop=True)
static["point_id"] = np.arange(len(static), dtype=np.int32)
static["landcover_class"] = static.landcover_class.round().astype(np.int16)
print(f"  grid cells          : {len(static)}")
print(f"  point_id range      : {static.point_id.min()}..{static.point_id.max()}")
print(f"  landcover classes   : {sorted(static.landcover_class.unique().tolist())}")

WC_NAME = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
           50: "Built-up", 60: "Bare/sparse", 80: "Water", 90: "Wetland"}
RARE = {10, 50, 60, 90}
static["landcover_name"] = static.landcover_class.map(WC_NAME)
static["landcover_grouped"] = np.where(
    static.landcover_class.isin(RARE), "Other", static.landcover_name)
print(f"  pooled Other cells  : {int((static.landcover_grouped == 'Other').sum())}")

lut = static[["lon_r", "lat_r", "point_id"]]


def load(group, cols):
    """Concatenate yearly CSVs and attach point_id via 6-dp coordinate join."""
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
    print(f"  {group:<7}: {len(out):>9,} rows  cols={cols}")
    return out


print("\n[2] LOADING YEARLY FILES")
modis = load("modis", ["ndvi", "evi"])
chirps = load("chirps", ["precip_mm"])
era5 = load("era5", ["temp_c", "soil_moisture", "et_mm", "pet_mm"])

# ---------------------------------------------------------------------- merge
print("\n[3] MERGE  (inner on point_id + year + month)")
KEYS = ["point_id", "year", "month"]
df = modis.merge(chirps, on=KEYS, how="inner").merge(era5, on=KEYS, how="inner")
print(f"  modis rows            : {len(modis):>9,}")
print(f"  after inner join      : {len(df):>9,}")
print(f"  dropped (MODIS gaps)  : {len(modis) - len(df):>9,}")
full = len(static) * 12 * len(YEARS)
print(f"  full grid would be    : {full:>9,}  -> retained {100 * len(df) / full:.2f}%")
assert not df.duplicated(KEYS).any(), "duplicate keys after merge"

# ------------------------------------------------------------------ PET clip
print("\n[4] PET WINSORISATION + ET/PET RATIO")
p999 = df.pet_mm.quantile(0.999)
n_clip = int((df.pet_mm > p999).sum())
print(f"  pet_mm 99.9th pct     : {p999:.2f} mm/month ({p999 / 30.44:.2f} mm/day)")
print(f"  raw pet_mm max        : {df.pet_mm.max():.2f} mm/month "
      f"({df.pet_mm.max() / 30.44:.2f} mm/day)")
print(f"  rows clipped          : {n_clip:,} ({100 * n_clip / len(df):.3f}%)")
df["pet_mm_raw"] = df.pet_mm
df["pet_mm"] = df.pet_mm.clip(upper=p999)
df["et_pet_ratio"] = df.et_mm / df.pet_mm
print(f"  et_pet_ratio          : min={df.et_pet_ratio.min():.4f} "
      f"med={df.et_pet_ratio.median():.4f} max={df.et_pet_ratio.max():.4f}")
print("  NOTE: ERA5-Land PET overestimates FAO-56 ET0 by ~1.9x in semi-arid")
print("  zones; the ratio is a valid relative predictor, not a calibrated")
print("  aridity index. Caveat carried into the technical report.")

# --------------------------------------------------------------- climatology
print("\n[5] ANOMALIES  (per-pixel per-calendar-month climatology, 2001-2023)")
for v in ("ndvi", "evi"):
    g = df.groupby(["point_id", "month"], observed=True)[v]
    df[f"{v}_clim"] = g.transform("mean")
    df[f"{v}_anomaly"] = df[v] - df[f"{v}_clim"]
    cnt = g.transform("size")
    print(f"  {v}: climatology n per (cell,month) min={cnt.min()} max={cnt.max()}")
    print(f"     {v}_anomaly: mean={df[f'{v}_anomaly'].mean():+.3e} "
          f"sd={df[f'{v}_anomaly'].std():.4f} "
          f"min={df[f'{v}_anomaly'].min():.4f} max={df[f'{v}_anomaly'].max():.4f}")
print("  NOTE: climatology spans the full 2001-2023 record as specified, so the")
print("  2020-2023 holdout target is defined against a baseline including those")
print("  years. Standard anomaly definition (a target transform, not a")
print("  predictor); no climate driver carries holdout information.")

# -------------------------------------------------------------- static join
print("\n[6] JOIN STATIC FEATURES + CYCLICAL MONTH")
df = df.merge(static[["point_id", "longitude", "latitude", "landcover_class",
                      "landcover_name", "landcover_grouped", "cropland_fraction"]],
              on="point_id", how="left")
df = df.rename(columns={"longitude": "lon", "latitude": "lat"})
df["month_sin"] = np.sin(2 * np.pi * df.month / 12.0)
df["month_cos"] = np.cos(2 * np.pi * df.month / 12.0)
assert df[["lon", "lat", "landcover_class"]].notna().all().all()
print(f"  joined static cols, no nulls. lon {df.lon.min():.4f}..{df.lon.max():.4f}"
      f"  lat {df.lat.min():.4f}..{df.lat.max():.4f}")

ORDER = ["point_id", "lon", "lat", "year", "month", "month_sin", "month_cos",
         "ndvi", "ndvi_clim", "ndvi_anomaly", "evi", "evi_clim", "evi_anomaly",
         "precip_mm", "temp_c", "soil_moisture", "et_mm", "pet_mm", "pet_mm_raw",
         "et_pet_ratio", "landcover_class", "landcover_name", "landcover_grouped",
         "cropland_fraction"]
df = df[ORDER].sort_values(["point_id", "year", "month"],
                           kind="mergesort").reset_index(drop=True)

# -------------------------------------------------------------------- report
banner("FEATURE MATRIX SUMMARY")
print(f"\nSHAPE: {df.shape[0]:,} rows x {df.shape[1]} columns")
print("\nCOLUMNS / DTYPES:")
for c in df.columns:
    print(f"  {c:<20} {str(df[c].dtype):<10}")
mv = df.isna().sum()
print(f"\nMISSING VALUES: total = {int(mv.sum())}")
print(mv[mv > 0].to_string() if mv.sum() else "  (none in any column)")

num = df.select_dtypes(include=[np.number]).columns
print("\nSUMMARY STATS:")
print(df[num].describe(percentiles=[.01, .25, .5, .75, .99]).T.round(4).to_string())

print(f"\nTEMPORAL COVERAGE: {df.year.min()}-{df.year.max()}, "
      f"{df.year.nunique()} years, {df.groupby('year').size().min():,}-"
      f"{df.groupby('year').size().max():,} rows/yr")
print(f"SPATIAL COVERAGE : {df.point_id.nunique():,} grid cells")
print(f"\nTRAIN 2001-2019  : {int((df.year <= 2019).sum()):>9,} rows")
print(f"HOLDOUT 2020-2023: {int((df.year >= 2020).sum()):>9,} rows")

print("\nLAND COVER DISTRIBUTION:")
cells = static.groupby("landcover_class").size()
lcs = df.groupby(["landcover_class", "landcover_name", "landcover_grouped"],
                 observed=True).size().rename("rows").reset_index()
lcs["cells"] = lcs.landcover_class.map(cells)
print(lcs.to_string(index=False))

print("\nTARGET ndvi_anomaly percentiles:")
print(df.ndvi_anomaly.describe(
    percentiles=[.001, .01, .05, .5, .95, .99, .999]).round(4).to_string())

print("\nCORRELATION of drivers with ndvi_anomaly:")
drv = ["precip_mm", "temp_c", "soil_moisture", "et_pet_ratio", "evi_anomaly",
       "cropland_fraction", "lat", "lon", "month_sin", "month_cos"]
corr = df[drv + ["ndvi_anomaly"]].corr(numeric_only=True)["ndvi_anomaly"]
print(corr.drop("ndvi_anomaly").sort_values(key=abs, ascending=False)
      .round(4).to_string())

# --------------------------------------------------------------------- save
out_csv = os.path.join(D, "feature_matrix.csv")
out_pq = os.path.join(D, "feature_matrix.parquet")
df.to_parquet(out_pq, index=False)
df.to_csv(out_csv, index=False, float_format="%.6g")
print("\nSAVED:")
print(f"  feature_matrix.csv     ({os.path.getsize(out_csv) / 1e6:.1f} MB)")
print(f"  feature_matrix.parquet ({os.path.getsize(out_pq) / 1e6:.1f} MB, "
      f"used by later steps)")
static.drop(columns=["lon_r", "lat_r"]).to_csv(
    os.path.join(D, "grid_points.csv"), index=False)
print(f"  grid_points.csv        (point_id lookup, {len(static)} cells)")
banner("STEP 1 COMPLETE")
