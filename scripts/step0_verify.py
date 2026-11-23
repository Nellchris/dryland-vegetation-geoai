"""Step 0: Data verification for Sokoto Basin GeoAI pipeline."""
import os, sys, glob
import numpy as np
import pandas as pd

D = r"C:\Users\USER\Documents\GeoAI_dryland"
YEARS = list(range(2001, 2024))
GROUPS = {"modis": ["ndvi", "evi"], "chirps": ["precip_mm"],
          "era5": ["temp_c", "soil_moisture", "et_mm", "pet_mm"]}

print("=" * 78)
print("STEP 0 - DATA VERIFICATION")
print("=" * 78)

# ---- file presence -------------------------------------------------------
print("\n[1] FILE PRESENCE")
missing = []
for g in GROUPS:
    for y in YEARS:
        p = os.path.join(D, f"{g}_{y}.csv")
        if not os.path.exists(p):
            missing.append(f"{g}_{y}.csv")
for extra in ["grid_static_features.csv", "study_area_boundary.geojson"]:
    if not os.path.exists(os.path.join(D, extra)):
        missing.append(extra)
print(f"  expected: 23 modis + 23 chirps + 23 era5 + static + boundary = 70 files")
print(f"  missing : {missing if missing else 'NONE'}")

# ---- static grid ---------------------------------------------------------
print("\n[2] STATIC GRID  (grid_static_features.csv)")
static = pd.read_csv(os.path.join(D, "grid_static_features.csv"))
print(f"  shape   : {static.shape}")
print(f"  columns : {list(static.columns)}")
print(f"  dtypes  : {dict(static.dtypes.astype(str))}")
print(f"  lon range: {static.longitude.min():.4f} .. {static.longitude.max():.4f}")
print(f"  lat range: {static.latitude.min():.4f} .. {static.latitude.max():.4f}")
lons = np.sort(static.longitude.unique()); lats = np.sort(static.latitude.unique())
print(f"  unique lon: {len(lons)}  unique lat: {len(lats)}")
if len(lons) > 1:
    print(f"  lon step (median): {np.median(np.diff(lons)):.6f} deg "
          f"(~{np.median(np.diff(lons))*111.32*np.cos(np.radians(static.latitude.mean())):.2f} km)")
if len(lats) > 1:
    print(f"  lat step (median): {np.median(np.diff(lats)):.6f} deg "
          f"(~{np.median(np.diff(lats))*110.57:.2f} km)")
print(f"  duplicate coord rows: {static.duplicated(['longitude','latitude']).sum()}")
print(f"  missing values:\n{static.isna().sum().to_string()}")
print("  landcover_class value counts (rounded):")
lc = static.landcover_class.round().astype('Int64')
print(lc.value_counts(dropna=False).sort_index().to_string())
print(f"  landcover_class non-integer deviation max: "
      f"{(static.landcover_class - static.landcover_class.round()).abs().max():.2e}")
print(f"  cropland_fraction: min={static.cropland_fraction.min():.4f} "
      f"max={static.cropland_fraction.max():.4f} mean={static.cropland_fraction.mean():.4f}")

# canonical point_id from static grid
static = static.sort_values(["latitude", "longitude"]).reset_index(drop=True)
KEY = ["lon_r", "lat_r"]
static["lon_r"] = static.longitude.round(6)
static["lat_r"] = static.latitude.round(6)
static["point_id"] = np.arange(len(static))
grid_keys = set(map(tuple, static[KEY].values))
print(f"\n  -> NOTE: no 'point_id' column exists in any file; files are keyed on")
print(f"     (longitude, latitude). Assigning point_id 0..{len(static)-1} from the")
print(f"     static grid, sorted by (latitude, longitude).")

# ---- yearly files --------------------------------------------------------
print("\n[3] YEARLY FILES")
rows = []
for g, vcols in GROUPS.items():
    for y in YEARS:
        p = os.path.join(D, f"{g}_{y}.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        df["lon_r"] = df.longitude.round(6); df["lat_r"] = df.latitude.round(6)
        k = set(map(tuple, df[KEY].values))
        rec = dict(group=g, year=y, rows=len(df), cols=len(df.columns),
                   n_pts=df.groupby(KEY).ngroup().nunique(),
                   months=df.month.nunique(),
                   yr_ok=bool((df.year == y).all()),
                   unmatched=len(k - grid_keys),
                   dup=int(df.duplicated(KEY + ["year", "month"]).sum()))
        for c in vcols:
            rec[f"na_{c}"] = int(df[c].isna().sum())
            rec[f"min_{c}"] = float(df[c].min())
            rec[f"max_{c}"] = float(df[c].max())
        rows.append(rec)
summ = pd.DataFrame(rows)

for g, vcols in GROUPS.items():
    s = summ[summ.group == g]
    print(f"\n  --- {g.upper()} ---  columns = "
          f"{['longitude','latitude','year','month'] + vcols}")
    base = ["year", "rows", "n_pts", "months", "yr_ok", "unmatched", "dup"]
    print(s[base + [f"na_{c}" for c in vcols]].to_string(index=False))
    print(f"  rows: min={s.rows.min()} max={s.rows.max()} total={s.rows.sum():,}")
    print(f"  expected full rows = {len(static)} pts x 12 mo = {len(static)*12}")
    for c in vcols:
        print(f"  {c}: global min={s[f'min_{c}'].min():.4f} "
              f"max={s[f'max_{c}'].max():.4f}")

# ---- cross-source point coverage ----------------------------------------
print("\n[4] POINT / COORDINATE CONSISTENCY")
print(f"  static grid points: {len(static)}")
print(f"  any yearly rows with coords NOT in static grid: "
      f"{int(summ.unmatched.sum())}")
print(f"  duplicate (point,year,month) rows anywhere: {int(summ.dup.sum())}")
cov = summ.groupby('group').n_pts.agg(['min','max'])
print(f"  distinct points per file by group:\n{cov.to_string()}")
print(f"\n  modis rows are SHORT of the full {len(static)*12} grid in every year")
print(f"  -> MODIS cloud/quality gaps; shortfall per year: "
      f"{(len(static)*12 - summ[summ.group=='modis'].rows).tolist()}")

# ---- boundary ------------------------------------------------------------
print("\n[5] STUDY AREA BOUNDARY")
try:
    import geopandas as gpd
    gdf = gpd.read_file(os.path.join(D, "study_area_boundary.geojson"))
    print(f"  features: {len(gdf)}  crs: {gdf.crs}")
    print(f"  columns : {list(gdf.columns)}")
    print(f"  bounds  : {gdf.total_bounds}")
    for c in gdf.columns:
        if c != gdf.geometry.name and gdf[c].nunique() <= 15:
            print(f"  {c}: {gdf[c].tolist()}")
except Exception as e:
    print(f"  geopandas read failed: {e}")

print("\n" + "=" * 78)
print("STEP 0 COMPLETE")
print("=" * 78)
