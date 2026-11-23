"""Step 8a: build the compact static data bundle for the web map.

Writes webmap/data.js defining window.BASIN = {...}. A plain .js file (not
fetched JSON) so the map also works when opened from the filesystem, with no
server and no CORS issues, while staying a static asset for GitHub Pages.

Numeric fields are stored as integers scaled by a power of ten and decoded in
the browser, which roughly halves the payload versus decimal strings.
"""
import json
import os

import numpy as np
import pandas as pd

D = r"C:\Users\USER\Documents\GeoAI_dryland"
W = os.path.join(D, "webmap")
os.makedirs(W, exist_ok=True)

cfg = json.load(open(os.path.join(D, "cv_config.json")))
FEATURES = cfg["features"]

# ----------------------------------------------------- readable feature names
NICE = {
    "precip_mm_an": "Rainfall anomaly, this month",
    "precip_mm_an_l1": "Rainfall anomaly, 1 month earlier",
    "precip_mm_an_l2": "Rainfall anomaly, 2 months earlier",
    "precip_mm_an_a3": "3-month rainfall deficit / surplus",
    "temp_c_an": "Temperature anomaly, this month",
    "temp_c_an_l1": "Temperature anomaly, 1 month earlier",
    "temp_c_an_l2": "Temperature anomaly, 2 months earlier",
    "temp_c_an_a3": "3-month temperature anomaly",
    "soil_moisture_an": "Soil moisture anomaly, this month",
    "soil_moisture_an_l1": "Soil moisture anomaly, 1 month earlier",
    "soil_moisture_an_l2": "Soil moisture anomaly, 2 months earlier",
    "soil_moisture_an_a3": "3-month soil moisture anomaly",
    "et_pet_ratio_an": "Evaporative stress, this month",
    "et_pet_ratio_an_l1": "Evaporative stress, 1 month earlier",
    "et_pet_ratio_an_l2": "Evaporative stress, 2 months earlier",
    "et_pet_ratio_an_a3": "3-month evaporative stress",
    "month_sin": "Time of year",
    "month_cos": "Time of year",
    "lat": "Location (north-south)",
    "lon": "Location (east-west)",
    "landcover_class": "Land cover type",
    "cropland_fraction": "Share of cell cropped",
}
# six driver families - a 22-class legend is unreadable, so the map colours by
# family while the popup still names the exact top feature
FAMILY = {"precip": "Rainfall", "soil_moisture": "Soil moisture",
          "temp": "Temperature", "et_pet": "Evaporative stress",
          "month": "Time of year", "place": "Place & land cover"}
FAM_ORDER = ["Rainfall", "Soil moisture", "Temperature", "Evaporative stress",
             "Time of year", "Place & land cover"]


def fam_of(f):
    if f.startswith("precip"):
        return "Rainfall"
    if f.startswith("soil_moisture"):
        return "Soil moisture"
    if f.startswith("temp"):
        return "Temperature"
    if f.startswith("et_pet"):
        return "Evaporative stress"
    if f.startswith("month"):
        return "Time of year"
    return "Place & land cover"


print("=" * 70)
print("STEP 8a - BUILD WEB MAP DATA BUNDLE")
print("=" * 70)

cons = pd.read_csv(os.path.join(D, "consensus_map.csv"))
res = pd.read_csv(os.path.join(D, "residuals.csv"))
print(f"\nconsensus_map {cons.shape} | residuals {res.shape}")
g = cons.merge(res[["point_id", "rf_rmse", "mlp_rmse", "both_poor",
                    "rf_resid_mean"]],
               on="point_id", how="left", validate="one_to_one")
assert len(g) == 5191 and g.rf_rmse.notna().all()

# ------------------------------------------------------- RF SHAP per cell
print("\naggregating RF SHAP per cell ...")
sh = pd.read_parquet(os.path.join(D, "shap_rf.parquet"))
scols = [f"shap_{f}" for f in FEATURES]
absm = (sh.assign(**{c: sh[c].abs() for c in scols})
        .groupby("point_id")[scols].mean())
absm.columns = FEATURES
print(f"  cells with SHAP: {len(absm):,}/5191")
assert len(absm) == 5191

A = absm.values
top_feat = A.argmax(1)
shap_mag = A.sum(axis=1)                        # total attribution magnitude
# top three features per cell, for the popup
o3 = np.argsort(-A, axis=1)[:, :3]
v3 = np.take_along_axis(A, o3, axis=1)
fam_cols = {}
for fam in FAM_ORDER:
    cs = [f for f in FEATURES if fam_of(f) == fam]
    fam_cols[fam] = absm[cs].sum(axis=1).values
fam_mat = np.column_stack([fam_cols[f] for f in FAM_ORDER])
top_fam = fam_mat.argmax(1)
absm = absm.reset_index()
order = {p: i for i, p in enumerate(absm.point_id)}
idx = g.point_id.map(order).to_numpy()
g["shap_top_feat"] = top_feat[idx]
g["shap_top_fam"] = top_fam[idx]
g["shap_mag"] = shap_mag[idx]
o3, v3 = o3[idx], v3[idx]
print("  top driver family across cells:")
for i, f in enumerate(FAM_ORDER):
    n = int((g.shap_top_fam == i).sum())
    print(f"    {f:<22} {n:>5,} cells ({100*n/len(g):5.1f}%)")
print("  top single feature (most common):")
tf = pd.Series(FEATURES).iloc[g.shap_top_feat].value_counts().head(5)
for k, v in tf.items():
    print(f"    {k:<22} {v:>5,}")

# --------------------------------------------------------------- encoding
CATS = ["Both agree stressed", "Both agree healthy", "Models disagree"]
cmap = {c: i for i, c in enumerate(CATS)}
g["cat_i"] = g.category.map(cmap).astype(int)
g["catbc_i"] = g.category_biascorr.map(cmap).astype(int)

LC = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
      50: "Built-up", 60: "Bare / sparse", 80: "Water", 90: "Wetland"}
LC_ORDER = [40, 30, 20, 10, 60, 50, 80, 90]
lci = {c: i for i, c in enumerate(LC_ORDER)}
g["lc_i"] = g.landcover_class.map(lci).astype(int)

g = g.sort_values("point_id").reset_index(drop=True)


def enc(s, scale):
    return [int(round(v * scale)) for v in s.to_numpy()]


data = dict(
    n=int(len(g)),
    cell=0.044916,                       # grid spacing in degrees
    lon=enc(g.lon, 1e4), lat=enc(g.lat, 1e4),
    obs=enc(g.observed, 1e5),
    rf=enc(g.rf_pred, 1e5),
    mlp=enc(g.mlp_pred, 1e5),
    cons=enc(g.consensus, 1e5),
    dis=enc(g.disagreement, 1e6),
    rfrel=enc(g.rf_rel, 1e5),
    mlprel=enc(g.mlp_rel, 1e5),
    consrel=enc(g.consensus_rel, 1e5),
    cat=g.cat_i.tolist(), catbc=g.catbc_i.tolist(),
    lc=g.lc_i.tolist(),
    tfam=g.shap_top_fam.tolist(), tfeat=g.shap_top_feat.tolist(),
    smag=enc(g.shap_mag, 1e6),
    t3=[[int(x) for x in r] for r in o3],
    t3v=[[int(round(x * 1e6)) for x in r] for r in v3],
    rmse=enc(g.rf_rmse, 1e5),
    poor=g.both_poor.astype(int).tolist(),
    resid=enc(g.rf_resid_mean, 1e5),
    pid=g.point_id.tolist(),
)
SCALE = dict(lon=1e4, lat=1e4, obs=1e5, rf=1e5, mlp=1e5, cons=1e5, dis=1e6,
             rfrel=1e5, mlprel=1e5, consrel=1e5, smag=1e6, rmse=1e5, resid=1e5)

# ------------------------------------------------------ dissolved boundary
print("\ndissolving study-area boundary ...")
import geopandas as gpd
b = gpd.read_file(os.path.join(D, "study_area_boundary.geojson"))
dis = b.dissolve().geometry.iloc[0]
print(f"  {len(b)} LGA polygons -> 1 outline")
best = None
for tol in (0.002, 0.004, 0.008):
    s = dis.simplify(tol).buffer(0)
    gj = gpd.GeoSeries([s], crs=b.crs).to_json()
    print(f"  tol={tol}: {len(gj)/1e3:.0f} KB")
    if best is None and len(gj) < 90_000:
        best = (tol, json.loads(gj))
if best is None:
    best = (0.008, json.loads(gpd.GeoSeries([dis.simplify(0.008).buffer(0)],
                                            crs=b.crs).to_json()))
print(f"  using tol={best[0]}")
boundary = best[1]
# round boundary coords to 4 dp


def rnd(o):
    if isinstance(o, list):
        return [rnd(x) for x in o]
    if isinstance(o, float):
        return round(o, 4)
    return o


for ft in boundary["features"]:
    ft["geometry"]["coordinates"] = rnd(ft["geometry"]["coordinates"])
    ft["properties"] = {}

# ------------------------------------------------------------------ write
meta = dict(
    features=FEATURES, nice=[NICE[f] for f in FEATURES],
    fam=FAM_ORDER, feat_fam=[FAM_ORDER.index(fam_of(f)) for f in FEATURES],
    cats=CATS, lc_codes=LC_ORDER, lc_names=[LC[c] for c in LC_ORDER],
    scale=SCALE,
    stats=dict(
        obs_sd=float(g.observed.std()), rf_sd=float(g.rf_pred.std()),
        mlp_sd=float(g.mlp_pred.std()),
        cat_counts=[int((g.cat_i == i).sum()) for i in range(3)],
        catbc_counts=[int((g.catbc_i == i).sum()) for i in range(3)],
        n_poor=int(g.both_poor.sum()), n_water=int((g.landcover_class == 80).sum()),
    ))
out = os.path.join(W, "data.js")
with open(out, "w", encoding="utf-8") as fh:
    fh.write("// Sokoto Basin GeoAI - static data bundle (generated)\n")
    fh.write("window.BASIN=")
    json.dump(data, fh, separators=(",", ":"))
    fh.write(";\nwindow.BASIN_META=")
    json.dump(meta, fh, separators=(",", ":"))
    fh.write(";\nwindow.BASIN_BOUNDARY=")
    json.dump(boundary, fh, separators=(",", ":"))
    fh.write(";\n")
print(f"\nwrote webmap/data.js ({os.path.getsize(out)/1e6:.2f} MB)")
print(f"  cells {data['n']:,} | boundary {len(json.dumps(boundary))/1e3:.0f} KB")
print("\nSTEP 8a COMPLETE")
