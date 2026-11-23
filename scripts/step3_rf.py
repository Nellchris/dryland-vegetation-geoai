"""Step 3: Random Forest with spatial block 5-fold CV + 2020-2023 holdout."""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

D = r"C:\Users\USER\Documents\GeoAI_dryland"
pd.set_option("display.width", 220)

N_TREES = 500
MAX_DEPTH = 10
SEED = 42
# Breiman's regression default m = p/3. Benchmarked against sklearn's
# max_features=1.0: 2.8x faster AND marginally more accurate (fold-0 R2
# 0.2409 vs 0.2378), since decorrelating the trees is what makes a random
# forest a forest rather than plain bagging.
MAX_FEATURES = 1 / 3


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78, flush=True)


def metrics(y, p):
    return dict(r2=r2_score(y, p),
                rmse=float(mean_squared_error(y, p) ** 0.5),
                mae=float(mean_absolute_error(y, p)))


banner("STEP 3 - RANDOM FOREST")

cfg = json.load(open(os.path.join(D, "cv_config.json")))
TARGET, FEATURES, N_FOLDS = cfg["target"], cfg["features"], cfg["n_folds"]
df = pd.read_parquet(os.path.join(D, "model_matrix.parquet"))
tr = df[df.split == "train"].reset_index(drop=True)
ho = df[df.split == "holdout"].reset_index(drop=True)
print(f"\nTARGET: {TARGET}   (sd train={tr[TARGET].std():.5f}, "
      f"holdout={ho[TARGET].std():.5f})")
print(f"train {len(tr):,} rows | holdout {len(ho):,} rows | "
      f"{len(FEATURES)} features")
print(f"RF: n_estimators={N_TREES}, max_depth={MAX_DEPTH}, "
      f"max_features=1/3 (Breiman regression default), random_state={SEED}")

Xtr, ytr = tr[FEATURES].to_numpy(np.float32), tr[TARGET].to_numpy(np.float64)
Xho, yho = ho[FEATURES].to_numpy(np.float32), ho[TARGET].to_numpy(np.float64)
folds = tr.fold.to_numpy()


def make():
    return RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                 max_features=MAX_FEATURES,
                                 random_state=SEED, n_jobs=-1)


# ----------------------------------------------------- spatial block 5-fold CV
banner(f"[1] SPATIAL BLOCK {N_FOLDS}-FOLD CV  (2001-2019)")
# resume from checkpoints if a previous run was interrupted
PART_CSV = os.path.join(D, "rf_cv_metrics_partial.csv")
PART_NPY = os.path.join(D, "rf_oof_partial.npy")
rows = []
oof = np.full(len(tr), np.nan)
done = set()
if os.path.exists(PART_CSV) and os.path.exists(PART_NPY):
    prev = pd.read_csv(PART_CSV)
    cached = np.load(PART_NPY)
    # a checkpoint from a different target must never be reused
    stale = "target" not in prev.columns or set(prev.target) != {TARGET}
    if stale:
        print(f"  checkpoint is for a different target "
              f"({prev.get('target', pd.Series(['<unlabelled>'])).iloc[0]!r}, "
              f"need {TARGET!r}) - ignoring it and starting fresh")
        prev, cached = None, None
    elif len(cached) == len(tr):
        rows = prev.to_dict("records")
        oof = cached
        done = set(prev.fold.astype(int))
        print(f"  RESUMING: folds {sorted(done)} already done "
              f"({int(np.isfinite(oof).sum()):,} oof predictions restored)")
        for r in rows:
            print(f"    fold {int(r['fold'])}: R2={r['r2']:+.4f}  "
                  f"RMSE={r['rmse']:.5f}  MAE={r['mae']:.5f}  (cached)")
    elif cached is not None:
        print("  checkpoint row count mismatch - starting fresh")

for f in range(N_FOLDS):
    if f in done:
        continue
    va = folds == f
    t0 = time.time()
    m = make().fit(Xtr[~va], ytr[~va])
    p = m.predict(Xtr[va])
    oof[va] = p
    mt = metrics(ytr[va], p)
    mt.update(fold=f, n_train=int((~va).sum()), n_val=int(va.sum()),
              val_cells=int(tr.loc[va, "point_id"].nunique()),
              secs=round(time.time() - t0, 1), target=TARGET)
    rows.append(mt)
    print(f"  fold {f}: R2={mt['r2']:+.4f}  RMSE={mt['rmse']:.5f}  "
          f"MAE={mt['mae']:.5f}  n_val={mt['n_val']:,}  "
          f"({mt['secs']:.0f}s)", flush=True)
    # checkpoint after every fold so a disconnect cannot lose completed work
    pd.DataFrame(rows).to_csv(os.path.join(D, "rf_cv_metrics_partial.csv"),
                              index=False)
    np.save(os.path.join(D, "rf_oof_partial.npy"), oof)

cv = (pd.DataFrame(rows).sort_values("fold")
      [["fold", "n_train", "n_val", "val_cells", "r2", "rmse", "mae", "secs"]]
      .reset_index(drop=True))
assert len(cv) == N_FOLDS and set(cv.fold) == set(range(N_FOLDS)), \
    f"expected {N_FOLDS} folds, got {sorted(cv.fold)}"
assert np.isfinite(oof).all(), "out-of-fold predictions incomplete"
print("\n  per-fold:")
print(cv.round(5).to_string(index=False))
print(f"\n  MEAN  R2={cv.r2.mean():+.4f} (sd {cv.r2.std():.4f})   "
      f"RMSE={cv.rmse.mean():.5f} (sd {cv.rmse.std():.5f})   "
      f"MAE={cv.mae.mean():.5f} (sd {cv.mae.std():.5f})")
oof_m = metrics(ytr, oof)
print(f"  POOLED out-of-fold: R2={oof_m['r2']:+.4f}  "
      f"RMSE={oof_m['rmse']:.5f}  MAE={oof_m['mae']:.5f}")

# ------------------------------------------------------------- final model
banner("[2] FINAL RF ON ALL 2001-2019")
t0 = time.time()
rf = make().fit(Xtr, ytr)
print(f"  trained on {len(Xtr):,} rows in {time.time()-t0:.0f}s")
ptr = rf.predict(Xtr)
mtr = metrics(ytr, ptr)
print(f"  in-sample (optimistic): R2={mtr['r2']:+.4f}  RMSE={mtr['rmse']:.5f}")

# ----------------------------------------------------------------- holdout
banner("[3] HOLDOUT 2020-2023")
pho = rf.predict(Xho)
mho = metrics(yho, pho)
print(f"  MONTHLY  R2={mho['r2']:+.4f}  RMSE={mho['rmse']:.5f}  "
      f"MAE={mho['mae']:.5f}  n={len(yho):,}")

# reference baselines
zero_m = metrics(yho, np.zeros_like(yho))
trmean = metrics(yho, np.full_like(yho, ytr.mean()))
print(f"\n  baselines on the same holdout:")
print(f"    predict 0 (= climatology): R2={zero_m['r2']:+.4f}  "
      f"RMSE={zero_m['rmse']:.5f}  MAE={zero_m['mae']:.5f}")
print(f"    predict train mean       : R2={trmean['r2']:+.4f}  "
      f"RMSE={trmean['rmse']:.5f}")
print(f"    RF vs climatology RMSE   : "
      f"{100*(1-mho['rmse']/zero_m['rmse']):+.2f}% change")

bias = float(pho.mean() - yho.mean())
r2_deb = r2_score(yho - yho.mean(), pho - pho.mean())
print(f"\n  mean bias (pred - obs) : {bias:+.5f}")
print(f"  R2 after removing bias : {r2_deb:+.4f}")
print(f"  pearson r (monthly)    : {np.corrcoef(yho, pho)[0,1]:+.4f}")

# ----------------------------------------------- spatial aggregate (per cell)
banner("[4] SPATIAL AGGREGATE - 2020-2023 MEAN PER GRID CELL")
agg = (ho.assign(obs=yho, pred=pho)
       .groupby("point_id")[["obs", "pred"]].mean())
ar = float(agg.obs.corr(agg.pred))
am = metrics(agg.obs, agg.pred)
print(f"  cells={len(agg):,}")
print(f"  pearson r            : {ar:+.4f}   (r^2 = {ar**2:.4f})")
print(f"  R2                   : {am['r2']:+.4f}")
print(f"  R2 after removing bias: "
      f"{r2_score(agg.obs-agg.obs.mean(), agg.pred-agg.pred.mean()):+.4f}")
print(f"  RMSE                 : {am['rmse']:.5f}")
print(f"  obs  sd across cells : {agg.obs.std():.5f}")
print(f"  pred sd across cells : {agg.pred.std():.5f}")

print("\n  per-year spatial aggregate (cell means within each holdout year):")
yr_rows = []
for y in sorted(ho.year.unique()):
    s = ho.assign(obs=yho, pred=pho).query("year == @y")
    a = s.groupby("point_id")[["obs", "pred"]].mean()
    yr_rows.append(dict(year=int(y), cells=len(a),
                        r=float(a.obs.corr(a.pred)),
                        obs_mean=float(a.obs.mean()),
                        pred_mean=float(a.pred.mean())))
print(pd.DataFrame(yr_rows).round(4).to_string(index=False))

print("\n  growing season only (Jun-Oct), cell means:")
gs = ho.assign(obs=yho, pred=pho).query("month >= 6 and month <= 10")
ga = gs.groupby("point_id")[["obs", "pred"]].mean()
print(f"    n={len(gs):,} rows -> {len(ga):,} cells   "
      f"pearson r={float(ga.obs.corr(ga.pred)):+.4f}   "
      f"monthly r={np.corrcoef(gs.obs, gs.pred)[0,1]:+.4f}")

# ------------------------------------------------------------- importances
banner("[5] FEATURE IMPORTANCE (impurity-based)")
imp = (pd.Series(rf.feature_importances_, index=FEATURES)
       .sort_values(ascending=False))
print(imp.round(4).to_string())

# ------------------------------------------------------------------- save
banner("[6] SAVE")
joblib.dump(rf, os.path.join(D, "rf_model.joblib"), compress=3)
print(f"  rf_model.joblib ("
      f"{os.path.getsize(os.path.join(D,'rf_model.joblib'))/1e6:.1f} MB)")
cv.to_csv(os.path.join(D, "rf_cv_metrics.csv"), index=False)
print("  rf_cv_metrics.csv")
out = ho[["point_id", "lon", "lat", "year", "month", "landcover_class",
          "landcover_name", "landcover_grouped", TARGET]].copy()
out["rf_pred"] = pho
out.to_parquet(os.path.join(D, "rf_predictions_holdout.parquet"), index=False)
print("  rf_predictions_holdout.parquet")
tr_oof = tr[["point_id", "year", "month", "fold", TARGET]].copy()
tr_oof["rf_oof"] = oof
tr_oof.to_parquet(os.path.join(D, "rf_oof_train.parquet"), index=False)
print("  rf_oof_train.parquet")
imp.rename("importance").to_csv(os.path.join(D, "rf_importance.csv"))
print("  rf_importance.csv")

summary = dict(
    model="RandomForestRegressor", n_estimators=N_TREES, max_depth=MAX_DEPTH,
    seed=SEED, n_features=len(FEATURES), n_train=int(len(tr)),
    n_holdout=int(len(ho)),
    cv_mean={k: float(cv[k].mean()) for k in ("r2", "rmse", "mae")},
    cv_sd={k: float(cv[k].std()) for k in ("r2", "rmse", "mae")},
    cv_per_fold=cv.to_dict("records"),
    oof_pooled=oof_m, holdout_monthly=mho,
    holdout_bias=bias, holdout_r2_debiased=float(r2_deb),
    holdout_pearson_r=float(np.corrcoef(yho, pho)[0, 1]),
    baseline_zero=zero_m, spatial_aggregate_r=ar,
    spatial_aggregate_r2=float(am["r2"]),
    growing_season_spatial_r=float(ga.obs.corr(ga.pred)))
json.dump(summary, open(os.path.join(D, "rf_summary.json"), "w"),
          indent=2, default=float)
print("  rf_summary.json")
banner("STEP 3 COMPLETE")
