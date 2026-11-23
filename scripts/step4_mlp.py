"""Step 4: MLP (128-64-32-1) with the same spatial block CV as the RF.

Early-stopping design
---------------------
Early stopping needs a validation signal. Using the outer CV fold for that
would select the stopping epoch on the very data the fold score reports, so
each fold instead carves an INNER validation set of 15% of the *blocks* from
its training portion. Blocks (not rows) keep the inner split spatially
separated too. The outer fold stays untouched until scoring.

StandardScaler is fit on the outer training portion only (inner-val included,
since that is still training data) and never on the outer fold or the holdout.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

D = r"C:\Users\USER\Documents\GeoAI_dryland"
pd.set_option("display.width", 220)

SEED = 42
BATCH = 4096
LR = 1e-3
# 16.5 s/epoch at full size (measured, 4 threads, identical for batch 4096 and
# 16384 -> compute-bound). Early stopping normally fires well before this cap;
# 100 bounds the worst case at ~28 min per fit.
MAX_EPOCHS = 100
PATIENCE = 15
DROPOUT = 0.2
INNER_VAL_FRAC = 0.15

torch.set_num_threads(max(1, os.cpu_count() or 2))
torch.manual_seed(SEED)
np.random.seed(SEED)
DEV = torch.device("cpu")


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78, flush=True)


def metrics(y, p):
    return dict(r2=float(r2_score(y, p)),
                rmse=float(mean_squared_error(y, p) ** 0.5),
                mae=float(mean_absolute_error(y, p)))


def build(n_in):
    return nn.Sequential(
        nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(DROPOUT),
        nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(DROPOUT),
        nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(DROPOUT),
        nn.Linear(32, 1))


@torch.no_grad()
def predict(model, X, bs=65536):
    model.eval()
    out = [model(X[i:i + bs]).squeeze(1) for i in range(0, len(X), bs)]
    return torch.cat(out).numpy()


def train_mlp(Xtr, ytr, Xva, yva, tag, seed=SEED):
    """Train with Adam + early stopping on inner-val MSE; restore best weights."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = build(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    n = len(Xtr)
    best, best_state, best_ep, bad = np.inf, None, -1, 0
    hist = []
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:          # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]).squeeze(1), ytr[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        vp = predict(model, Xva)
        vmse = float(np.mean((vp - yva.numpy()) ** 2))
        hist.append(vmse)
        if vmse < best - 1e-9:
            best, best_ep, bad = vmse, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 10 == 0 or bad == 0:
            print(f"      [{tag}] ep{ep:3d} train_mse={tot/n:.6f} "
                  f"val_mse={vmse:.6f}{'  *' if bad == 0 else ''}", flush=True)
        if bad >= PATIENCE:
            print(f"      [{tag}] early stop at epoch {ep} "
                  f"(best epoch {best_ep}, val_mse {best:.6f})", flush=True)
            break
    model.load_state_dict(best_state)
    return model, best_ep, best, len(hist)


def inner_split(block_ids, frac=INNER_VAL_FRAC, seed=SEED):
    """Hold out `frac` of distinct blocks (not rows) as inner validation."""
    ub = np.unique(block_ids)
    rng = np.random.default_rng(seed)
    pick = rng.choice(ub, size=max(1, int(round(len(ub) * frac))), replace=False)
    return np.isin(block_ids, pick)


banner("STEP 4 - MLP (128-64-32-1)")

cfg = json.load(open(os.path.join(D, "cv_config.json")))
TARGET, FEATURES, N_FOLDS = cfg["target"], cfg["features"], cfg["n_folds"]
df = pd.read_parquet(os.path.join(D, "model_matrix.parquet"))
tr = df[df.split == "train"].reset_index(drop=True)
ho = df[df.split == "holdout"].reset_index(drop=True)

print(f"\nTARGET: {TARGET}  (sd train={tr[TARGET].std():.5f}, "
      f"holdout={ho[TARGET].std():.5f})")
print(f"train {len(tr):,} | holdout {len(ho):,} | {len(FEATURES)} features")
print(f"arch 128-64-32-1, ReLU, BatchNorm, dropout {DROPOUT}; Adam lr={LR}, "
      f"batch {BATCH}, patience {PATIENCE}, max_epochs {MAX_EPOCHS}")
print(f"torch {torch.__version__}, threads {torch.get_num_threads()}")

Xtr_raw = tr[FEATURES].to_numpy(np.float32)
ytr_np = tr[TARGET].to_numpy(np.float32)
Xho_raw = ho[FEATURES].to_numpy(np.float32)
yho_np = ho[TARGET].to_numpy(np.float32)
folds = tr.fold.to_numpy()
blocks = tr.block_id.to_numpy()

# --------------------------------------------------------------- CV
banner(f"[1] SPATIAL BLOCK {N_FOLDS}-FOLD CV  (2001-2019)")
PART = os.path.join(D, "mlp_cv_metrics_partial.csv")
POOF = os.path.join(D, "mlp_oof_partial.npy")
rows, done = [], set()
oof = np.full(len(tr), np.nan, dtype=np.float64)
if os.path.exists(PART) and os.path.exists(POOF):
    prev = pd.read_csv(PART)
    cached = np.load(POOF)
    if "target" in prev.columns and set(prev.target) == {TARGET} \
            and len(cached) == len(tr):
        rows, oof, done = prev.to_dict("records"), cached, set(prev.fold.astype(int))
        print(f"  RESUMING: folds {sorted(done)} cached")
    else:
        print("  checkpoint target/shape mismatch - starting fresh")

for f in range(N_FOLDS):
    if f in done:
        continue
    t0 = time.time()
    va = folds == f
    ot = ~va                                   # outer-train rows
    iv = np.zeros(len(tr), bool)
    iv[ot] = inner_split(blocks[ot], seed=SEED + f)
    it = ot & ~iv                              # inner-train
    sc = StandardScaler().fit(Xtr_raw[ot])     # fit on outer-train only
    Xi = torch.from_numpy(sc.transform(Xtr_raw[it]).astype(np.float32))
    Xv = torch.from_numpy(sc.transform(Xtr_raw[iv]).astype(np.float32))
    Xo = torch.from_numpy(sc.transform(Xtr_raw[va]).astype(np.float32))
    yi = torch.from_numpy(ytr_np[it])
    yv = torch.from_numpy(ytr_np[iv])
    print(f"  fold {f}: inner-train {it.sum():,} | inner-val {iv.sum():,} "
          f"({len(np.unique(blocks[iv]))} blocks) | outer-val {va.sum():,}",
          flush=True)
    model, bep, bval, neps = train_mlp(Xi, yi, Xv, yv, f"fold{f}", seed=SEED + f)
    p = predict(model, Xo)
    oof[va] = p
    mt = metrics(ytr_np[va], p)
    mt.update(fold=f, n_train=int(it.sum()), n_val=int(va.sum()),
              val_cells=int(tr.loc[va, "point_id"].nunique()),
              best_epoch=int(bep), epochs_run=int(neps),
              inner_val_mse=float(bval), secs=round(time.time() - t0, 1),
              target=TARGET)
    rows.append(mt)
    print(f"  fold {f}: R2={mt['r2']:+.4f}  RMSE={mt['rmse']:.5f}  "
          f"MAE={mt['mae']:.5f}  (best ep {bep}, {mt['secs']:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(PART, index=False)
    np.save(POOF, oof)

cv = (pd.DataFrame(rows).sort_values("fold")
      [["fold", "n_train", "n_val", "val_cells", "r2", "rmse", "mae",
        "best_epoch", "epochs_run", "secs"]].reset_index(drop=True))
assert len(cv) == N_FOLDS and np.isfinite(oof).all()
print("\n  per-fold:")
print(cv.round(5).to_string(index=False))
print(f"\n  MEAN  R2={cv.r2.mean():+.4f} (sd {cv.r2.std():.4f})   "
      f"RMSE={cv.rmse.mean():.5f} (sd {cv.rmse.std():.5f})   "
      f"MAE={cv.mae.mean():.5f} (sd {cv.mae.std():.5f})")
oof_m = metrics(ytr_np, oof)
print(f"  POOLED out-of-fold: R2={oof_m['r2']:+.4f}  RMSE={oof_m['rmse']:.5f}  "
      f"MAE={oof_m['mae']:.5f}")

# ------------------------------------------------------------ final model
banner("[2] FINAL MLP ON ALL 2001-2019")
iv = inner_split(blocks, seed=SEED + 99)
scaler = StandardScaler().fit(Xtr_raw)
Xi = torch.from_numpy(scaler.transform(Xtr_raw[~iv]).astype(np.float32))
Xv = torch.from_numpy(scaler.transform(Xtr_raw[iv]).astype(np.float32))
yi = torch.from_numpy(ytr_np[~iv])
yv = torch.from_numpy(ytr_np[iv])
print(f"  inner-train {(~iv).sum():,} | inner-val {iv.sum():,} "
      f"({len(np.unique(blocks[iv]))} blocks held out for early stopping)")
t0 = time.time()
final, bep, bval, neps = train_mlp(Xi, yi, Xv, yv, "final", seed=SEED)
print(f"  trained in {time.time()-t0:.0f}s, best epoch {bep} of {neps}")
Xtr_s = torch.from_numpy(scaler.transform(Xtr_raw).astype(np.float32))
mtr = metrics(ytr_np, predict(final, Xtr_s))
print(f"  in-sample (optimistic): R2={mtr['r2']:+.4f}  RMSE={mtr['rmse']:.5f}")

# ---------------------------------------------------------------- holdout
banner("[3] HOLDOUT 2020-2023")
Xho = torch.from_numpy(scaler.transform(Xho_raw).astype(np.float32))
pho = predict(final, Xho)
mho = metrics(yho_np, pho)
print(f"  MONTHLY  R2={mho['r2']:+.4f}  RMSE={mho['rmse']:.5f}  "
      f"MAE={mho['mae']:.5f}  n={len(yho_np):,}")
zero_m = metrics(yho_np, np.zeros_like(yho_np))
print(f"\n  baselines:")
print(f"    predict 0 (= climatology): R2={zero_m['r2']:+.4f}  "
      f"RMSE={zero_m['rmse']:.5f}  MAE={zero_m['mae']:.5f}")
print(f"    MLP vs climatology RMSE  : "
      f"{100*(1-mho['rmse']/zero_m['rmse']):+.2f}% change")
bias = float(pho.mean() - yho_np.mean())
print(f"\n  mean bias (pred - obs) : {bias:+.5f}")
print(f"  R2 after removing bias : "
      f"{r2_score(yho_np - yho_np.mean(), pho - pho.mean()):+.4f}")
print(f"  pearson r (monthly)    : {np.corrcoef(yho_np, pho)[0,1]:+.4f}")

# ------------------------------------------------------- spatial aggregate
banner("[4] SPATIAL AGGREGATE - 2020-2023 MEAN PER GRID CELL")
agg = (ho[["point_id"]].assign(obs=yho_np, pred=pho)
       .groupby("point_id")[["obs", "pred"]].mean())
ar = float(agg.obs.corr(agg.pred))
am = metrics(agg.obs, agg.pred)
print(f"  cells={len(agg):,}")
print(f"  pearson r             : {ar:+.4f}   (r^2 = {ar**2:.4f})")
print(f"  R2                    : {am['r2']:+.4f}")
print(f"  R2 after removing bias: "
      f"{r2_score(agg.obs-agg.obs.mean(), agg.pred-agg.pred.mean()):+.4f}")
print(f"  RMSE                  : {am['rmse']:.5f}")
print(f"  obs  sd across cells  : {agg.obs.std():.5f}")
print(f"  pred sd across cells  : {agg.pred.std():.5f} "
      f"({agg.pred.std()/agg.obs.std():.1%} of observed spread)")
print("\n  per-year spatial aggregate:")
yr = []
for y in sorted(ho.year.unique()):
    s = ho[["point_id"]].assign(obs=yho_np, pred=pho, year=ho.year.to_numpy())
    s = s[s.year == y]
    a = s.groupby("point_id")[["obs", "pred"]].mean()
    yr.append(dict(year=int(y), cells=len(a), r=float(a.obs.corr(a.pred)),
                   obs_mean=float(a.obs.mean()), pred_mean=float(a.pred.mean())))
print(pd.DataFrame(yr).round(4).to_string(index=False))
gsm = ho.month.between(6, 10).to_numpy()
ga = (ho[["point_id"]][gsm].assign(obs=yho_np[gsm], pred=pho[gsm])
      .groupby("point_id")[["obs", "pred"]].mean())
print(f"\n  growing season (Jun-Oct): {int(gsm.sum()):,} rows -> {len(ga):,} cells"
      f"   cell-mean r={float(ga.obs.corr(ga.pred)):+.4f}   "
      f"monthly r={np.corrcoef(yho_np[gsm], pho[gsm])[0,1]:+.4f}")

# ------------------------------------------------------------------- save
banner("[5] SAVE")
torch.save(dict(state_dict=final.state_dict(), features=FEATURES, target=TARGET,
                arch=[128, 64, 32, 1], dropout=DROPOUT, seed=SEED),
           os.path.join(D, "mlp_model.pt"))
print(f"  mlp_model.pt ({os.path.getsize(os.path.join(D,'mlp_model.pt'))/1e6:.2f} MB)")
joblib.dump(scaler, os.path.join(D, "mlp_scaler.joblib"))
print("  mlp_scaler.joblib")
cv.to_csv(os.path.join(D, "mlp_cv_metrics.csv"), index=False)
print("  mlp_cv_metrics.csv")
out = ho[["point_id", "lon", "lat", "year", "month", "landcover_class",
          "landcover_name", "landcover_grouped", TARGET]].copy()
out["mlp_pred"] = pho
out.to_parquet(os.path.join(D, "mlp_predictions_holdout.parquet"), index=False)
print("  mlp_predictions_holdout.parquet")
o = tr[["point_id", "year", "month", "fold", TARGET]].copy()
o["mlp_oof"] = oof
o.to_parquet(os.path.join(D, "mlp_oof_train.parquet"), index=False)
print("  mlp_oof_train.parquet")
json.dump(dict(model="MLP-128-64-32-1", target=TARGET, dropout=DROPOUT, lr=LR,
               batch=BATCH, patience=PATIENCE, max_epochs=MAX_EPOCHS,
               inner_val_frac=INNER_VAL_FRAC, seed=SEED,
               cv_mean={k: float(cv[k].mean()) for k in ("r2", "rmse", "mae")},
               cv_sd={k: float(cv[k].std()) for k in ("r2", "rmse", "mae")},
               cv_per_fold=cv.to_dict("records"), oof_pooled=oof_m,
               holdout_monthly=mho, holdout_bias=bias,
               holdout_pearson_r=float(np.corrcoef(yho_np, pho)[0, 1]),
               baseline_zero=zero_m, spatial_aggregate_r=ar,
               spatial_aggregate_r2=float(am["r2"]),
               growing_season_spatial_r=float(ga.obs.corr(ga.pred))),
          open(os.path.join(D, "mlp_summary.json"), "w"), indent=2, default=float)
print("  mlp_summary.json")
banner("STEP 4 COMPLETE")
