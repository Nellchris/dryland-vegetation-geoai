"""Step 7: residual analysis for both models on the 2020-2023 holdout.

residual = observed - predicted  (positive = model under-predicted greenness)

"Poorly predicted" is applied at two levels:
  monthly - any row with |residual| > 1.5 x that model's holdout RMSE
  cell    - any cell whose own RMSE over the holdout > 1.5 x the overall RMSE
Cells failing for BOTH models feed the Step 8 residual-hotspot layer.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = r"C:\Users\USER\Documents\GeoAI_dryland"
FIG = os.path.join(D, "figures")
os.makedirs(FIG, exist_ok=True)
pd.set_option("display.width", 240)
K = 1.5


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78, flush=True)


banner("STEP 7 - RESIDUAL ANALYSIS")
cfg = json.load(open(os.path.join(D, "cv_config.json")))
TARGET = cfg["target"]
rf = pd.read_parquet(os.path.join(D, "rf_predictions_holdout.parquet"))
ml = pd.read_parquet(os.path.join(D, "mlp_predictions_holdout.parquet"))
KEY = ["point_id", "year", "month"]
m = rf.merge(ml[KEY + ["mlp_pred"]], on=KEY, how="inner", validate="one_to_one")
m["rf_resid"] = m[TARGET] - m.rf_pred
m["mlp_resid"] = m[TARGET] - m.mlp_pred
print(f"\nholdout rows {len(m):,} | cells {m.point_id.nunique():,} | target {TARGET}")

RMSE = {k: float(np.sqrt((m[f"{k}_resid"] ** 2).mean())) for k in ("rf", "mlp")}
THR = {k: K * v for k, v in RMSE.items()}
print("\n[1] MONTHLY RESIDUAL DISTRIBUTION")
for k in ("rf", "mlp"):
    r = m[f"{k}_resid"]
    print(f"  {k.upper():<4} RMSE={RMSE[k]:.5f}  threshold(1.5x)={THR[k]:.5f}")
    print(f"       mean={r.mean():+.5f}  sd={r.std():.5f}  "
          f"MAE={r.abs().mean():.5f}")
    print(f"       pct |resid| > 1.5xRMSE : {100*(r.abs() > THR[k]).mean():.2f}%")
    print(f"       quantiles 1/25/50/75/99: "
          + "  ".join(f"{r.quantile(q):+.4f}" for q in (.01, .25, .5, .75, .99)))
m["rf_poor_m"] = m.rf_resid.abs() > THR["rf"]
m["mlp_poor_m"] = m.mlp_resid.abs() > THR["mlp"]
m["both_poor_m"] = m.rf_poor_m & m.mlp_poor_m
print(f"  rows poor for BOTH models: {int(m.both_poor_m.sum()):,} "
      f"({100*m.both_poor_m.mean():.2f}%)")
print(f"  overlap: of RF-poor rows, {100*m.loc[m.rf_poor_m,'mlp_poor_m'].mean():.1f}%"
      f" are also MLP-poor")
print(f"  residual correlation rf vs mlp: "
      f"{m.rf_resid.corr(m.mlp_resid):+.4f}")

# -------------------------------------------------------------- per cell
print("\n[2] CELL-LEVEL AGGREGATION")
g = (m.groupby("point_id")
     .agg(lat=("lat", "first"), lon=("lon", "first"),
          landcover_class=("landcover_class", "first"),
          landcover_name=("landcover_name", "first"),
          landcover_grouped=("landcover_grouped", "first"),
          n_months=("rf_resid", "size"),
          observed_mean=(TARGET, "mean"),
          rf_resid_mean=("rf_resid", "mean"),
          mlp_resid_mean=("mlp_resid", "mean"),
          rf_abs_resid_mean=("rf_resid", lambda s: s.abs().mean()),
          mlp_abs_resid_mean=("mlp_resid", lambda s: s.abs().mean()),
          rf_rmse=("rf_resid", lambda s: float(np.sqrt((s ** 2).mean()))),
          mlp_rmse=("mlp_resid", lambda s: float(np.sqrt((s ** 2).mean()))),
          rf_poor_months=("rf_poor_m", "sum"),
          mlp_poor_months=("mlp_poor_m", "sum"))
     .reset_index())
g["rf_poor"] = g.rf_rmse > THR["rf"]
g["mlp_poor"] = g.mlp_rmse > THR["mlp"]
g["both_poor"] = g.rf_poor & g.mlp_poor
g["either_poor"] = g.rf_poor | g.mlp_poor
g["mean_abs_resid"] = (g.rf_abs_resid_mean + g.mlp_abs_resid_mean) / 2
print(f"  cells {len(g):,}")
for k in ("rf", "mlp"):
    print(f"  {k.upper():<4} cell RMSE: mean={g[f'{k}_rmse'].mean():.5f} "
          f"median={g[f'{k}_rmse'].median():.5f} max={g[f'{k}_rmse'].max():.5f}")
    print(f"       poorly predicted cells: {int(g[f'{k}_poor'].sum()):,} "
          f"({100*g[f'{k}_poor'].mean():.1f}%)")
print(f"  BOTH models poor   : {int(g.both_poor.sum()):,} "
      f"({100*g.both_poor.mean():.1f}%)   <- Step 8 hotspot layer")
print(f"  EITHER model poor  : {int(g.either_poor.sum()):,} "
      f"({100*g.either_poor.mean():.1f}%)")
print(f"  cell RMSE corr rf vs mlp: {g.rf_rmse.corr(g.mlp_rmse):+.4f}")
print(f"  mean residual sign: RF {100*(g.rf_resid_mean>0).mean():.1f}% of cells "
      f"under-predicted; MLP {100*(g.mlp_resid_mean>0).mean():.1f}%")

# ------------------------------------------------------- by land cover
print("\n[3] STRATIFIED BY LAND COVER - WHICH ARE HARDEST TO PREDICT?")
for key, lbl in (("landcover_grouped", "pooled"), ("landcover_name", "raw")):
    t = (g.groupby(key)
         .agg(cells=("point_id", "size"),
              rf_rmse=("rf_rmse", "mean"), mlp_rmse=("mlp_rmse", "mean"),
              mean_abs_resid=("mean_abs_resid", "mean"),
              rf_poor_pct=("rf_poor", lambda s: 100 * s.mean()),
              mlp_poor_pct=("mlp_poor", lambda s: 100 * s.mean()),
              both_poor_pct=("both_poor", lambda s: 100 * s.mean()),
              obs_sd=("observed_mean", "std"))
         .sort_values("mean_abs_resid", ascending=False).reset_index())
    print(f"\n  by {key} ({lbl}), hardest first:")
    print(t.round(5).to_string(index=False))
    if lbl == "pooled":
        strat = t

print("\n  interpretation: mean |residual| scales with how variable a class's")
print("  observed anomaly is, so the ranking reflects intrinsic variability as")
print("  much as model failure. Normalised error (cell RMSE / observed sd):")
norm = (g.groupby("landcover_grouped")
        .apply(lambda d: pd.Series({
            "cells": len(d),
            "rf_rmse": d.rf_rmse.mean(),
            "obs_sd_within": d.observed_mean.std(),
            "rf_rmse_over_obs_sd": d.rf_rmse.mean() / d.observed_mean.std()
            if d.observed_mean.std() > 0 else np.nan}), include_groups=False)
        .sort_values("rf_rmse_over_obs_sd", ascending=False))
print(norm.round(4).to_string())

# ------------------------------------------------------------ by season
print("\n[4] STRATIFIED BY MONTH AND YEAR")
mo = (m.groupby("month")
      .agg(rows=("rf_resid", "size"),
           rf_rmse=("rf_resid", lambda s: float(np.sqrt((s ** 2).mean()))),
           mlp_rmse=("mlp_resid", lambda s: float(np.sqrt((s ** 2).mean()))),
           obs_sd=(TARGET, "std")).reset_index())
print(mo.round(5).to_string(index=False))
yr = (m.groupby("year")
      .agg(rows=("rf_resid", "size"),
           rf_rmse=("rf_resid", lambda s: float(np.sqrt((s ** 2).mean()))),
           mlp_rmse=("mlp_resid", lambda s: float(np.sqrt((s ** 2).mean()))),
           rf_bias=("rf_resid", "mean"), mlp_bias=("mlp_resid", "mean"))
      .reset_index())
print()
print(yr.round(5).to_string(index=False))

# ---------------------------------------------------------------- maps
print("\n[5] MAPS")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
lim = float(np.nanpercentile(np.abs(np.r_[g.rf_resid_mean, g.mlp_resid_mean]), 99))
for ax, (col, ttl) in zip(axes.flat[:2], [
        ("rf_resid_mean", "RF mean residual (obs - pred)"),
        ("mlp_resid_mean", "MLP mean residual (obs - pred)")]):
    s = ax.scatter(g.lon, g.lat, c=g[col], s=7, cmap="RdBu_r", vmin=-lim, vmax=lim)
    plt.colorbar(s, ax=ax, label="residual")
    ax.set_title(f"{ttl}\nmean {g[col].mean():+.5f}")
s = axes.flat[2].scatter(g.lon, g.lat, c=g.mean_abs_resid, s=7, cmap="magma_r")
plt.colorbar(s, ax=axes.flat[2], label="mean |residual|")
axes.flat[2].set_title("Mean |residual|, both models averaged")
ax = axes.flat[3]
ax.scatter(g.lon, g.lat, c="#dddddd", s=6, label="ok")
for flag, col, lb in ((g.rf_poor & ~g.mlp_poor, "#3498db", "RF only"),
                      (g.mlp_poor & ~g.rf_poor, "#e67e22", "MLP only"),
                      (g.both_poor, "#c0392b", "BOTH fail")):
    s2 = g[flag]
    if len(s2):
        ax.scatter(s2.lon, s2.lat, c=col, s=9, label=f"{lb} ({len(s2):,})")
ax.legend(loc="lower left", fontsize=8, markerscale=2)
ax.set_title(f"Poorly predicted cells (cell RMSE > {K}x overall)")
for a in axes.flat:
    a.set_xlabel("longitude")
    a.set_ylabel("latitude")
    a.set_aspect("equal")
fig.suptitle("Sokoto Basin - holdout residuals, 2020-2023", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "step7_residual_maps.png"), dpi=140)
plt.close(fig)
print("  figures/step7_residual_maps.png")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(m.rf_resid, bins=120, alpha=.6, label="RF", color="#2b7a4b")
axes[0].hist(m.mlp_resid, bins=120, alpha=.5, label="MLP", color="#8e44ad")
for k, c in (("rf", "#2b7a4b"), ("mlp", "#8e44ad")):
    for sgn in (-1, 1):
        axes[0].axvline(sgn * THR[k], color=c, ls="--", lw=.9)
axes[0].set_xlabel("residual (obs - pred)")
axes[0].set_ylabel("rows")
axes[0].set_title(f"Monthly residuals\ndashed = +/-{K}x RMSE")
axes[0].legend()
b = strat.sort_values("mean_abs_resid")
axes[1].barh(b.landcover_grouped, b.mean_abs_resid, color="#c0392b")
axes[1].set_xlabel("mean |residual|")
axes[1].set_title("Mean |residual| by land cover")
axes[1].grid(axis="x", alpha=.3)
axes[2].plot(mo.month, mo.rf_rmse, "o-", label="RF", color="#2b7a4b")
axes[2].plot(mo.month, mo.mlp_rmse, "s-", label="MLP", color="#8e44ad")
axes[2].plot(mo.month, mo.obs_sd, "^--", label="observed sd", color="#555")
axes[2].set_xlabel("month")
axes[2].set_ylabel("RMSE")
axes[2].set_title("Error by calendar month")
axes[2].legend()
axes[2].grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "step7_residual_diagnostics.png"), dpi=140)
plt.close(fig)
print("  figures/step7_residual_diagnostics.png")

# ---------------------------------------------------------------- save
print("\n[6] SAVE")
COLS = ["point_id", "lat", "lon", "landcover_class", "landcover_name",
        "landcover_grouped", "n_months", "observed_mean",
        "rf_resid_mean", "mlp_resid_mean", "rf_abs_resid_mean",
        "mlp_abs_resid_mean", "rf_rmse", "mlp_rmse", "mean_abs_resid",
        "rf_poor_months", "mlp_poor_months",
        "rf_poor", "mlp_poor", "both_poor", "either_poor"]
g[COLS].sort_values("point_id").to_csv(os.path.join(D, "residuals.csv"),
                                       index=False)
print(f"  residuals.csv ({len(g):,} rows, {len(COLS)} cols, "
      f"{os.path.getsize(os.path.join(D,'residuals.csv'))/1e3:.0f} KB)")
m[KEY + ["lat", "lon", TARGET, "rf_pred", "mlp_pred", "rf_resid",
         "mlp_resid", "rf_poor_m", "mlp_poor_m", "both_poor_m"]] \
    .to_parquet(os.path.join(D, "residuals_monthly.parquet"), index=False)
print("  residuals_monthly.parquet (row-level residuals)")
json.dump(dict(target=TARGET, k=K, rmse=RMSE, thresholds=THR,
               monthly_rows=int(len(m)), cells=int(len(g)),
               monthly_poor_pct={k: float(100 * m[f"{k}_poor_m"].mean())
                                 for k in ("rf", "mlp")},
               monthly_both_poor=int(m.both_poor_m.sum()),
               residual_corr=float(m.rf_resid.corr(m.mlp_resid)),
               cells_poor={k: int(g[f"{k}_poor"].sum()) for k in ("rf", "mlp")},
               cells_both_poor=int(g.both_poor.sum()),
               cells_either_poor=int(g.either_poor.sum()),
               by_landcover=strat.to_dict("records"),
               by_month=mo.to_dict("records"), by_year=yr.to_dict("records")),
          open(os.path.join(D, "residuals_summary.json"), "w"),
          indent=2, default=float)
print("  residuals_summary.json")
banner("STEP 7 COMPLETE")
