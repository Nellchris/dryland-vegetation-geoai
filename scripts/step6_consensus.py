"""Step 6: Model consensus and disagreement mapping (2020-2023 holdout).

Per grid cell, average each model's predicted NDVI anomaly over the holdout
period, then:
    consensus    = mean(rf, mlp)
    disagreement = |rf - mlp|
    category     = both stressed / both healthy / models disagree  (by sign)

Because both models are strongly under-dispersed (predicted spatial spread is
~6-8% of observed), a sign split puts many cells very close to zero. A
dead-band variant is therefore also computed as a sensitivity check and
carried in the output for the web map.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.metrics import mean_squared_error, r2_score

D = r"C:\Users\USER\Documents\GeoAI_dryland"
FIG = os.path.join(D, "figures")
os.makedirs(FIG, exist_ok=True)
pd.set_option("display.width", 220)


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78, flush=True)


banner("STEP 6 - MODEL CONSENSUS + DISAGREEMENT MAPPING")
cfg = json.load(open(os.path.join(D, "cv_config.json")))
TARGET = cfg["target"]

rf = pd.read_parquet(os.path.join(D, "rf_predictions_holdout.parquet"))
ml = pd.read_parquet(os.path.join(D, "mlp_predictions_holdout.parquet"))
KEY = ["point_id", "year", "month"]
m = rf.merge(ml[KEY + ["mlp_pred"]], on=KEY, how="inner", validate="one_to_one")
assert len(m) == len(rf) == len(ml), "prediction tables must align row-for-row"
print(f"\nmonthly holdout predictions joined: {len(m):,} rows "
      f"({m.point_id.nunique():,} cells x {m.year.nunique()} years)")
print(f"target: {TARGET}")

# ------------------------------------------------- monthly-level agreement
print("\n[1] MONTHLY-LEVEL AGREEMENT (before cell aggregation)")
print(f"  corr(rf_pred, mlp_pred)       : {m.rf_pred.corr(m.mlp_pred):+.4f}")
print(f"  rf  pred: mean={m.rf_pred.mean():+.5f} sd={m.rf_pred.std():.5f}")
print(f"  mlp pred: mean={m.mlp_pred.mean():+.5f} sd={m.mlp_pred.std():.5f}")
print(f"  obs     : mean={m[TARGET].mean():+.5f} sd={m[TARGET].std():.5f}")
same = (np.sign(m.rf_pred) == np.sign(m.mlp_pred)).mean()
print(f"  same sign at month level      : {100*same:.1f}%")

# ------------------------------------------------------- cell aggregation
print("\n[2] CELL-LEVEL AGGREGATION (mean over 2020-2023)")
g = (m.groupby("point_id")
     .agg(lat=("lat", "first"), lon=("lon", "first"),
          landcover_class=("landcover_class", "first"),
          landcover_name=("landcover_name", "first"),
          landcover_grouped=("landcover_grouped", "first"),
          rf_pred=("rf_pred", "mean"), mlp_pred=("mlp_pred", "mean"),
          observed=(TARGET, "mean"), n_months=("rf_pred", "size"))
     .reset_index())
g["consensus"] = (g.rf_pred + g.mlp_pred) / 2.0
g["disagreement"] = (g.rf_pred - g.mlp_pred).abs()
print(f"  cells: {len(g):,}  months per cell: min={g.n_months.min()} "
      f"max={g.n_months.max()}")
print(f"  rf_pred      : mean={g.rf_pred.mean():+.6f} sd={g.rf_pred.std():.6f} "
      f"[{g.rf_pred.min():+.5f}, {g.rf_pred.max():+.5f}]")
print(f"  mlp_pred     : mean={g.mlp_pred.mean():+.6f} sd={g.mlp_pred.std():.6f} "
      f"[{g.mlp_pred.min():+.5f}, {g.mlp_pred.max():+.5f}]")
print(f"  consensus    : mean={g.consensus.mean():+.6f} sd={g.consensus.std():.6f}")
print(f"  observed     : mean={g.observed.mean():+.6f} sd={g.observed.std():.6f}")
print(f"  disagreement : mean={g.disagreement.mean():.6f} "
      f"median={g.disagreement.median():.6f} max={g.disagreement.max():.6f}")
print(f"  corr(rf, mlp) at cell level: {g.rf_pred.corr(g.mlp_pred):+.4f}")

# -------------------------------------------------------- categorisation
print("\n[3] CATEGORISATION")


def categorise(rf_v, mlp_v, eps=0.0):
    """Sign agreement with an optional dead band around zero."""
    rs = np.where(rf_v < -eps, -1, np.where(rf_v > eps, 1, 0))
    ms = np.where(mlp_v < -eps, -1, np.where(mlp_v > eps, 1, 0))
    out = np.full(len(rf_v), "Models disagree", dtype=object)
    out[(rs == -1) & (ms == -1)] = "Both agree stressed"
    out[(rs == 1) & (ms == 1)] = "Both agree healthy"
    return out


g["category"] = categorise(g.rf_pred.to_numpy(), g.mlp_pred.to_numpy())
CATS = ["Both agree stressed", "Both agree healthy", "Models disagree"]
vc = g.category.value_counts().reindex(CATS).fillna(0).astype(int)
print("  primary (sign-based, no dead band):")
for c in CATS:
    print(f"    {c:<22} {vc[c]:>5,} cells ({100*vc[c]/len(g):5.1f}%)")

# sensitivity: dead band at the median |consensus|
EPS = float(np.median(np.abs(g.consensus)))
g["category_deadband"] = categorise(g.rf_pred.to_numpy(), g.mlp_pred.to_numpy(),
                                    eps=EPS)
print(f"\n  sensitivity (dead band eps = median|consensus| = {EPS:.6f}):")
vc2 = g.category_deadband.value_counts()
for c in vc2.index:
    print(f"    {c:<22} {vc2[c]:>5,} cells ({100*vc2[c]/len(g):5.1f}%)")
print("  (cells inside the dead band fall to 'Models disagree' by construction,")
print("   so this variant is a conservative read, not a second opinion.)")

# ---------------------------------------------- does consensus help at all?
print("\n[4] IS THE CONSENSUS BETTER THAN EITHER MODEL ALONE?")
rows = []
for nm, col in (("RF", "rf_pred"), ("MLP", "mlp_pred"), ("Consensus", "consensus")):
    rows.append(dict(model=nm, pearson_r=float(g.observed.corr(g[col])),
                     r2=float(r2_score(g.observed, g[col])),
                     rmse=float(mean_squared_error(g.observed, g[col]) ** 0.5),
                     pred_sd=float(g[col].std())))
cmp = pd.DataFrame(rows)
cmp["sd_vs_obs_pct"] = 100 * cmp.pred_sd / g.observed.std()
print(cmp.round(5).to_string(index=False))
print(f"  observed sd across cells: {g.observed.std():.5f}")

print("\n  does disagreement flag unreliable cells?")
g["abs_err_consensus"] = (g.consensus - g.observed).abs()
q = pd.qcut(g.disagreement, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
dq = g.groupby(q, observed=True).agg(
    cells=("point_id", "size"),
    mean_disagreement=("disagreement", "mean"),
    mean_abs_err=("abs_err_consensus", "mean"),
    obs_sd=("observed", "std")).reset_index()
print(dq.round(5).to_string(index=False))
print(f"  corr(disagreement, |consensus error|) = "
      f"{g.disagreement.corr(g.abs_err_consensus):+.4f}")

# --------------------------------------------------- agreement vs reality
print("\n[5] DO THE AGREED CATEGORIES MATCH OBSERVATION?")
chk = g.groupby("category", observed=True).agg(
    cells=("point_id", "size"), mean_obs=("observed", "mean"),
    obs_negative_pct=("observed", lambda s: 100 * (s < 0).mean()),
    mean_consensus=("consensus", "mean")).reindex(CATS)
print(chk.round(5).to_string())
for c in ("Both agree stressed", "Both agree healthy"):
    s = g[g.category == c]
    if len(s):
        want = (s.observed < 0) if "stressed" in c else (s.observed > 0)
        print(f"  {c}: {100*want.mean():.1f}% of those cells observed "
              f"{'negative' if 'stressed' in c else 'positive'}")

print("\n  by land cover (pooled classes):")
lc = (pd.crosstab(g.landcover_grouped, g.category, normalize="index") * 100)
print(lc.reindex(columns=[c for c in CATS if c in lc.columns])
      .round(1).to_string())

# ------------------------------------------------------------------ maps
print("\n[6] MAPS")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
pmin = min(g.rf_pred.min(), g.mlp_pred.min(), g.consensus.min())
pmax = max(g.rf_pred.max(), g.mlp_pred.max(), g.consensus.max())
lim = max(abs(pmin), abs(pmax))
for ax, (col, ttl) in zip(axes.flat[:3], [
        ("rf_pred", "RF predicted anomaly"),
        ("mlp_pred", "MLP predicted anomaly"),
        ("consensus", "Consensus (mean of both)")]):
    s = ax.scatter(g.lon, g.lat, c=g[col], s=7, cmap="RdYlGn",
                   vmin=-lim, vmax=lim)
    plt.colorbar(s, ax=ax, label="NDVI anomaly")
    ax.set_title(f"{ttl}\n(sd {g[col].std():.5f})")
ax = axes.flat[3]
s = ax.scatter(g.lon, g.lat, c=g.disagreement, s=7, cmap="magma_r")
plt.colorbar(s, ax=ax, label="|RF - MLP|")
ax.set_title(f"Disagreement\n(mean {g.disagreement.mean():.5f})")
for a in axes.flat:
    a.set_xlabel("longitude")
    a.set_ylabel("latitude")
    a.set_aspect("equal")
fig.suptitle("Sokoto Basin - RF vs MLP predicted NDVI anomaly, "
             "2020-2023 mean per cell", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "step6_predictions_disagreement.png"), dpi=140)
plt.close(fig)
print("  figures/step6_predictions_disagreement.png")

COL = {"Both agree stressed": "#c0392b", "Both agree healthy": "#27ae60",
       "Models disagree": "#f1c40f"}
fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
for ax, (ccol, ttl) in zip(axes, [
        ("category", "Sign-based (primary)"),
        ("category_deadband", f"With dead band (eps={EPS:.5f})")]):
    for c in CATS:
        s = g[g[ccol] == c]
        if len(s):
            ax.scatter(s.lon, s.lat, c=COL[c], s=8,
                       label=f"{c} ({len(s):,})")
    ax.legend(loc="lower left", fontsize=8, markerscale=2)
    ax.set_title(ttl)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect("equal")
fig.suptitle("Sokoto Basin - model consensus categories, 2020-2023", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "step6_consensus_categories.png"), dpi=140)
plt.close(fig)
print("  figures/step6_consensus_categories.png")

fig, ax = plt.subplots(figsize=(7.5, 7))
for c in CATS:
    s = g[g.category == c]
    ax.scatter(s.rf_pred, s.mlp_pred, c=COL[c], s=8, label=c, alpha=.7)
ax.axhline(0, color="k", lw=.8)
ax.axvline(0, color="k", lw=.8)
ax.set_xlabel("RF predicted anomaly (cell mean)")
ax.set_ylabel("MLP predicted anomaly (cell mean)")
ax.set_title(f"RF vs MLP per cell\ncell-level r = {g.rf_pred.corr(g.mlp_pred):+.3f}")
ax.legend(fontsize=8)
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "step6_rf_vs_mlp_scatter.png"), dpi=140)
plt.close(fig)
print("  figures/step6_rf_vs_mlp_scatter.png")

# ------------------------------------------------------------------ save
print("\n[7] SAVE")
COLS = ["point_id", "lat", "lon", "rf_pred", "mlp_pred", "consensus",
        "disagreement", "category", "category_deadband", "observed",
        "abs_err_consensus", "landcover_class", "landcover_name",
        "landcover_grouped", "n_months"]
out = g[COLS].sort_values("point_id")
out.to_csv(os.path.join(D, "consensus_map.csv"), index=False)
print(f"  consensus_map.csv ({len(out):,} rows, "
      f"{os.path.getsize(os.path.join(D,'consensus_map.csv'))/1e3:.0f} KB)")
print(f"    columns: {COLS}")
json.dump(dict(target=TARGET, n_cells=int(len(g)),
               monthly_rows=int(len(m)),
               monthly_corr_rf_mlp=float(m.rf_pred.corr(m.mlp_pred)),
               cell_corr_rf_mlp=float(g.rf_pred.corr(g.mlp_pred)),
               categories={c: int(vc[c]) for c in CATS},
               categories_pct={c: float(100 * vc[c] / len(g)) for c in CATS},
               deadband_eps=EPS,
               categories_deadband={str(k): int(v) for k, v in vc2.items()},
               disagreement_mean=float(g.disagreement.mean()),
               disagreement_max=float(g.disagreement.max()),
               skill=cmp.to_dict("records"),
               corr_disagreement_vs_error=float(
                   g.disagreement.corr(g.abs_err_consensus)),
               observed_sd=float(g.observed.std())),
          open(os.path.join(D, "consensus_summary.json"), "w"),
          indent=2, default=float)
print("  consensus_summary.json")
banner("STEP 6 COMPLETE")
