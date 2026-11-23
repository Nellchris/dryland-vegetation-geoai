"""Step 6b: add a bias-corrected (relative) category layer to consensus_map.csv.

Each model's predictions are centred on its own basin-mean, so a cell is
"stressed" when it is drier than the basin average *for that model*, rather
than when its absolute predicted anomaly is negative. This isolates the
relative spatial pattern, which is the part that carries real skill
(RF r = 0.287), from the absolute offset, which is dominated by model bias and
under-dispersion.

Note: aligning the predicted mean to the OBSERVED mean instead (adding
obs_mean - pred_mean) would push virtually every cell positive, because the
predicted spatial spread is only ~6% of observed. Centring on each model's own
mean is therefore the meaningful correction.
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
pd.set_option("display.width", 220)
CATS = ["Both agree stressed", "Both agree healthy", "Models disagree"]


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78, flush=True)


banner("STEP 6b - BIAS-CORRECTED (RELATIVE) CATEGORY LAYER")
g = pd.read_csv(os.path.join(D, "consensus_map.csv"))
print(f"\nloaded consensus_map.csv: {len(g):,} cells")

rf_mu, mlp_mu = g.rf_pred.mean(), g.mlp_pred.mean()
obs_mu = g.observed.mean()
print(f"  rf_pred  mean = {rf_mu:+.6f}   (bias vs observed {rf_mu-obs_mu:+.6f})")
print(f"  mlp_pred mean = {mlp_mu:+.6f}   (bias vs observed {mlp_mu-obs_mu:+.6f})")
print(f"  observed mean = {obs_mu:+.6f}")

g["rf_rel"] = g.rf_pred - rf_mu
g["mlp_rel"] = g.mlp_pred - mlp_mu
g["consensus_rel"] = (g.rf_rel + g.mlp_rel) / 2.0
g["observed_rel"] = g.observed - obs_mu

rs, ms = np.sign(g.rf_rel), np.sign(g.mlp_rel)
cat = np.full(len(g), "Models disagree", dtype=object)
cat[(rs < 0) & (ms < 0)] = "Both agree stressed"
cat[(rs > 0) & (ms > 0)] = "Both agree healthy"
g["category_biascorr"] = cat

vc = g.category_biascorr.value_counts().reindex(CATS).fillna(0).astype(int)
print("\n[1] BIAS-CORRECTED CATEGORIES (relative to each model's basin mean)")
for c in CATS:
    print(f"  {c:<22} {vc[c]:>5,} cells ({100*vc[c]/len(g):5.1f}%)")

print("\n[2] VALIDATION - does the relative label track relative observation?")
base_lo = float((g.observed_rel < 0).mean())
rows = []
for c in CATS:
    s = g[g.category_biascorr == c]
    if not len(s):
        continue
    p_lo = float((s.observed_rel < 0).mean())
    rows.append(dict(category=c, cells=len(s),
                     mean_observed_rel=float(s.observed_rel.mean()),
                     pct_obs_below_mean=100 * p_lo,
                     lift_vs_base=p_lo / base_lo))
v = pd.DataFrame(rows)
print(f"  base rate, observed below basin mean: {100*base_lo:.1f}%")
print(v.round(4).to_string(index=False))

# the same validation for the sign-based layer, for side-by-side reporting
print("\n[3] SIGN-BASED LAYER, same relative yardstick (for comparison)")
rows = []
for c in CATS:
    s = g[g.category == c]
    if not len(s):
        continue
    rows.append(dict(category=c, cells=len(s),
                     mean_observed_rel=float(s.observed_rel.mean()),
                     pct_obs_below_mean=100 * float((s.observed_rel < 0).mean())))
print(pd.DataFrame(rows).round(4).to_string(index=False))

print("\n[4] SEPARATION OF OBSERVED ANOMALY BETWEEN THE AGREED CLASSES")
for lbl, col in (("sign-based", "category"), ("bias-corrected", "category_biascorr")):
    a = g.loc[g[col] == "Both agree stressed", "observed"]
    b = g.loc[g[col] == "Both agree healthy", "observed"]
    if len(a) and len(b):
        sp = b.mean() - a.mean()
        pooled = np.sqrt((a.var() * (len(a) - 1) + b.var() * (len(b) - 1))
                         / (len(a) + len(b) - 2))
        print(f"  {lbl:<15}: mean observed  stressed={a.mean():+.5f}  "
              f"healthy={b.mean():+.5f}  gap={sp:+.5f}  "
              f"Cohen d={sp/pooled:+.3f}")

print("\n[5] LAND COVER BREAKDOWN (bias-corrected)")
lc = pd.crosstab(g.landcover_grouped, g.category_biascorr, normalize="index") * 100
print(lc.reindex(columns=[c for c in CATS if c in lc.columns]).round(1).to_string())

# ------------------------------------------------------------------ map
COL = {"Both agree stressed": "#c0392b", "Both agree healthy": "#27ae60",
       "Models disagree": "#f1c40f"}
fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
for ax, (col, ttl) in zip(axes, [
        ("category", "Sign-based (absolute); stressed label 42.3% precise"),
        ("category_biascorr", "Bias-corrected (relative to basin mean) - DEFAULT")]):
    for c in CATS:
        s = g[g[col] == c]
        if len(s):
            ax.scatter(s.lon, s.lat, c=COL[c], s=8, label=f"{c} ({len(s):,})")
    ax.legend(loc="lower left", fontsize=8, markerscale=2)
    ax.set_title(ttl, fontsize=11)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect("equal")
fig.suptitle("Sokoto Basin - consensus categories, absolute vs relative, 2020-2023",
             fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "step6b_category_layers.png"), dpi=140)
plt.close(fig)
print("\n[6] SAVE")
print("  figures/step6b_category_layers.png")

cols = list(g.columns)
for c in ("rf_rel", "mlp_rel", "consensus_rel", "observed_rel",
          "category_biascorr"):
    cols.remove(c)
ins = cols.index("category_deadband") + 1
cols = (cols[:ins] + ["category_biascorr", "rf_rel", "mlp_rel",
                      "consensus_rel", "observed_rel"] + cols[ins:])
g[cols].to_csv(os.path.join(D, "consensus_map.csv"), index=False)
print(f"  consensus_map.csv updated ({len(g):,} rows, {len(cols)} cols)")

p = os.path.join(D, "consensus_summary.json")
s = json.load(open(p))
s["biascorr"] = dict(
    rf_centre=float(rf_mu), mlp_centre=float(mlp_mu),
    observed_mean=float(obs_mu),
    categories={c: int(vc[c]) for c in CATS},
    categories_pct={c: float(100 * vc[c] / len(g)) for c in CATS},
    base_rate_below_mean=base_lo,
    validation=v.to_dict("records"))
json.dump(s, open(p, "w"), indent=2, default=float)
print("  consensus_summary.json updated")
banner("STEP 6b COMPLETE")
