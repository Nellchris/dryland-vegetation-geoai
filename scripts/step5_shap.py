"""Step 5: SHAP explainability.

Run as three modes so the two explainers can run concurrently:
    SHAP_MODE=tree    python step5_shap.py   # TreeSHAP, RF,  ~3.5 h
    SHAP_MODE=kernel  python step5_shap.py   # KernelSHAP, MLP, ~15 min
    SHAP_MODE=compare python step5_shap.py   # merge + comparison table

TreeSHAP uses a cell-stratified subsample (equal months per grid cell) so the
per-cell spatial |SHAP| maps cover every cell in the basin. Chunks are
checkpointed to disk and reloaded on restart.
"""
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn

D = r"C:\Users\USER\Documents\GeoAI_dryland"
FIG = os.path.join(D, "figures")
os.makedirs(FIG, exist_ok=True)
pd.set_option("display.width", 220)

MODE = os.environ.get("SHAP_MODE", "tree").lower()
TREE_N = int(os.environ.get("SHAP_TREE_N", "50000"))
N_KERNEL = int(os.environ.get("SHAP_KERNEL_N", "2000"))
BG_K = int(os.environ.get("SHAP_BG_K", "100"))
SEED = 42
# leave headroom when both explainers run side by side on 4 cores
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78, flush=True)


cfg = json.load(open(os.path.join(D, "cv_config.json")))
TARGET, FEATURES = cfg["target"], cfg["features"]
df = pd.read_parquet(os.path.join(D, "model_matrix.parquet"))
ho = df[df.split == "holdout"].reset_index(drop=True)
Xho = ho[FEATURES].to_numpy(np.float32)
META = ["point_id", "lon", "lat", "year", "month", "landcover_name"]
print(f"MODE={MODE} | holdout {len(ho):,} rows x {len(FEATURES)} features "
      f"| shap {shap.__version__}", flush=True)


# ------------------------------------------------------------- shared bits
def save_shap(sv, idx, name):
    out = ho.loc[idx, META].reset_index(drop=True)
    for j, c in enumerate(FEATURES):
        out[f"shap_{c}"] = sv[:, j]
    p = os.path.join(D, f"shap_{name}.parquet")
    out.to_parquet(p, index=False)
    print(f"  shap_{name}.parquet ({os.path.getsize(p)/1e6:.1f} MB, {len(out):,} rows)")
    return out


def global_importance(sv, name):
    imp = pd.DataFrame({"feature": FEATURES,
                        "mean_abs_shap": np.abs(sv).mean(0),
                        "mean_shap": sv.mean(0),
                        "sd_shap": sv.std(0)}) \
        .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["pct_of_total"] = 100 * imp.mean_abs_shap / imp.mean_abs_shap.sum()
    imp.to_csv(os.path.join(D, f"shap_importance_{name}.csv"), index=False)
    print(f"\n  GLOBAL IMPORTANCE ({name}), mean |SHAP|:")
    print(imp.round(6).to_string(index=False))
    return imp


def plot_bar(imp, name, title):
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.feature, top.mean_abs_shap, color="#2b7a4b")
    ax.set_xlabel("mean |SHAP|  (NDVI anomaly units)")
    ax.set_title(f"{title}\nglobal feature importance (top 15)")
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"shap_{name}_bar.png"), dpi=140)
    plt.close(fig)
    print(f"  figures/shap_{name}_bar.png")


def plot_beeswarm(sv, X, name, title):
    plt.figure()
    shap.plots.beeswarm(shap.Explanation(values=sv, data=X,
                                         feature_names=FEATURES),
                        max_display=15, show=False)
    plt.title(f"{title}\nSHAP beeswarm (top 15)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"shap_{name}_beeswarm.png"), dpi=140,
                bbox_inches="tight")
    plt.close("all")
    print(f"  figures/shap_{name}_beeswarm.png")


def plot_spatial(tab, imp, name, title):
    top3 = imp.head(3).feature.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    for ax, feat in zip(axes, top3):
        g = (tab.assign(_a=tab[f"shap_{feat}"].abs())
             .groupby("point_id").agg(lon=("lon", "first"),
                                      lat=("lat", "first"), v=("_a", "mean")))
        s = ax.scatter(g.lon, g.lat, c=g.v, s=7, cmap="viridis")
        plt.colorbar(s, ax=ax, label="mean |SHAP|")
        ax.set_title(feat)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_aspect("equal")
    fig.suptitle(f"{title} - spatial mean |SHAP|, top 3 features (2020-2023)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"shap_{name}_spatial_top3.png"), dpi=140)
    plt.close(fig)
    print(f"  figures/shap_{name}_spatial_top3.png  (top 3: {top3})")
    return top3


def build_mlp(n):
    return nn.Sequential(
        nn.Linear(n, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(32, 1))


# ============================================================ TreeSHAP (RF)
if MODE == "tree":
    ncell = ho.point_id.nunique()
    per_cell = max(1, int(round(TREE_N / ncell)))
    rf_idx = np.sort(ho.sample(frac=1.0, random_state=SEED)
                     .groupby("point_id").head(per_cell).index.to_numpy())
    banner(f"TreeSHAP - RF, CELL-STRATIFIED {len(rf_idx):,}-ROW SUBSAMPLE")
    print(f"  {per_cell} months per cell x {ncell:,} cells = {len(rf_idx):,} rows "
          f"({100*len(rf_idx)/len(ho):.1f}% of holdout)")
    print(f"  cells covered: {ho.loc[rf_idx].point_id.nunique():,}/{ncell:,}")
    print(f"  by year: {ho.loc[rf_idx].year.value_counts().sort_index().to_dict()}")

    Xrf = Xho[rf_idx]
    rf = joblib.load(os.path.join(D, "rf_model.joblib"))
    ex = shap.TreeExplainer(rf)
    CKPT = os.path.join(D, "_shap_rf_chunks")
    os.makedirs(CKPT, exist_ok=True)
    CH, parts, t0, ndone = 5000, [], time.time(), 0
    for ci, i in enumerate(range(0, len(Xrf), CH)):
        cp = os.path.join(CKPT, f"c_{len(Xrf)}_{ci:05d}.npy")
        if os.path.exists(cp):
            parts.append(np.load(cp))
            print(f"    chunk {ci} restored from checkpoint", flush=True)
            continue
        v = ex.shap_values(Xrf[i:i + CH], check_additivity=False)
        np.save(cp, v)
        parts.append(v)
        ndone += len(v)
        el = time.time() - t0
        nd = min(i + CH, len(Xrf))
        print(f"    {nd:>7,}/{len(Xrf):,}  {el/60:6.1f} min elapsed, "
              f"~{(el/ndone)*(len(Xrf)-nd)/60:6.1f} min left", flush=True)
    sv = np.concatenate(parts, 0)
    print(f"  done, shape {sv.shape}")

    base = float(np.atleast_1d(ex.expected_value)[0])
    rfp = pd.read_parquet(os.path.join(D, "rf_predictions_holdout.parquet")).rf_pred
    err = np.abs(sv.sum(1) + base - rfp.to_numpy()[rf_idx]).max()
    print(f"  additivity: max|sum(SHAP)+base - pred| = {err:.3e} (base={base:+.6f})")

    imp = global_importance(sv, "rf")
    tab = save_shap(sv, rf_idx, "rf")
    plot_bar(imp, "rf", "Random Forest (TreeSHAP)")
    bn = min(20000, len(sv))
    b = np.random.default_rng(SEED).choice(len(sv), bn, replace=False)
    plot_beeswarm(sv[b], Xrf[b], "rf", f"Random Forest (TreeSHAP, {bn:,}-pt view)")
    top3 = plot_spatial(tab, imp, "rf", "Random Forest")
    json.dump(dict(base_value=base, n=int(len(sv)), per_cell=per_cell,
                   cells=int(ho.loc[rf_idx].point_id.nunique()),
                   additivity_max_err=float(err), top3=top3),
              open(os.path.join(D, "shap_rf_meta.json"), "w"), indent=2, default=float)
    banner("TreeSHAP COMPLETE")

# ========================================================== KernelSHAP (MLP)
elif MODE == "kernel":
    banner(f"KernelSHAP - MLP, {N_KERNEL}-POINT SUBSAMPLE")
    blob = torch.load(os.path.join(D, "mlp_model.pt"), weights_only=False)
    scaler = joblib.load(os.path.join(D, "mlp_scaler.joblib"))
    mlp = build_mlp(len(FEATURES))
    mlp.load_state_dict(blob["state_dict"])
    mlp.eval()

    @torch.no_grad()
    def fpred(X):
        return mlp(torch.from_numpy(np.asarray(X, np.float32))).squeeze(1).numpy()

    Xs = scaler.transform(Xho).astype(np.float32)
    rng = np.random.default_rng(SEED)
    idx = np.sort(np.concatenate([
        rng.choice(np.where(ho.year.to_numpy() == y)[0],
                   int(round(N_KERNEL * (ho.year == y).mean())), replace=False)
        for y in sorted(ho.year.unique())]))
    print(f"  {len(idx):,} rows, by year: "
          f"{ho.loc[idx].year.value_counts().sort_index().to_dict()}")
    print(f"  background: k-means k={BG_K} over 20,000 scaled holdout rows",
          flush=True)
    bg = shap.kmeans(Xs[rng.choice(len(Xs), 20000, replace=False)], BG_K)
    ke = shap.KernelExplainer(fpred, bg)
    CKPT = os.path.join(D, "_shap_mlp_chunks")
    os.makedirs(CKPT, exist_ok=True)
    CH, parts, t0, ndone = 250, [], time.time(), 0
    for ci, i in enumerate(range(0, len(idx), CH)):
        cp = os.path.join(CKPT, f"c_{len(idx)}_{ci:05d}.npy")
        if os.path.exists(cp):
            parts.append(np.load(cp))
            continue
        v = ke.shap_values(Xs[idx[i:i + CH]], silent=True)
        np.save(cp, v)
        parts.append(v)
        ndone += len(v)
        el = time.time() - t0
        nd = min(i + CH, len(idx))
        print(f"    {nd:>5}/{len(idx)}  {el/60:5.1f} min elapsed, "
              f"~{(el/ndone)*(len(idx)-nd)/60:5.1f} min left", flush=True)
    sv = np.concatenate(parts, 0)
    base = float(np.atleast_1d(ke.expected_value)[0])
    err = np.abs(sv.sum(1) + base - fpred(Xs[idx])).max()
    print(f"  done, shape {sv.shape}")
    print(f"  additivity: max|sum(SHAP)+base - pred| = {err:.3e} (base={base:+.6f})")

    imp = global_importance(sv, "mlp")
    tab = save_shap(sv, idx, "mlp")
    plot_bar(imp, "mlp", f"MLP (KernelSHAP, {len(idx):,} pts)")
    plot_beeswarm(sv, Xho[idx], "mlp", f"MLP (KernelSHAP, {len(idx):,} pts)")
    top3 = plot_spatial(tab, imp, "mlp", "MLP")
    json.dump(dict(base_value=base, n=int(len(sv)), background_k=BG_K,
                   additivity_max_err=float(err), top3=top3),
              open(os.path.join(D, "shap_mlp_meta.json"), "w"), indent=2, default=float)
    banner("KernelSHAP COMPLETE")

# ============================================================== comparison
elif MODE == "compare":
    banner("RF vs MLP - SHAP COMPARISON")
    a = pd.read_csv(os.path.join(D, "shap_importance_rf.csv"))
    b = pd.read_csv(os.path.join(D, "shap_importance_mlp.csv"))
    ma = json.load(open(os.path.join(D, "shap_rf_meta.json")))
    mb = json.load(open(os.path.join(D, "shap_mlp_meta.json")))
    c = (a[["feature", "mean_abs_shap", "pct_of_total"]]
         .rename(columns={"mean_abs_shap": "rf_abs", "pct_of_total": "rf_pct"})
         .merge(b[["feature", "mean_abs_shap", "pct_of_total"]]
                .rename(columns={"mean_abs_shap": "mlp_abs",
                                 "pct_of_total": "mlp_pct"}), on="feature"))
    c["rf_rank"] = c.rf_abs.rank(ascending=False).astype(int)
    c["mlp_rank"] = c.mlp_abs.rank(ascending=False).astype(int)
    c["rank_diff"] = (c.rf_rank - c.mlp_rank).abs()
    c = c.sort_values("rf_rank").reset_index(drop=True)
    print(c.round(6).to_string(index=False))
    rho = c.rf_rank.corr(c.mlp_rank, method="spearman")
    print(f"\n  Spearman rank corr of feature importance: {rho:+.4f}")
    print(f"  RF  top3: {ma['top3']}")
    print(f"  MLP top3: {mb['top3']}")
    print(f"  shared in top 3: {len(set(ma['top3']) & set(mb['top3']))}/3")
    c.to_csv(os.path.join(D, "shap_importance_comparison.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(c.rf_pct, c.mlp_pct, s=45, color="#2b7a4b")
    lim = max(c.rf_pct.max(), c.mlp_pct.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=.5)
    for r in c.itertuples():
        ax.annotate(r.feature, (r.rf_pct, r.mlp_pct), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("RF  share of total mean |SHAP| (%)")
    ax.set_ylabel("MLP share of total mean |SHAP| (%)")
    ax.set_title(f"Feature importance agreement, RF vs MLP\n"
                 f"Spearman rank corr = {rho:+.3f}")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "shap_rf_vs_mlp.png"), dpi=140)
    plt.close(fig)
    print("  figures/shap_rf_vs_mlp.png")
    json.dump(dict(rf=ma, mlp=mb, spearman_rank_corr=float(rho),
                   comparison=c.to_dict("records")),
              open(os.path.join(D, "shap_summary.json"), "w"), indent=2, default=float)
    print("  shap_summary.json")
    banner("STEP 5 COMPLETE")
else:
    raise SystemExit(f"unknown SHAP_MODE={MODE!r}")
