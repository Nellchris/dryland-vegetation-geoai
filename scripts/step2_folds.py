"""Step 2: Feature matrix check + spatial block cross-validation setup.

~25 km blocks are laid over the 5 km grid, then assigned to 5 folds by a
greedy graph colouring that forbids any two 8-connected (edge- OR corner-
adjacent) blocks from sharing a fold. Train and validation sets are therefore
spatially separated, so CV scores are not inflated by spatial autocorrelation.
"""
import json
import os

import numpy as np
import pandas as pd

D = r"C:\Users\USER\Documents\GeoAI_dryland"
pd.set_option("display.width", 220)

N_FOLDS = 5
BLOCK_KM = 25.0
HOLDOUT_YEARS = (2020, 2021, 2022, 2023)
SEED = 42


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


banner("STEP 2 - FEATURE SET + SPATIAL BLOCK CV SETUP")

cfg = json.load(open(os.path.join(D, "feature_config.json")))
TARGET, FEATURES = cfg["target"], cfg["features"]
df = pd.read_parquet(os.path.join(D, "feature_matrix.parquet"))
print(f"\nloaded feature_matrix.parquet: {df.shape[0]:,} rows x {df.shape[1]} cols")

# ------------------------------------------------------------- feature set
print(f"\n[1] FEATURE SET  ({len(FEATURES)} features)")
print(f"  target: {TARGET}   sd={df[TARGET].std():.5f}")
miss = [c for c in FEATURES + [TARGET] if c not in df.columns]
assert not miss, f"missing columns: {miss}"
print(f"\n  {'feature':<24}{'dtype':<9}{'min':>11}{'max':>11}{'sd':>10}"
      f"{'r with target':>15}")
for f in FEATURES:
    print(f"  {f:<24}{str(df[f].dtype):<9}{df[f].min():>11.4f}{df[f].max():>11.4f}"
          f"{df[f].std():>10.4f}{df[TARGET].corr(df[f]):>15.4f}")
print(f"\n  NaNs in features+target: "
      f"{int(df[FEATURES + [TARGET]].isna().sum().sum())}")
print("  NOTE: landcover_class is an ESA WorldCover code used numerically; RF")
print("  splits on it cleanly, the MLP standardises it like any other column")
print("  (so it acts as an ordinal proxy, not a true categorical).")

# -------------------------------------------------------- temporal holdout
print("\n[2] TEMPORAL HOLDOUT")
tr_n = int((df.split == "train").sum())
ho_n = int((df.split == "holdout").sum())
print(f"  train   2001-2019 : {tr_n:>9,} rows ({100*tr_n/len(df):.1f}%)  "
      f"mean target {df.loc[df.split=='train', TARGET].mean():+.5f}")
print(f"  holdout 2020-2023 : {ho_n:>9,} rows ({100*ho_n/len(df):.1f}%)  "
      f"mean target {df.loc[df.split=='holdout', TARGET].mean():+.5f}")
grid = df[["point_id", "lon", "lat"]].drop_duplicates("point_id").reset_index(drop=True)
print(f"  grid cells        : {len(grid):,} (every cell appears in both splits)")

# ---------------------------------------------------------- spatial blocks
print(f"\n[3] SPATIAL BLOCKS (~{BLOCK_KM:.0f} km)")
lat0 = float(grid.lat.mean())
dlat = BLOCK_KM / 110.574
dlon = BLOCK_KM / (111.320 * np.cos(np.radians(lat0)))
print(f"  mean latitude : {lat0:.4f} deg")
print(f"  block size    : {dlat:.4f} deg lat x {dlon:.4f} deg lon")
grid["bi"] = np.floor((grid.lat - grid.lat.min()) / dlat).astype(int)
grid["bj"] = np.floor((grid.lon - grid.lon.min()) / dlon).astype(int)
blocks = (grid.groupby(["bi", "bj"]).size().rename("n_cells").reset_index()
          .sort_values(["bi", "bj"]).reset_index(drop=True))
blocks["block_id"] = np.arange(len(blocks))
grid = grid.merge(blocks[["bi", "bj", "block_id"]], on=["bi", "bj"], how="left")
cell_deg = 0.044916
full_blk = int(round(dlat / cell_deg)) * int(round(dlon / cell_deg))
print(f"  blocks occupied: {len(blocks)}  (grid {blocks.bi.max()+1} x "
      f"{blocks.bj.max()+1})")
print(f"  cells/block    : min={blocks.n_cells.min()} "
      f"median={int(blocks.n_cells.median())} max={blocks.n_cells.max()} "
      f"(a full block holds {full_blk})")

# --------------------------------------------- fold assignment via colouring
print(f"\n[4] FOLD ASSIGNMENT ({N_FOLDS} folds, adjacent blocks must differ)")
coord2bid = {(r.bi, r.bj): r.block_id for r in blocks.itertuples()}
NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
nb = {b.block_id: [coord2bid[(b.bi + di, b.bj + dj)] for di, dj in NB8
                   if (b.bi + di, b.bj + dj) in coord2bid]
      for b in blocks.itertuples()}
deg = {k: len(v) for k, v in nb.items()}
print(f"  8-connected adjacency: mean degree {np.mean(list(deg.values())):.2f}, "
      f"max {max(deg.values())}")

# The 8-connected block lattice is a king graph. For every neighbour offset
# (di,dj) in {-1,0,1}^2 \ {(0,0)}, (2*di + dj) mod 5 != 0, so the periodic
# colouring fold = (2*bi + bj) mod 5 provably gives NO two adjacent blocks the
# same fold. A load-balancing greedy cannot guarantee this and gets stuck.
OFFS = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
assert all((2 * di + dj) % N_FOLDS != 0 for di, dj in OFFS), \
    "periodic colouring invalid for this fold count"
blocks["fold"] = ((2 * blocks.bi + blocks.bj) % N_FOLDS).astype(int)
fold_of = dict(zip(blocks.block_id, blocks.fold))
grid["fold"] = grid.block_id.map(fold_of)

viol = sum(1 for b, ns in nb.items() for n in ns if fold_of[b] == fold_of[n]) // 2
print(f"  scheme: fold = (2*bi + bj) mod {N_FOLDS}  (periodic king-graph colouring)")
print(f"  adjacent block pairs sharing fold: {viol}  "
      f"{'<- constraint satisfied' if viol == 0 else '<- VIOLATIONS'}")
assert viol == 0, f"{viol} adjacent block pairs share a fold"
fs = (blocks.groupby("fold")
      .agg(blocks=("block_id", "size"), cells=("n_cells", "sum")).reset_index())
fs["cells_pct"] = (100 * fs.cells / fs.cells.sum()).round(2)
print("\n  fold sizes:")
print(fs.to_string(index=False))

# --------------------------------------------------- minimum separation check
cen = grid.groupby("fold")[["lon", "lat"]].mean()
print("\n  fold centroids (should be near-identical - folds interleave):")
print(cen.round(3).to_string())

# ------------------------------------------------------------ attach to df
df = df.merge(grid[["point_id", "bi", "bj", "block_id", "fold"]],
              on="point_id", how="left")
assert df[["block_id", "fold"]].notna().all().all()
df["block_id"] = df.block_id.astype(np.int32)
df["fold"] = df.fold.astype(np.int8)

print("\n[5] CV FOLD ROW COUNTS (2001-2019 train rows only)")
tr = df[df.split == "train"]
rows = [dict(fold=f,
             val_rows=int((tr.fold == f).sum()),
             train_rows=int((tr.fold != f).sum()),
             val_cells=int(tr.loc[tr.fold == f, "point_id"].nunique()),
             val_pct=round(100 * (tr.fold == f).sum() / len(tr), 2))
        for f in range(N_FOLDS)]
print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------------------------- save
print("\n[6] SAVE")
grid[["point_id", "lon", "lat", "bi", "bj", "block_id", "fold"]] \
    .sort_values("point_id").to_csv(os.path.join(D, "spatial_folds.csv"), index=False)
print(f"  spatial_folds.csv    ({len(grid):,} cells -> block_id, fold)")
df.to_parquet(os.path.join(D, "model_matrix.parquet"), index=False)
print(f"  model_matrix.parquet ("
      f"{os.path.getsize(os.path.join(D,'model_matrix.parquet'))/1e6:.1f} MB)")
cfg.update(n_folds=N_FOLDS, block_km=BLOCK_KM, holdout_years=list(HOLDOUT_YEARS),
           seed=SEED, block_deg_lat=round(dlat, 6), block_deg_lon=round(dlon, 6),
           n_blocks=int(len(blocks)), train_rows=tr_n, holdout_rows=ho_n,
           adjacent_same_fold_pairs=int(viol),
           fold_cells={int(r.fold): int(r.cells) for r in fs.itertuples()})
json.dump(cfg, open(os.path.join(D, "cv_config.json"), "w"), indent=2)
print("  cv_config.json       (feature list + fold geometry)")

# --------------------------------------------------------------- fold map
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(D, "figures"), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    s0 = ax[0].scatter(grid.lon, grid.lat, c=grid.fold, s=7, cmap="tab10",
                       vmin=-0.5, vmax=9.5)
    ax[0].set_title(f"Spatial block CV folds\n({BLOCK_KM:.0f} km blocks, "
                    f"{N_FOLDS} folds, no adjacent block shares a fold)")
    plt.colorbar(s0, ax=ax[0], label="fold", ticks=range(N_FOLDS))
    ax[1].scatter(grid.lon, grid.lat, c=grid.block_id % 20, s=7, cmap="tab20")
    ax[1].set_title(f"{len(blocks)} spatial blocks (~{BLOCK_KM:.0f} km)")
    for a in ax:
        a.set_xlabel("longitude")
        a.set_ylabel("latitude")
        a.set_aspect("equal")
    fig.suptitle("Sokoto Basin - spatial block cross-validation design",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "figures", "step2_spatial_folds.png"), dpi=130)
    print("  figures/step2_spatial_folds.png")
except Exception as e:
    print(f"  (fold map skipped: {e})")

banner("STEP 2 COMPLETE")
