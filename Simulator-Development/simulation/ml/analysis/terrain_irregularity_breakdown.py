#!/usr/bin/env python3
"""
Answers a sharper question than error_breakdown.py's clearance bins: does
ML beat the analytical (flat-ground) baseline *because* it's better on
irregular (sloped/curved) terrain where the flat-ground assumption breaks
down -- or does it win overall while still losing specifically where that
assumption breaks down (which would be the more interesting, coutner-
intuitive finding)?

Terrain slope and curvature are ground-truth-only quantities (the firmware
is never allowed to see them, see physics/aero_ground_effect.c) -- they
have to be reconstructed here from (seed, x, y) using viz/terrain_ref.py's
bit-exact Python port of physics/terrain.c's height field, with the same
finite-difference eps physics/aero_ground_effect.c itself uses. This is
independent of any model's own prediction error, unlike bucketing by
|analytical - truth| (which would be circular -- analytical's own error
defining the bins it's then scored against).

Usage:
    python3 ml/analysis/terrain_irregularity_breakdown.py --csv dataset/lowalt_dataset/dataset_all.csv
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "viz"))
from data import BASELINE_COL, FEATURE_COLS, add_block_id, lead_s_to_rows, load_dataset
from models_torch import MLPNet, predict_torch_model
import joblib
from terrain_ref import generate_features, _feature_height_grid

MODEL_COLOR = "#3b82f6"
BASELINE_COLOR = "#c2410c"

GRAD_EPS = 0.02   # matches physics/terrain.c terrain_gradient()
CURV_EPS = 0.05   # matches physics/terrain.c terrain_curvature()


def terrain_height_at(x: np.ndarray, y: np.ndarray, features) -> np.ndarray:
    H = np.zeros_like(x, dtype=float)
    for f in features:
        H += _feature_height_grid(f, x, y)
    return H


def slope_and_curvature(x: np.ndarray, y: np.ndarray, features):
    """Vectorized finite-difference slope magnitude [rad] and Laplacian
    curvature [1/m] at each (x, y), mirroring physics/terrain.c exactly."""
    h_xp = terrain_height_at(x + GRAD_EPS, y, features)
    h_xm = terrain_height_at(x - GRAD_EPS, y, features)
    h_yp = terrain_height_at(x, y + GRAD_EPS, features)
    h_ym = terrain_height_at(x, y - GRAD_EPS, features)
    dhdx = (h_xp - h_xm) / (2.0 * GRAD_EPS)
    dhdy = (h_yp - h_ym) / (2.0 * GRAD_EPS)
    slope = np.arctan(np.sqrt(dhdx ** 2 + dhdy ** 2))

    h0 = terrain_height_at(x, y, features)
    h_xp2 = terrain_height_at(x + CURV_EPS, y, features)
    h_xm2 = terrain_height_at(x - CURV_EPS, y, features)
    h_yp2 = terrain_height_at(x, y + CURV_EPS, features)
    h_ym2 = terrain_height_at(x, y - CURV_EPS, features)
    d2x = (h_xp2 - 2.0 * h0 + h_xm2) / (CURV_EPS ** 2)
    d2y = (h_yp2 - 2.0 * h0 + h_ym2) / (CURV_EPS ** 2)
    curvature = d2x + d2y
    return slope, curvature


def build_features_with_lag(df: pd.DataFrame, lead_rows: int):
    """Same construction as the deployed firmware: thrust_cmd_norm is the
    *previous* row's value (see controller.c), and the label is lead_rows
    ahead, never crossing a block boundary."""
    feat_cols = FEATURE_COLS[:-1]
    X_list, y_list, an_list, slope_list, curv_list = [], [], [], [], []
    for block_id, g in df.groupby("block_id", sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        if n <= lead_rows + 1:
            continue
        thrust_prev = g["thrust_cmd_norm"].shift(1).to_numpy()
        feats = g[feat_cols].to_numpy(dtype=np.float64)
        for i in range(1, n - lead_rows):
            X_list.append(np.append(feats[i], thrust_prev[i]))
            y_list.append(g["ge_factor_truth_mean"].iloc[i + lead_rows])
            an_list.append(g[BASELINE_COL].iloc[i])
            slope_list.append(g["slope"].iloc[i])
            curv_list.append(g["curvature"].iloc[i])
    return (np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32),
            np.array(an_list, dtype=np.float32), np.array(slope_list), np.array(curv_list))


def _binned_mae(x: np.ndarray, model_err: np.ndarray, an_err: np.ndarray, n_bins: int = 5, min_count: int = 200):
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    idx = np.digitize(x, edges[1:-1])
    centers, model_mae, an_mae, counts, lo_edges, hi_edges = [], [], [], [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() < min_count:
            continue
        centers.append(x[mask].mean())
        model_mae.append(model_err[mask].mean())
        an_mae.append(an_err[mask].mean())
        counts.append(int(mask.sum()))
        lo_edges.append(edges[b])
        hi_edges.append(edges[b + 1])
    return (np.array(centers), np.array(model_mae), np.array(an_mae), np.array(counts),
            np.array(lo_edges), np.array(hi_edges))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset/lowalt_dataset/dataset_all.csv")
    ap.add_argument("--joblib", default="ml/results/lowalt_mlp/mlp_only.joblib")
    ap.add_argument("--block-s", type=float, default=10.0)
    ap.add_argument("--lead-s", type=float, default=0.1)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--out-dir", default="ml/results/terrain_irregularity_breakdown")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(Path(args.csv))
    df = add_block_id(df, args.block_s)
    lead_rows = lead_s_to_rows(df, args.lead_s)

    print("[terrain] computing slope/curvature per row (this regenerates terrain features per seed)...")
    df["slope"] = np.nan
    df["curvature"] = np.nan
    for seed, g in df.groupby("seed"):
        features = generate_features(int(seed))
        x = g["x"].to_numpy(dtype=np.float64)
        y = g["y"].to_numpy(dtype=np.float64)
        slope, curv = slope_and_curvature(x, y, features)
        df.loc[g.index, "slope"] = slope
        df.loc[g.index, "curvature"] = curv
    print(f"[terrain] slope: mean={df['slope'].mean():.4f} rad, 95th pct={df['slope'].quantile(0.95):.4f} rad")
    print(f"[terrain] |curvature|: mean={df['curvature'].abs().mean():.4f} 1/m, 95th pct={df['curvature'].abs().quantile(0.95):.4f} 1/m")

    X, y, an_pred, slope, curvature = build_features_with_lag(df, lead_rows)
    print(f"[terrain] built {len(y)} pointwise (feature, label) rows")

    bundle = joblib.load(args.joblib)
    scaler = bundle["mlp_scaler"]
    model = MLPNet(n_features=len(FEATURE_COLS))
    model.load_state_dict(bundle["mlp_model_state"])
    Xs = scaler.transform(X).astype(np.float32)
    ml_pred = predict_torch_model(model, Xs, device="cpu")

    model_err = np.abs(ml_pred - y)
    an_err = np.abs(an_pred - y)
    print(f"[terrain] overall MAE: ML={model_err.mean():.5f}  analytical={an_err.mean():.5f}")

    _plot(slope, model_err, an_err, args.n_bins, "Terrain slope [rad]",
          "Model error vs. local terrain slope\n(does ML's advantage grow on steeper ground, as the flat-ground theory predicts?)",
          out_dir / "error_by_slope.png")
    _plot(np.abs(curvature), model_err, an_err, args.n_bins, "|Terrain curvature| [1/m]",
          "Model error vs. local terrain curvature\n(bumps/dips the flat-ground formula has no notion of at all)",
          out_dir / "error_by_curvature.png")


def _plot(x, model_err, an_err, n_bins, xlabel, title, path):
    centers, model_mae, an_mae, counts, lo, hi = _binned_mae(x, model_err, an_err, n_bins)
    reduction_pct = 100.0 * (an_mae - model_mae) / np.maximum(an_mae, 1e-9)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax1, ax2, ax3 = axes

    width = np.median(hi - lo) * 0.35
    ax1.bar(centers - width / 2, an_mae, width=width, color=BASELINE_COLOR, label="Analytical (baseline)")
    ax1.bar(centers + width / 2, model_mae, width=width, color=MODEL_COLOR, label="ML (MLP)")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Mean absolute error")
    ax1.set_title("Error by bin")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(fontsize=8)

    colors = ["#059669" if v >= 0 else "#dc2626" for v in reduction_pct]
    ax2.bar(centers, reduction_pct, width=np.median(hi - lo) * 0.85, color=colors)
    ax2.axhline(0.0, color="#8892a6", linewidth=1)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("ML's MAE reduction vs. analytical [%]")
    ax2.set_title("Where ML wins (green) vs. loses (red)")
    ax2.spines[["top", "right"]].set_visible(False)

    ax3.bar(centers, counts, width=np.median(hi - lo) * 0.85, color="#8892a6")
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel("Rows in bin")
    ax3.set_title("Bin coverage")
    ax3.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=150)
    print(f"[terrain] wrote {path}")
    for c, m, a, r, n in zip(centers, model_mae, an_mae, reduction_pct, counts):
        print(f"[terrain]   bin center={c:.4f}  n={n:6d}  ML_MAE={m:.5f}  analytical_MAE={a:.5f}  reduction={r:+.1f}%")


if __name__ == "__main__":
    main()
