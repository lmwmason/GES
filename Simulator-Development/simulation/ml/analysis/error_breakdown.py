#!/usr/bin/env python3
"""
Breaks the trained model's test-set error down by *where* it happens,
instead of reporting a single aggregate RMSE/R2:

  1. Error vs. clearance (rangefinder altitude) -- ground effect itself is
     strongest close to the ground, so this checks whether the model's
     advantage over the analytical baseline holds up right where it
     matters most, or only far from the ground where GE is weak anyway.
  2. Error vs. "terrain irregularity", proxied by how wrong the analytical
     (flat-ground) estimate already is at that row -- |ge_ff_analytical -
     truth|. A large proxy value means the local terrain is doing
     something a flat-ground formula can't see (slope/curvature/bump). If
     the trained model's error grows *less* than the baseline's in that
     regime, that's direct evidence it's picking up on non-flat-terrain
     cues from the sensor stream, not just re-deriving the same formula.

Usage:
    python3 ml/analysis/error_breakdown.py --csv dataset/dataset_all.csv
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import BASELINE_COL, add_block_id, block_split, build_pointwise_xy, lead_s_to_rows, load_dataset

MODEL_COLOR = "#3b82f6"
BASELINE_COLOR = "#c2410c"


def _binned_mae(x: np.ndarray, err_abs: np.ndarray, bins: np.ndarray, min_count: int = 30):
    idx = np.digitize(x, bins)
    centers, maes, counts = [], [], []
    for b in range(1, len(bins)):
        mask = idx == b
        if mask.sum() < min_count:
            continue
        centers.append((bins[b - 1] + bins[b]) / 2)
        maes.append(np.mean(err_abs[mask]))
        counts.append(int(mask.sum()))
    return np.array(centers), np.array(maes), np.array(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset/dataset_all.csv")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--block-s", type=float, default=10.0)
    ap.add_argument("--lead-s", type=float, default=0.1)
    ap.add_argument("--out-dir", default="ml/results/error_analysis")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(Path(args.csv))
    df = add_block_id(df, args.block_s)
    lead_rows = lead_s_to_rows(df, args.lead_s)
    train_df, test_df = block_split(df, args.test_frac, args.block_s, seed=0)

    X_train, y_train, _, _ = build_pointwise_xy(train_df, lead_rows)
    X_test, y_test, block_test, extra_test = build_pointwise_xy(test_df, lead_rows, extra_cols=[BASELINE_COL])
    baseline_pred = extra_test[BASELINE_COL]

    print("[error] fitting Random Forest...")
    model = RandomForestRegressor(n_estimators=200, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=0)
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)

    # clearance for each test row: pull rangefinder_alt from test_df aligned
    # to the same (block, row-order) construction build_pointwise_xy used.
    clearance = build_pointwise_xy(test_df, lead_rows, extra_cols=["rangefinder_alt"])[3]["rangefinder_alt"]

    model_err = np.abs(model_pred - y_test)
    baseline_err = np.abs(baseline_pred - y_test)
    terrain_irregularity = np.abs(baseline_pred - y_test)  # proxy: how wrong flat-ground already is

    print(f"[error] overall  model MAE={model_err.mean():.5f}  baseline MAE={baseline_err.mean():.5f}")

    summary = pd.DataFrame({
        "clearance": clearance, "model_abs_err": model_err, "baseline_abs_err": baseline_err,
    })
    summary.to_csv(out_dir / "per_row_errors.csv", index=False)

    _plot_by_clearance(clearance, model_err, baseline_err, args.lead_s, out_dir / "error_by_clearance.png")
    _plot_error_reduction(clearance, model_err, baseline_err, out_dir / "error_reduction_by_clearance.png")


def _plot_by_clearance(clearance, model_err, baseline_err, lead_s, path: Path):
    bins = np.linspace(0, min(np.percentile(clearance, 99), 1.2), 16)
    c_m, mae_m, n_m = _binned_mae(clearance, model_err, bins)
    c_b, mae_b, n_b = _binned_mae(clearance, baseline_err, bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    width = (bins[1] - bins[0]) * 0.4
    ax1.bar(c_b - width / 2, mae_b, width=width, color=BASELINE_COLOR, label="Analytical (baseline)")
    ax1.bar(c_m + width / 2, mae_m, width=width, color=MODEL_COLOR, label="Random Forest")
    ax1.set_xlabel("Rangefinder clearance [m]")
    ax1.set_ylabel("Mean absolute error")
    ax1.set_title(f"Error by clearance ({lead_s * 1000:.0f}ms-ahead forecast)")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(fontsize=8)

    ax2.bar(c_m, n_m, width=(bins[1] - bins[0]) * 0.85, color="#8892a6")
    ax2.set_xlabel("Rangefinder clearance [m]")
    ax2.set_ylabel("Test rows in bin")
    ax2.set_title("Test-set coverage by clearance")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[error] wrote {path}")


def _plot_error_reduction(clearance, model_err, baseline_err, path: Path):
    bins = np.linspace(0, min(np.percentile(clearance, 99), 1.2), 16)
    c_m, mae_m, n_m = _binned_mae(clearance, model_err, bins)
    c_b, mae_b, _ = _binned_mae(clearance, baseline_err, bins)
    reduction_pct = 100.0 * (mae_b - mae_m) / np.maximum(mae_b, 1e-9)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["#059669" if v >= 0 else "#dc2626" for v in reduction_pct]
    ax.bar(c_m, reduction_pct, width=(bins[1] - bins[0]) * 0.85, color=colors)
    ax.axhline(0.0, color="#8892a6", linewidth=1)
    ax.set_xlabel("Rangefinder clearance [m]")
    ax.set_ylabel("MAE reduction vs. analytical baseline [%]")
    ax.set_title("Where the learned model helps most\n(green = model beats analytical baseline)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[error] wrote {path}")


if __name__ == "__main__":
    main()
