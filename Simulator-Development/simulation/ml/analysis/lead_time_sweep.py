#!/usr/bin/env python3
"""
Sweeps the forecast horizon (--lead-s in train_all_models.py) across a range
of values and tracks how each model's test error grows with how far ahead
it has to predict the ground-effect factor.

This is the key robustness question for a real feedforward controller: the
100ms single-horizon comparison in train_all_models.py answers "how good is
the model at the horizon we picked", but says nothing about how gracefully
it degrades if inference/control latency turns out to be longer (or how
much is left on the table if it turns out to be shorter). Only classical
models are swept here (Analytical baseline, Linear Regression, Random
Forest, Gradient Boosting) -- the PyTorch sequence models are compared at a
single horizon in train_all_models.py, since training all of them at every
horizon would multiply an already CPU-bound run by len(horizons).

Usage:
    python3 ml/analysis/lead_time_sweep.py --csv dataset/dataset_all.csv
    python3 ml/analysis/lead_time_sweep.py --csv dataset/dataset_all.csv \
        --lead-s-list 0,0.05,0.1,0.2,0.3,0.5
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import BASELINE_COL, add_block_id, block_split, build_pointwise_xy, lead_s_to_rows, load_dataset

NEUTRAL = "#8892a6"
BASELINE_COLOR = "#c2410c"
MODEL_COLORS = {
    "Linear Regression": "#a855f7",
    "Random Forest": "#3b82f6",
    "Gradient Boosting": "#059669",
}


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def run_one_horizon(df, lead_s, block_s, test_frac):
    df = add_block_id(df, block_s)
    lead_rows = lead_s_to_rows(df, lead_s)
    train_df, test_df = block_split(df, test_frac, block_s, seed=0)

    rows = []

    X_train, y_train, _, extra_train = build_pointwise_xy(train_df, lead_rows)
    X_test, y_test, _, extra_test = build_pointwise_xy(test_df, lead_rows, extra_cols=[BASELINE_COL])

    baseline_pred = extra_test[BASELINE_COL]
    rows.append({"lead_s": lead_s, "model": "Analytical (baseline)", "train_s": 0.0,
                 **evaluate(y_test, baseline_pred)})

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=0),
        "Gradient Boosting": HistGradientBoostingRegressor(max_depth=6, max_iter=250, random_state=0),
    }
    for name, model in models.items():
        if name == "Linear Regression":
            scaler = StandardScaler().fit(X_train)
            Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
        else:
            Xtr, Xte = X_train, X_test
        t0 = time.time()
        model.fit(Xtr, y_train)
        train_s = time.time() - t0
        pred = model.predict(Xte)
        rows.append({"lead_s": lead_s, "model": name, "train_s": train_s, **evaluate(y_test, pred)})

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset/dataset_all.csv")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--block-s", type=float, default=10.0)
    ap.add_argument("--lead-s-list", default="0,0.05,0.1,0.15,0.2,0.3,0.5")
    ap.add_argument("--out-dir", default="ml/results/lead_time_sweep")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lead_values = [float(v) for v in args.lead_s_list.split(",")]

    df = load_dataset(Path(args.csv))

    all_rows = []
    for lead_s in lead_values:
        print(f"\n[sweep] === lead_s = {lead_s:.3f}s ===")
        rows = run_one_horizon(df, lead_s, args.block_s, args.test_frac)
        for r in rows:
            print(f"[sweep]   {r['model']:<22s} RMSE={r['RMSE']:.5f}  R2={r['R2']:.4f}")
        all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(out_dir / "lead_time_sweep.csv", index=False)
    print(f"\n[sweep] wrote {out_dir / 'lead_time_sweep.csv'}")

    _plot(results_df, out_dir / "lead_time_sweep.png")


def _plot(results_df: pd.DataFrame, path: Path):
    fig, (ax_rmse, ax_r2) = plt.subplots(1, 2, figsize=(13, 5))

    for name, g in results_df.groupby("model"):
        g = g.sort_values("lead_s")
        color = BASELINE_COLOR if name.startswith("Analytical") else MODEL_COLORS.get(name, NEUTRAL)
        style = "--" if name.startswith("Analytical") else "-"
        marker = "o"
        ax_rmse.plot(g["lead_s"] * 1000, g["RMSE"], style, marker=marker, color=color, label=name)
        ax_r2.plot(g["lead_s"] * 1000, g["R2"], style, marker=marker, color=color, label=name)

    ax_rmse.set_xlabel("Forecast horizon [ms]")
    ax_rmse.set_ylabel("Test RMSE (ground-effect factor)")
    ax_rmse.set_title("Error vs. forecast horizon")
    ax_rmse.spines[["top", "right"]].set_visible(False)
    ax_rmse.legend(fontsize=8)

    ax_r2.axhline(0.0, color=NEUTRAL, linewidth=1, linestyle=":")
    ax_r2.set_xlabel("Forecast horizon [ms]")
    ax_r2.set_ylabel("Test R2")
    ax_r2.set_title("Explained variance vs. forecast horizon")
    ax_r2.spines[["top", "right"]].set_visible(False)
    ax_r2.legend(fontsize=8)

    fig.suptitle("How far ahead can ground-effect be forecast from live sensors?", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    print(f"[sweep] wrote {path}")


if __name__ == "__main__":
    main()
