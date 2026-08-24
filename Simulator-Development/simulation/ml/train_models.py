#!/usr/bin/env python3
"""
Train and compare regression models that predict the ground-effect thrust
factor (ge_factor_truth_mean) from only what a real drone could sense --
gyro, a downward rangefinder, motor rpm/current, and commanded thrust.
This is the model meant to eventually replace ge_ff_ai_predict_stub() in
firmware/ge_ff.c.

The analytical Cheeseman-Bennett estimate (ge_ff_analytical, already
computed by the firmware from rangefinder alone) is reported alongside the
trained models as a zero-training-cost baseline -- the whole point of this
project is to see how much a model can beat it by picking up on terrain
curvature/slope cues the flat-ground formula can't see.

Usage:
    python3 ml/train_models.py [--csv PATH] [--test-frac 0.2] [--out-dir ml/results/pilot_seed1_nowcast]
"""
import argparse
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

FEATURE_COLS = [
    "gyro_x", "gyro_y", "gyro_z",
    "rangefinder_alt",
    "motor_rpm_0", "motor_rpm_1", "motor_rpm_2", "motor_rpm_3",
    "motor_current_0", "motor_current_1", "motor_current_2", "motor_current_3",
    "thrust_cmd_norm",
]
TARGET_COL = "ge_factor_truth_mean"
BASELINE_COL = "ge_ff_analytical"

# Bar color: single neutral hue for all models, a distinct accent for the
# best one -- one series (RMSE per model), so no legend is needed; the
# x-axis labels already carry identity.
NEUTRAL = "#8892a6"
ACCENT = "#3b82f6"
BASELINE_COLOR = "#c2410c"


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL, BASELINE_COL])
    df = df.sort_values("t").reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[train] dropped {dropped} malformed/NaN rows (likely a truncated last line)")
    print(f"[train] loaded {len(df)} rows, {df['seed'].nunique()} episode(s), "
          f"t in [{df['t'].min():.1f}, {df['t'].max():.1f}]s")
    return df


def block_split(df: pd.DataFrame, test_frac: float, block_s: float = 10.0, seed: int = 0):
    """Chop the flight into contiguous block_s-second blocks and randomly
    hold out whole blocks for testing, scattered across the whole track.

    A plain random row split would leak badly (rows are ~60Hz samples of a
    continuous flight, so neighbors are nearly identical). A single
    "last N%" time split avoids that leak but is a coin flip on whether
    that tail happens to be a representative stretch of terrain -- on a
    single long flight it easily lands on an unusually flat/uniform
    segment and gives a misleading (or here, wildly negative) R2. Holding
    out many small, scattered blocks instead keeps each block internally
    contiguous (so leakage is limited to block edges) while still giving
    the test set a representative mix of flat and irregular terrain. """
    block_id = (df["t"] // block_s).astype(int)
    unique_blocks = block_id.unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_blocks)
    n_test_blocks = max(1, int(len(unique_blocks) * test_frac))
    test_blocks = set(unique_blocks[:n_test_blocks])
    is_test = block_id.isin(test_blocks)
    train_df, test_df = df[~is_test], df[is_test]
    print(f"[train] split: {len(train_df)} train rows / {len(test_df)} test rows "
          f"({n_test_blocks}/{len(unique_blocks)} {block_s:.0f}s blocks held out, scattered across the track)")
    return train_df, test_df


def build_models():
    return {
        "Linear Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=0),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_depth=6, max_iter=300, random_state=0),
        "Neural Net (MLP)": Pipeline([
            ("scale", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=800, early_stopping=True,
                random_state=0)),
        ]),
    }


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset/ff_train.csv")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--out-dir", default="ml/results/pilot_seed1_nowcast")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(Path(args.csv))
    train_df, test_df = block_split(df, args.test_frac)

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    results = []

    # Baseline: the firmware's own analytical (flat-ground, rangefinder-only)
    # estimate, at zero training cost.
    baseline_metrics = evaluate(y_test, test_df[BASELINE_COL])
    results.append({"model": "Analytical (baseline)", "train_s": 0.0, **baseline_metrics})

    fitted = {}
    for name, model in build_models().items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_s = time.time() - t0
        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        results.append({"model": name, "train_s": train_s, **metrics})
        fitted[name] = model
        print(f"[train] {name:<20s} trained in {train_s:6.2f}s  "
              f"MAE={metrics['MAE']:.5f}  RMSE={metrics['RMSE']:.5f}  R2={metrics['R2']:.4f}")

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    results_df.to_csv(out_dir / "model_comparison.csv", index=False)
    print("\n[train] === results (sorted by test RMSE, lower is better) ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    if best_name in fitted:
        joblib.dump(
            {"model": fitted[best_name], "features": FEATURE_COLS, "target": TARGET_COL},
            out_dir / "best_model.joblib",
        )
        print(f"\n[train] best model: {best_name} -> saved to {out_dir / 'best_model.joblib'}")
    else:
        print(f"\n[train] best is the analytical baseline (no trained model beat it)")

    _plot_comparison(results_df, out_dir / "model_comparison.png")
    if best_name in fitted:
        _plot_predictions(y_test.values, fitted[best_name].predict(X_test), best_name,
                           out_dir / "best_model_predictions.png")


def _plot_comparison(results_df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = results_df["model"].tolist()
    rmse = results_df["RMSE"].tolist()
    colors = [BASELINE_COLOR if n.startswith("Analytical") else
              (ACCENT if i == results_df["RMSE"].idxmin() else NEUTRAL)
              for i, n in enumerate(names)]
    bars = ax.bar(names, rmse, color=colors, width=0.6)
    ax.set_ylabel("Test RMSE (ground-effect factor)")
    ax.set_title("Ground-effect prediction: model comparison (lower = better)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=20)
    for bar, v in zip(bars, rmse):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[train] wrote {path}")


def _plot_predictions(y_true, y_pred, name: str, path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    lo, hi = float(np.min(y_true)), float(np.max(y_true))
    ax.plot([lo, hi], [lo, hi], color=NEUTRAL, linewidth=1, linestyle="--")
    ax.scatter(y_true, y_pred, s=6, color=ACCENT, alpha=0.35, edgecolors="none")
    ax.set_xlabel("Ground truth GE factor")
    ax.set_ylabel("Predicted GE factor")
    ax.set_title(f"{name}: predicted vs. ground truth\n(held-out track segments)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[train] wrote {path}")


if __name__ == "__main__":
    main()
