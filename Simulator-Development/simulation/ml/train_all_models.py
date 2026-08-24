#!/usr/bin/env python3
"""
Full model comparison for ground-effect prediction: classical/algorithmic
models (sklearn) through deep sequence models (PyTorch LSTM, Transformer),
all predicting ge_factor_truth_mean from only realistically-sensed inputs
(gyro, rangefinder, motor rpm/current, commanded thrust).

Models compared:
  - Analytical (baseline)   the firmware's flat-ground formula, zero training
  - Linear Regression       algorithmic baseline
  - Decision Tree           algorithmic, nonlinear
  - Random Forest           algorithmic, ensemble/bagging
  - Gradient Boosting       algorithmic, ensemble/boosting
  - MLP (PyTorch)           pointwise neural net
  - LSTM (PyTorch)          sequence model, last `--window` samples
  - Transformer (PyTorch)   sequence model, self-attention, last `--window` samples

This is a genuine *feedforward* task, not a nowcasting one: by the time a
prediction is computed and the control loop acts on it, the vehicle is no
longer where it was when the input sensors were sampled. So every model
here predicts the ground-effect factor `--lead-s` seconds *ahead* of its
input window, not at the input window's own timestamp. Set --lead-s 0 to
go back to plain nowcasting for comparison.

Usage:
    python3 ml/train_all_models.py --csv dataset/ff_train.csv
    python3 ml/train_all_models.py --csv dataset/ff_train.csv --window 32 --epochs 25 --lead-s 0.1
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
import torch
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from data import (FEATURE_COLS, TARGET_COL, BASELINE_COL, add_block_id, block_split,
                   build_pointwise_xy, build_windows, lead_s_to_rows, load_dataset)
from models_torch import MLPNet, LSTMNet, TransformerNet, fit_torch_model, predict_torch_model

NEUTRAL = "#8892a6"
ACCENT = "#3b82f6"
BASELINE_COLOR = "#c2410c"


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def run_classical_models(train_df, test_df, lead_rows):
    X_train, y_train, _, _ = build_pointwise_xy(train_df, lead_rows)
    X_test, y_test, _, _ = build_pointwise_xy(test_df, lead_rows)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=0),
        "Gradient Boosting": HistGradientBoostingRegressor(max_depth=6, max_iter=300, random_state=0),
    }

    results = []
    fitted = {}
    preds = {}
    for name, model in models.items():
        if name == "Linear Regression":
            # only the linear model is scale-sensitive here
            scaler = StandardScaler().fit(X_train)
            Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
        else:
            Xtr, Xte = X_train, X_test
        t0 = time.time()
        model.fit(Xtr, y_train)
        train_s = time.time() - t0
        pred = model.predict(Xte)
        metrics = evaluate(y_test, pred)
        results.append({"model": name, "train_s": train_s, "n_test": len(y_test), **metrics})
        fitted[name] = model
        preds[name] = (y_test, pred)
        print(f"[train] {name:<20s} trained in {train_s:6.2f}s  "
              f"MAE={metrics['MAE']:.5f}  RMSE={metrics['RMSE']:.5f}  R2={metrics['R2']:.4f}")
    return results, fitted, preds


def run_torch_pointwise(train_df, val_df, test_df, lead_rows, epochs, device):
    Xtr_raw, ytr, _, _ = build_pointwise_xy(train_df, lead_rows)
    Xval_raw, yval, _, _ = build_pointwise_xy(val_df, lead_rows)
    Xte_raw, yte, _, _ = build_pointwise_xy(test_df, lead_rows)

    scaler = StandardScaler().fit(Xtr_raw)
    Xtr = scaler.transform(Xtr_raw).astype(np.float32)
    Xval = scaler.transform(Xval_raw).astype(np.float32)
    Xte = scaler.transform(Xte_raw).astype(np.float32)

    print("[train] MLP (PyTorch)")
    model = MLPNet(n_features=len(FEATURE_COLS))
    t0 = time.time()
    fit_torch_model(model, Xtr, ytr, Xval, yval, epochs=epochs, device=device)
    train_s = time.time() - t0
    pred = predict_torch_model(model, Xte, device=device)
    metrics = evaluate(yte, pred)
    print(f"[train] {'MLP (PyTorch)':<20s} trained in {train_s:6.2f}s  "
          f"MAE={metrics['MAE']:.5f}  RMSE={metrics['RMSE']:.5f}  R2={metrics['R2']:.4f}")
    result = {"model": "MLP (PyTorch)", "train_s": train_s, "n_test": len(yte), **metrics}
    return result, model, scaler, (yte, pred)


def run_torch_sequence(name, model, train_df, val_df, test_df, window, lead_rows, epochs, device):
    Xtr_raw, ytr, _ = build_windows(train_df, window, lead_rows)
    Xval_raw, yval, _ = build_windows(val_df, window, lead_rows)
    Xte_raw, yte, _ = build_windows(test_df, window, lead_rows)

    scaler = StandardScaler().fit(Xtr_raw.reshape(-1, Xtr_raw.shape[-1]))

    def scale(X):
        shp = X.shape
        return scaler.transform(X.reshape(-1, shp[-1])).reshape(shp).astype(np.float32)

    Xtr, Xval, Xte = scale(Xtr_raw), scale(Xval_raw), scale(Xte_raw)

    print(f"[train] {name} ({len(Xtr)} train / {len(Xval)} val / {len(Xte)} test windows, window={window})")
    t0 = time.time()
    fit_torch_model(model, Xtr, ytr, Xval, yval, epochs=epochs, device=device)
    train_s = time.time() - t0
    pred = predict_torch_model(model, Xte, device=device)
    metrics = evaluate(yte, pred)
    print(f"[train] {name:<20s} trained in {train_s:6.2f}s  "
          f"MAE={metrics['MAE']:.5f}  RMSE={metrics['RMSE']:.5f}  R2={metrics['R2']:.4f}")
    result = {"model": name, "train_s": train_s, "n_test": len(yte), **metrics}
    return result, (yte, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset/ff_train.csv")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.15, help="fraction of the *training* blocks reserved for early stopping")
    ap.add_argument("--block-s", type=float, default=10.0)
    ap.add_argument("--window", type=int, default=32, help="sequence length (timesteps) for LSTM/Transformer")
    ap.add_argument("--lead-s", type=float, default=0.1,
                     help="forecast horizon in seconds: predict the GE factor this far into the "
                          "future (accounting for inference + control-loop latency) instead of "
                          "the current instant. 0 disables this (nowcasting).")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--out-dir", default="ml/results/pilot_seed1_lead100ms")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] device: {args.device}")

    df = load_dataset(Path(args.csv))
    df = add_block_id(df, args.block_s)
    lead_rows = lead_s_to_rows(df, args.lead_s)
    train_df, test_df = block_split(df, args.test_frac, args.block_s, seed=0)
    # Carve a validation split out of *training* blocks only, for the
    # PyTorch models' early stopping -- never peek at the test set to pick
    # the best epoch.
    train2_df, val_df = block_split(train_df, args.val_frac, args.block_s, seed=1)

    results = []
    all_preds = {}

    _, y_base_true, _, base_extra = build_pointwise_xy(test_df, lead_rows, extra_cols=[BASELINE_COL])
    baseline_pred = base_extra[BASELINE_COL]
    baseline_metrics = evaluate(y_base_true, baseline_pred)
    results.append({"model": "Analytical (baseline)", "train_s": 0.0, "n_test": len(y_base_true), **baseline_metrics})
    all_preds["Analytical (baseline)"] = (y_base_true, baseline_pred)
    print(f"[train] {'Analytical (baseline)':<20s} "
          f"MAE={baseline_metrics['MAE']:.5f}  RMSE={baseline_metrics['RMSE']:.5f}  R2={baseline_metrics['R2']:.4f}")

    # classical models train on the full (train2+val) block set -- they
    # don't need a validation split for early stopping the way the PyTorch
    # models do, so there's no reason to withhold that data from them.
    classical_results, classical_fitted, classical_preds = run_classical_models(train_df, test_df, lead_rows)
    results.extend(classical_results)
    all_preds.update(classical_preds)

    mlp_result, mlp_model, mlp_scaler, mlp_preds = run_torch_pointwise(
        train2_df, val_df, test_df, lead_rows, args.epochs, args.device)
    results.append(mlp_result)
    all_preds["MLP (PyTorch)"] = mlp_preds

    lstm_model = LSTMNet(n_features=len(FEATURE_COLS))
    lstm_result, lstm_preds = run_torch_sequence("LSTM (PyTorch)", lstm_model, train2_df, val_df, test_df,
                                                  args.window, lead_rows, args.epochs, args.device)
    results.append(lstm_result)
    all_preds["LSTM (PyTorch)"] = lstm_preds

    transformer_model = TransformerNet(n_features=len(FEATURE_COLS), window=args.window)
    transformer_result, transformer_preds = run_torch_sequence(
        "Transformer (PyTorch)", transformer_model, train2_df, val_df, test_df,
        args.window, lead_rows, args.epochs, args.device)
    results.append(transformer_result)
    all_preds["Transformer (PyTorch)"] = transformer_preds

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    results_df.to_csv(out_dir / "model_comparison_all.csv", index=False)
    print("\n[train] === results (sorted by test RMSE, lower is better) ===")
    print(results_df.to_string(index=False))
    print(f"\n[train] note: all models predict {args.lead_s * 1000:.0f}ms into the future "
          f"({lead_rows} row(s)), not the current instant. Sequence models (LSTM/Transformer) "
          "are also scored on fewer test rows than the pointwise ones -- each block loses its "
          "first (window-1) rows, which can't fill a full window.")

    best_name = results_df.iloc[0]["model"]
    print(f"\n[train] best model: {best_name}")

    joblib.dump(
        {"classical_models": classical_fitted, "mlp_model_state": mlp_model.state_dict(),
         "mlp_scaler": mlp_scaler, "features": FEATURE_COLS, "target": TARGET_COL},
        out_dir / "all_models.joblib",
    )

    _plot_comparison(results_df, args.lead_s, out_dir / "model_comparison_all.png")
    y_true_best, y_pred_best = all_preds[best_name]
    _plot_predictions(y_true_best, y_pred_best, best_name, args.lead_s, out_dir / "best_model_predictions_all.png")

    # One combined figure showing every model's predicted-vs-actual panel,
    # so you can see *how* each one is wrong, not just its RMSE.
    _plot_predictions_grid(all_preds, results_df, args.lead_s, out_dir / "all_models_predictions_grid.png")


def _plot_comparison(results_df: pd.DataFrame, lead_s: float, path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    names = results_df["model"].tolist()
    rmse = results_df["RMSE"].tolist()
    best_idx = int(results_df["RMSE"].idxmin())
    colors = [BASELINE_COLOR if n.startswith("Analytical") else
              (ACCENT if i == best_idx else NEUTRAL)
              for i, n in enumerate(names)]
    bars = ax.bar(names, rmse, color=colors, width=0.6)
    ax.set_ylabel("Test RMSE (ground-effect factor)")
    horizon = f"{lead_s * 1000:.0f}ms ahead" if lead_s > 0 else "nowcast, no lead"
    ax.set_title(f"Ground-effect prediction ({horizon}): full model comparison (lower = better)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=25)
    for bar, v in zip(bars, rmse):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[train] wrote {path}")


def _plot_predictions(y_true, y_pred, name: str, lead_s: float, path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    lo, hi = float(np.min(y_true)), float(np.max(y_true))
    ax.plot([lo, hi], [lo, hi], color=NEUTRAL, linewidth=1, linestyle="--")
    ax.scatter(y_true, y_pred, s=6, color=ACCENT, alpha=0.35, edgecolors="none")
    ax.set_xlabel(f"Ground truth GE factor ({lead_s * 1000:.0f}ms ahead)" if lead_s > 0 else "Ground truth GE factor")
    ax.set_ylabel("Predicted GE factor")
    ax.set_title(f"{name}: predicted vs. ground truth\n(held-out track segments)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[train] wrote {path}")


def _plot_predictions_grid(all_preds: dict, results_df: pd.DataFrame, lead_s: float, path: Path):
    names = results_df["model"].tolist()
    n = len(names)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, name in enumerate(names):
        ax = axes[i]
        y_true, y_pred = all_preds[name]
        lo, hi = float(np.min(y_true)), float(np.max(y_true))
        color = BASELINE_COLOR if name.startswith("Analytical") else ACCENT
        ax.plot([lo, hi], [lo, hi], color=NEUTRAL, linewidth=1, linestyle="--")
        ax.scatter(y_true, y_pred, s=4, color=color, alpha=0.3, edgecolors="none")
        rmse = results_df.loc[results_df["model"] == name, "RMSE"].iloc[0]
        ax.set_title(f"{name}\nRMSE={rmse:.4f}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    horizon = f"{lead_s * 1000:.0f}ms ahead" if lead_s > 0 else "nowcast"
    fig.suptitle(f"Predicted vs. ground truth GE factor ({horizon}), all models (held-out data)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    print(f"[train] wrote {path}")


if __name__ == "__main__":
    main()
