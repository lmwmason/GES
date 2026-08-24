#!/usr/bin/env python3
"""
Trains just the pointwise MLP (skips the classical models and the
sequence models train_all_models.py also compares) and saves a joblib in
the same shape ml/export_mlp_to_c.py expects. Meant for quick, targeted
retrains -- e.g. after the flight envelope a deployed model needs to cover
changes (see docs/research_report.md Section 5.9: the MLP trained on the
~0.34m-hover dataset performed worse than the analytical baseline once
deployed at a 0.22m hover, a covariate-shift problem, not a capacity
problem -- retraining on data from the actual deployment altitude is the
fix, and doesn't require redoing the full classical+LSTM+Transformer
comparison).

Usage:
    python3 ml/train_mlp_only.py --csv dataset/lowalt_dataset/dataset_all.csv \
        --out ml/results/lowalt_mlp/mlp_only.joblib
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from data import FEATURE_COLS, TARGET_COL, BASELINE_COL, add_block_id, block_split, build_pointwise_xy, lead_s_to_rows, load_dataset
from models_torch import MLPNet, fit_torch_model, predict_torch_model


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--block-s", type=float, default=10.0)
    ap.add_argument("--lead-s", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    df = load_dataset(Path(args.csv))
    df = add_block_id(df, args.block_s)
    lead_rows = lead_s_to_rows(df, args.lead_s)
    train_df, test_df = block_split(df, args.test_frac, args.block_s, seed=0)
    train2_df, val_df = block_split(train_df, args.val_frac, args.block_s, seed=1)

    _, y_base_true, _, base_extra = build_pointwise_xy(test_df, lead_rows, extra_cols=[BASELINE_COL])
    baseline_metrics = evaluate(y_base_true, base_extra[BASELINE_COL])
    print(f"[train] Analytical (baseline)  MAE={baseline_metrics['MAE']:.5f}  "
          f"RMSE={baseline_metrics['RMSE']:.5f}  R2={baseline_metrics['R2']:.4f}")

    Xtr_raw, ytr, _, _ = build_pointwise_xy(train2_df, lead_rows)
    Xval_raw, yval, _, _ = build_pointwise_xy(val_df, lead_rows)
    Xte_raw, yte, _, _ = build_pointwise_xy(test_df, lead_rows)

    scaler = StandardScaler().fit(Xtr_raw)
    Xtr = scaler.transform(Xtr_raw).astype(np.float32)
    Xval = scaler.transform(Xval_raw).astype(np.float32)
    Xte = scaler.transform(Xte_raw).astype(np.float32)

    print("[train] MLP (PyTorch)")
    model = MLPNet(n_features=len(FEATURE_COLS))
    t0 = time.time()
    fit_torch_model(model, Xtr, ytr, Xval, yval, epochs=args.epochs, device=args.device)
    train_s = time.time() - t0
    pred = predict_torch_model(model, Xte, device=args.device)
    metrics = evaluate(yte, pred)
    print(f"[train] MLP (PyTorch)  trained in {train_s:.2f}s  "
          f"MAE={metrics['MAE']:.5f}  RMSE={metrics['RMSE']:.5f}  R2={metrics['R2']:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"mlp_model_state": model.state_dict(), "mlp_scaler": scaler,
         "features": FEATURE_COLS, "target": TARGET_COL},
        out_path,
    )
    print(f"[train] wrote {out_path}")


if __name__ == "__main__":
    main()
