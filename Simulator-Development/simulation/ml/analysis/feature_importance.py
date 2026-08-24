#!/usr/bin/env python3
"""
Feature-importance analysis for the ground-effect forecasting task.

Fits a Random Forest (impurity-based importance, fast) and reports
permutation importance too (slower, but not biased toward
high-cardinality/continuous features the way impurity-based importance
can be) so the two can be sanity-checked against each other. The question
this answers: of everything a real drone can sense, what is the model
actually leaning on to predict ground effect -- is it mostly the
rangefinder (i.e. did it just rediscover the analytical formula's only
input), or does it also lean on motor current/rpm/gyro, which would mean
it's picking up cues the flat-ground analytical formula has no access to
at all.

Usage:
    python3 ml/analysis/feature_importance.py --csv dataset/dataset_all.csv
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
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import FEATURE_COLS, add_block_id, block_split, build_pointwise_xy, lead_s_to_rows, load_dataset

ACCENT = "#3b82f6"
ACCENT2 = "#059669"

PRETTY = {
    "gyro_x": "gyro x", "gyro_y": "gyro y", "gyro_z": "gyro z",
    "rangefinder_alt": "rangefinder altitude",
    "motor_rpm_0": "motor rpm 0", "motor_rpm_1": "motor rpm 1",
    "motor_rpm_2": "motor rpm 2", "motor_rpm_3": "motor rpm 3",
    "motor_current_0": "motor current 0", "motor_current_1": "motor current 1",
    "motor_current_2": "motor current 2", "motor_current_3": "motor current 3",
    "thrust_cmd_norm": "commanded thrust",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset/dataset_all.csv")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--block-s", type=float, default=10.0)
    ap.add_argument("--lead-s", type=float, default=0.1)
    ap.add_argument("--out-dir", default="ml/results/feature_importance")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(Path(args.csv))
    df = add_block_id(df, args.block_s)
    lead_rows = lead_s_to_rows(df, args.lead_s)
    train_df, test_df = block_split(df, args.test_frac, args.block_s, seed=0)

    X_train, y_train, _, _ = build_pointwise_xy(train_df, lead_rows)
    X_test, y_test, _, _ = build_pointwise_xy(test_df, lead_rows)

    print("[importance] fitting Random Forest...")
    model = RandomForestRegressor(n_estimators=200, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=0)
    model.fit(X_train, y_train)
    print(f"[importance] test R2 = {model.score(X_test, y_test):.4f}")

    impurity_imp = model.feature_importances_

    print("[importance] computing permutation importance (this re-scores the test set once per feature)...")
    perm = permutation_importance(model, X_test, y_test, n_repeats=8, random_state=0, n_jobs=-1)
    perm_imp = perm.importances_mean
    perm_std = perm.importances_std

    result_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "impurity_importance": impurity_imp,
        "permutation_importance_mean": perm_imp,
        "permutation_importance_std": perm_std,
    }).sort_values("permutation_importance_mean", ascending=False)
    result_df.to_csv(out_dir / "feature_importance.csv", index=False)
    print(f"[importance] wrote {out_dir / 'feature_importance.csv'}")
    print(result_df.to_string(index=False))

    _plot(result_df, args.lead_s, out_dir / "feature_importance.png")


def _plot(result_df: pd.DataFrame, lead_s: float, path: Path):
    order = result_df.sort_values("permutation_importance_mean")
    labels = [PRETTY.get(f, f) for f in order["feature"]]
    y_pos = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    ax1.barh(y_pos, order["impurity_importance"], color=ACCENT)
    ax1.set_yticks(y_pos, labels)
    ax1.set_xlabel("RF impurity-based importance")
    ax1.set_title("Impurity-based")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.barh(y_pos, order["permutation_importance_mean"],
             xerr=order["permutation_importance_std"], color=ACCENT2, capsize=3)
    ax2.set_xlabel("Permutation importance (test-set R2 drop)")
    ax2.set_title("Permutation-based")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"What does the model use to predict ground effect {lead_s * 1000:.0f}ms ahead?", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    print(f"[importance] wrote {path}")


if __name__ == "__main__":
    main()
