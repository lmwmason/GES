#!/usr/bin/env python3
"""Combine independently checkpointed offline runs without editing metrics."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(); rows = []
    for run in args.runs:
        path = run / "offline_metrics.csv"
        if not path.exists(): raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {"model", "MAE", "RMSE", "R2", "n_test"}
        if frame.empty or required - set(frame): raise ValueError(f"malformed {path}")
        frame["source_run"] = run.name
        frame["evaluation_support"] = frame["model"].map(
            lambda name: "seedwise_window32_lead100ms" if str(name).upper() in {"LSTM", "TRANSFORMER"}
            else "seedwise_pointwise_lead100ms")
        rows.append(frame)
    combined = pd.concat(rows, ignore_index=True)
    duplicates = combined[combined.duplicated("model", keep=False)]
    if not duplicates.empty: raise ValueError(f"duplicate model rows: {duplicates.model.tolist()}")
    if combined.n_test.nunique() != 1: raise ValueError("models were not evaluated on equal test sample counts")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined.sort_values("RMSE").to_csv(args.out_dir / "offline_metrics.csv", index=False)


if __name__ == "__main__": main()
