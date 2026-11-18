#!/usr/bin/env python3
"""Generate review-ready tables and figures from immutable experiment outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


def require(path: Path, columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = columns - set(frame)
    if missing or frame.empty:
        raise ValueError(f"invalid {path}: missing={sorted(missing)}, rows={len(frame)}")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--closed-loop", type=Path)
    ap.add_argument("--bias", type=Path)
    ap.add_argument("--offline", type=Path)
    ap.add_argument("--robustness", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 300, "font.size": 9})

    if args.closed_loop:
        df = require(args.closed_loop / "per_seed_metrics.csv",
                     {"mode", "seed", "agl_rms_primary", "gyro_rms", "minimum_clearance_m"})
        summary = df.groupby("mode").agg(
            n=("seed", "count"), agl_rms_mean=("agl_rms_primary", "mean"),
            agl_rms_sd=("agl_rms_primary", "std"), gyro_rms_mean=("gyro_rms", "mean"),
            minimum_clearance_m=("minimum_clearance_m", "min"),
            violation_duration_mean_s=("low_clearance_duration_s", "mean")).reset_index()
        summary["agl_rms_ci95"] = stats.t.ppf(.975, summary.n - 1) * summary.agl_rms_sd / summary.n.pow(.5)
        summary.to_csv(args.out_dir / "closed_loop_table.csv", index=False)
        ax = summary.plot.bar(x="mode", y="agl_rms_mean", yerr="agl_rms_ci95",
            legend=False, capsize=3, color="#4472C4")
        ax.set(xlabel="Controller", ylabel="AGL tracking RMS (m)")
        plt.tight_layout(); plt.savefig(args.out_dir / "closed_loop_agl_rms.png"); plt.close()

    if args.bias:
        df = require(args.bias / "per_seed_metrics.csv", {"oracle_bias", "agl_rms_primary", "minimum_clearance_m"})
        summary = df.groupby("oracle_bias").agg(n=("seed","count"),
            agl_rms_mean=("agl_rms_primary", "mean"), agl_rms_sem=("agl_rms_primary", "sem"),
            p05_clearance_mean=("p05_clearance_m","mean"), p05_clearance_sem=("p05_clearance_m","sem"),
            violation_duration_mean=("low_clearance_duration_s","mean"),
            violation_duration_sem=("low_clearance_duration_s","sem"),
            minimum_clearance_mean=("minimum_clearance_m", "mean")).reset_index()
        tcrit = stats.t.ppf(.975, summary.n - 1)
        summary.to_csv(args.out_dir / "oracle_bias_table.csv", index=False)
        fig, axes = plt.subplots(1,3,figsize=(9,3))
        panels=(("agl_rms_mean","agl_rms_sem","AGL tracking RMS (m)"),
                ("p05_clearance_mean","p05_clearance_sem","5th-percentile clearance (m)"),
                ("violation_duration_mean","violation_duration_sem","Low-clearance duration (s)"))
        for ax,(mean,sem,label) in zip(axes,panels):
            ax.errorbar(summary.oracle_bias,summary[mean],yerr=tcrit*summary[sem],marker="o",capsize=3)
            ax.axvline(0,color="0.5",lw=.8); ax.set(xlabel="Signed oracle GE-factor bias",ylabel=label)
        fig.tight_layout(); fig.savefig(args.out_dir / "oracle_bias_intervention.png"); plt.close(fig)

    if args.offline:
        df = require(args.offline / "offline_metrics.csv", {"model", "RMSE", "MAE", "R2"})
        df.sort_values("RMSE").to_csv(args.out_dir / "offline_seedwise_table.csv", index=False)
        ax = df.sort_values("RMSE").plot.bar(x="model", y="RMSE", legend=False, color="#70AD47")
        ax.set(xlabel="Model", ylabel="Held-out-seed RMSE"); plt.xticks(rotation=35, ha="right")
        plt.tight_layout(); plt.savefig(args.out_dir / "offline_seedwise_rmse.png"); plt.close()

    if args.robustness:
        df = require(args.robustness / "per_seed_metrics.csv",
                     {"slope_exponent", "curvature_gain", "mode", "agl_rms_primary"})
        means = df.groupby(["slope_exponent", "curvature_gain", "mode"], as_index=False).agl_rms_primary.mean()
        means.to_csv(args.out_dir / "robustness_table.csv", index=False)
        wide = means.pivot(index=["slope_exponent", "curvature_gain"], columns="mode",
                           values="agl_rms_primary")
        delta = (wide["ml"] - wide["analytical"]).unstack()
        fig, ax = plt.subplots(); image = ax.imshow(delta, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(len(delta.columns)), delta.columns)
        ax.set_yticks(range(len(delta.index)), delta.index)
        ax.set(xlabel="Curvature gain", ylabel="Slope exponent",
               title="Hybrid MLP minus analytical AGL RMS (m)")
        for i in range(delta.shape[0]):
            for j in range(delta.shape[1]): ax.text(j, i, f"{delta.iloc[i,j]:.4f}", ha="center", va="center")
        fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(args.out_dir / "robustness_heatmap.png"); plt.close(fig)


if __name__ == "__main__": main()
