#!/usr/bin/env python3
"""
Illustrative (not predictive-model) figures showing *what ground effect
over irregular terrain actually looks like* in this simulator: a track
segment's terrain profile next to the ground-truth GE factor and the
analytical (flat-ground Cheeseman-Bennett) estimate the firmware computes
from rangefinder alone, plus a scatter of GE factor vs. clearance across
the whole dataset colored by how far the analytical formula is from the
truth. These are meant to motivate *why* a flat-ground formula alone is
insufficient -- terrain slope/curvature break the flat-ground assumption
the closed-form estimate relies on -- and to give a reader an intuitive
picture before looking at model error numbers.

Usage:
    python3 ml/analysis/terrain_ge_illustration.py --csv dataset/ff_train.csv
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import load_dataset

TRUTH_COLOR = "#3b82f6"
ANALYTICAL_COLOR = "#c2410c"
TERRAIN_COLOR = "#8892a6"
NEUTRAL = "#8892a6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset/ff_train.csv")
    ap.add_argument("--seed", type=int, default=None, help="restrict the track-segment panel to one episode (default: whichever seed is first in the file)")
    ap.add_argument("--t-start", type=float, default=60.0)
    ap.add_argument("--t-end", type=float, default=110.0)
    ap.add_argument("--scatter-csv", default=None,
                     help="dataset to draw the whole-dataset clearance scatter from (default: same as --csv)")
    ap.add_argument("--out-dir", default="ml/results/terrain_examples")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(Path(args.csv))
    seed = args.seed if args.seed is not None else int(df["seed"].iloc[0])
    seg = df[(df["seed"] == seed) & (df["t"] >= args.t_start) & (df["t"] <= args.t_end)].copy()
    if len(seg) == 0:
        seg = df[df["seed"] == seed].iloc[: min(3000, len(df))].copy()
        print(f"[terrain] requested t-range had no rows for seed={seed}; falling back to the episode's first rows")

    seg["terrain_h"] = seg["z"] - seg["rangefinder_alt"]
    _plot_track_segment(seg, seed, out_dir / "track_segment_ge_vs_terrain.png")

    scatter_src = Path(args.scatter_csv) if args.scatter_csv else Path(args.csv)
    scatter_df = df if scatter_src == Path(args.csv) else load_dataset(scatter_src)
    _plot_clearance_scatter(scatter_df, out_dir / "ge_factor_vs_clearance.png")


def _plot_track_segment(seg: pd.DataFrame, seed: int, path: Path):
    fig, (ax_terrain, ax_ge) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                             gridspec_kw={"height_ratios": [1, 1.4]})

    ax_terrain.fill_between(seg["t"], seg["terrain_h"], seg["terrain_h"].min() - 0.05,
                             color=TERRAIN_COLOR, alpha=0.35, label="terrain height")
    ax_terrain.plot(seg["t"], seg["z"], color="#111827", linewidth=1.3, label="vehicle altitude (world z)")
    ax_terrain.set_ylabel("Height [m]")
    ax_terrain.set_title(f"Track segment (seed={seed}): terrain profile vs. vehicle altitude")
    ax_terrain.spines[["top", "right"]].set_visible(False)
    ax_terrain.legend(fontsize=8, loc="upper right")

    ax_ge.plot(seg["t"], seg["ge_factor_truth_mean"], color=TRUTH_COLOR, linewidth=1.5,
               label="ground truth GE factor (mean over 4 rotors)")
    ax_ge.plot(seg["t"], seg["ge_ff_analytical"], color=ANALYTICAL_COLOR, linewidth=1.2,
               linestyle="--", label="analytical (flat-ground Cheeseman-Bennett) estimate")
    ax_ge.axhline(1.0, color=NEUTRAL, linewidth=1, linestyle=":")
    ax_ge.set_xlabel("Time [s]")
    ax_ge.set_ylabel("Thrust factor")
    ax_ge.set_title("Ground-truth ground-effect factor vs. the analytical estimate")
    ax_ge.spines[["top", "right"]].set_visible(False)
    ax_ge.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[terrain] wrote {path}")


def _plot_clearance_scatter(df: pd.DataFrame, path: Path):
    err = (df["ge_ff_analytical"] - df["ge_factor_truth_mean"]).to_numpy()
    clearance = df["rangefinder_alt"].to_numpy()
    truth = df["ge_factor_truth_mean"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    idx = np.random.RandomState(0).choice(len(df), size=min(40000, len(df)), replace=False)
    sc = ax1.scatter(clearance[idx], truth[idx], s=3, c=np.abs(err[idx]), cmap="viridis", alpha=0.5)
    zz = np.linspace(max(clearance.min(), 1e-3), clearance.max(), 200)
    rotor_r = 0.127
    z_eff = np.maximum(zz, rotor_r * 0.5)
    flat_curve = 1.0 / (1.0 - (rotor_r / (4.0 * z_eff)) ** 2)
    ax1.plot(zz, np.clip(flat_curve, 0.85, 1.6), color=ANALYTICAL_COLOR, linewidth=1.5,
              label="flat-ground Cheeseman-Bennett curve")
    ax1.set_xlabel("Rangefinder clearance [m]")
    ax1.set_ylabel("Ground truth GE factor")
    ax1.set_title("GE factor vs. clearance\n(color = |analytical error|, over irregular terrain)")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(fontsize=8)
    cbar = fig.colorbar(sc, ax=ax1)
    cbar.set_label("|analytical - truth|")

    bins = np.linspace(0, min(clearance.max(), 1.0), 21)
    bin_idx = np.digitize(clearance, bins)
    bin_centers, mean_abs_err, counts = [], [], []
    for b in range(1, len(bins)):
        mask = bin_idx == b
        if mask.sum() < 20:
            continue
        bin_centers.append((bins[b - 1] + bins[b]) / 2)
        mean_abs_err.append(np.mean(np.abs(err[mask])))
        counts.append(mask.sum())
    ax2.bar(bin_centers, mean_abs_err, width=(bins[1] - bins[0]) * 0.85, color=ANALYTICAL_COLOR, alpha=0.85)
    ax2.set_xlabel("Rangefinder clearance [m]")
    ax2.set_ylabel("Mean |analytical estimate - ground truth|")
    ax2.set_title("Where the flat-ground formula is most wrong")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Irregular terrain breaks the flat-ground ground-effect assumption", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    print(f"[terrain] wrote {path}")


if __name__ == "__main__":
    main()
