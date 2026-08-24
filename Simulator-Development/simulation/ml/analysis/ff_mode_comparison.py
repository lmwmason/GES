#!/usr/bin/env python3
"""
Runs the compiled simulator over several terrain seeds, once per
ge_ff_mode_t (none / analytical / ml -- see firmware/ge_ff.h and
GE_FF_MODE in ge_ff_mode_from_env()) per seed, and compares them two ways:

  1. A detailed IMU trace (gyro, AGL clearance, commanded throttle) for a
     single representative seed, so the effect of each feedforward
     strategy on the movement pattern ground effect induces (see
     docs/research_report.md Section 5.3) is directly visible.
  2. A per-seed bar chart of mean +/- std tracking metrics across ALL
     seeds. A single seed's aggregate numbers turned out to be dominated
     by run-to-run chaos (tiny feedforward differences get amplified by
     the somewhat underdamped altitude loop -- see Section 5.7), not by
     which feedforward mode is actually better, so any claim about which
     mode "wins" needs to be averaged over multiple independent seeds
     before it means anything.

Requires the simulator to already be built (./run.sh once, or cmake+make
in build/) -- this script only runs the existing binary with different
environment variables, it does not build anything.

Usage:
    python3 ml/analysis/ff_mode_comparison.py --seeds 1,2,3,4,5,6,7,8 --duration 60
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODES = ["none", "analytical", "ml"]
MODE_LABELS = {
    "none": "No FF (feedback only)",
    "analytical": "Cheeseman-Bennett FF",
    "ml": "ML (MLP) FF",
}
MODE_COLORS = {"none": "#c2410c", "analytical": "#8892a6", "ml": "#3b82f6"}
HOVER_AGL_M = 0.22  # must match firmware/flight_plan.c's target_agl setpoint


def run_simulation(sim_bin: Path, fc_lib: Path, mode: str, seed: int, duration: float, csv_path: Path):
    env = os.environ.copy()
    env["GE_FF_MODE"] = mode
    cmd = [str(sim_bin), "--fc", str(fc_lib), "--seed", str(seed), "--duration", str(duration),
           "--fast", "--no-telemetry", "--csv", str(csv_path)]
    print(f"[ff_compare] running mode={mode}: seed={seed} duration={duration}s")
    subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)


GE_FF_LOW_CLEARANCE_M = 0.15  # must match firmware/ge_ff.h's cutover threshold


def metrics_for(df: pd.DataFrame) -> dict:
    gyro_rms = float((df[["gyro_x", "gyro_y", "gyro_z"]] ** 2).sum(axis=1).pow(0.5).mean())
    agl_err = df["rangefinder_alt"] - HOVER_AGL_M
    active = df["rangefinder_alt"] >= GE_FF_LOW_CLEARANCE_M
    cutover = ~active
    return {
        "gyro_rms": gyro_rms,
        "agl_err_rms": float((agl_err ** 2).mean() ** 0.5),
        "agl_err_max": float(agl_err.abs().max()),
        # Segmented by whether this tick is inside the ML mode's own
        # low-clearance analytical cutover zone or not -- lets us ask "does
        # ML's offline predictive edge (Sec. 5.13) show up as a closed-loop
        # edge specifically where ML is actually still driving the
        # throttle," instead of only looking at the whole-trajectory
        # average where the cutover zone's forced-identical-to-analytical
        # behavior can hide it.
        "agl_err_rms_active_zone": float((agl_err[active] ** 2).mean() ** 0.5) if active.sum() > 10 else float("nan"),
        "agl_err_rms_cutover_zone": float((agl_err[cutover] ** 2).mean() ** 0.5) if cutover.sum() > 10 else float("nan"),
        "frac_time_in_cutover_zone": float(cutover.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8",
                     help="comma-separated terrain seeds; the first is used for the detailed IMU trace")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--sim-dir", default=".", help="simulation/ directory (containing build/)")
    ap.add_argument("--out-dir", default="ml/results/ff_mode_comparison")
    args = ap.parse_args()

    sim_dir = Path(args.sim_dir)
    sim_bin = sim_dir / "build/bin/simulator"
    fc_lib = sim_dir / "build/bin/libfc.so"
    if not sim_bin.exists() or not fc_lib.exists():
        sys.exit(f"[ff_compare] build the simulator first (./run.sh or cmake+make) -- "
                 f"expected {sim_bin} and {fc_lib}")

    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_dfs = {}
    all_metrics = {mode: [] for mode in MODES}
    for seed in seeds:
        for mode in MODES:
            csv_path = out_dir / f"telemetry_seed{seed}_{mode}.csv"
            run_simulation(sim_bin, fc_lib, mode, seed, args.duration, csv_path)
            df = pd.read_csv(csv_path, on_bad_lines="skip")
            all_metrics[mode].append(metrics_for(df))
            if seed == seeds[0]:
                detail_dfs[mode] = df
            else:
                csv_path.unlink()  # only the first seed's raw telemetry is kept on disk

    _print_summary(all_metrics, seeds)
    _plot_detail(detail_dfs, seeds[0], out_dir / "ff_mode_imu_comparison.png")
    _plot_summary(all_metrics, seeds, out_dir / "ff_mode_summary.png")
    _plot_active_zone(all_metrics, seeds, out_dir / "ff_mode_active_zone.png")


def _print_summary(all_metrics, seeds):
    print(f"\n[ff_compare] === averaged over {len(seeds)} seeds {seeds} (mean +/- std) ===")
    for mode in MODES:
        rows = pd.DataFrame(all_metrics[mode])
        print(f"[ff_compare]   {MODE_LABELS[mode]:<28s} "
              f"gyro_rms={rows['gyro_rms'].mean():.5f}+/-{rows['gyro_rms'].std():.5f}  "
              f"agl_err_rms={rows['agl_err_rms'].mean():.5f}+/-{rows['agl_err_rms'].std():.5f}")

    print(f"\n[ff_compare] === segmented by clearance >= {GE_FF_LOW_CLEARANCE_M}m "
          "(\"active zone\", where ML mode is actually still driving the throttle) ===")
    for mode in MODES:
        rows = pd.DataFrame(all_metrics[mode])
        frac = rows["frac_time_in_cutover_zone"].mean()
        print(f"[ff_compare]   {MODE_LABELS[mode]:<28s} "
              f"active_zone agl_err_rms={rows['agl_err_rms_active_zone'].mean():.5f}+/-{rows['agl_err_rms_active_zone'].std():.5f}  "
              f"(cutover-zone time fraction: {frac * 100:.1f}%)")


def _plot_detail(dfs, seed, path):
    signals = [("gyro_x", "Gyro X [rad/s]"), ("gyro_y", "Gyro Y [rad/s]"),
               ("gyro_z", "Gyro Z [rad/s]"), ("rangefinder_alt", "AGL clearance [m]"),
               ("thrust_cmd_norm", "Commanded throttle [0-1]")]
    fig, axes = plt.subplots(len(signals), len(MODES), figsize=(4.2 * len(MODES), 2.6 * len(signals)),
                              sharex="col", sharey="row")

    for col, mode in enumerate(MODES):
        df = dfs[mode]
        for row, (col_name, ylabel) in enumerate(signals):
            ax = axes[row, col]
            ax.plot(df["t"], df[col_name], color=MODE_COLORS[mode], linewidth=0.8)
            if col_name == "rangefinder_alt":
                ax.axhline(HOVER_AGL_M, color="#111827", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(MODE_LABELS[mode], fontsize=11)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == len(signals) - 1:
                ax.set_xlabel("Time [s]")

    fig.suptitle(f"IMU response to ground effect by feedforward mode (seed={seed})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    print(f"[ff_compare] wrote {path}")


def _plot_summary(all_metrics, seeds, path):
    metric_specs = [("gyro_rms", "Gyro RMS [rad/s]"), ("agl_err_rms", "AGL tracking error RMS [m]")]
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(6 * len(metric_specs), 5))

    for ax, (key, ylabel) in zip(axes, metric_specs):
        means = [np.mean([m[key] for m in all_metrics[mode]]) for mode in MODES]
        stds = [np.std([m[key] for m in all_metrics[mode]]) for mode in MODES]
        colors = [MODE_COLORS[mode] for mode in MODES]
        ax.bar([MODE_LABELS[m] for m in MODES], means, yerr=stds, color=colors, capsize=5, width=0.6)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle(f"Feedforward mode comparison, averaged over {len(seeds)} seeds {seeds}\n"
                 "(error bars = std across seeds -- overlapping bars mean the difference isn't significant)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(path, dpi=150)
    print(f"[ff_compare] wrote {path}")


def _plot_active_zone(all_metrics, seeds, path):
    """Restricted to clearance >= GE_FF_LOW_CLEARANCE_M -- for the ML mode
    this is exactly the zone where it's still actually driving the
    throttle (below that it's hard-cut over to analytical, see Section
    5.12), so this isolates whether ML's offline predictive edge (Section
    5.13, which held at every terrain slope/curvature) shows up as a
    closed-loop edge specifically where ML gets to act, instead of being
    diluted by the whole-trajectory average where the cutover zone forces
    ML and analytical to be identical."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    means = [np.nanmean([m["agl_err_rms_active_zone"] for m in all_metrics[mode]]) for mode in MODES]
    stds = [np.nanstd([m["agl_err_rms_active_zone"] for m in all_metrics[mode]]) for mode in MODES]
    colors = [MODE_COLORS[mode] for mode in MODES]
    ax.bar([MODE_LABELS[m] for m in MODES], means, yerr=stds, color=colors, capsize=5, width=0.6)
    ax.set_ylabel("AGL tracking error RMS [m], clearance >= "
                   f"{GE_FF_LOW_CLEARANCE_M}m ticks only")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=15)
    fig.suptitle(f"Restricted to the zone ML is actually still active in ({len(seeds)} seeds {seeds})\n"
                 "(does ML's offline terrain-irregularity edge, Sec. 5.13, survive here?)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(path, dpi=150)
    print(f"[ff_compare] wrote {path}")


if __name__ == "__main__":
    main()
