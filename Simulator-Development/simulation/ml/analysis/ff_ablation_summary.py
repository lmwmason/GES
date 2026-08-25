#!/usr/bin/env python3
"""
Ablation chart of every closed-loop feedforward configuration tried in
docs/research_report.md Sections 5.7-5.12, in one figure -- the numbers
here are transcribed from those sections (each required a different
firmware/data state to reproduce, not a single reusable command), with a
citation per bar so the provenance is traceable back to the report.

This is a summary/narrative figure, not a new experiment.

Usage:
    python3 ml/analysis/ff_ablation_summary.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Corrected quadratic-actuator 8-seed rerun (2026-08-25). The narrowed gate
# is +/-5%; rejected predictions retain ge_ff_select()'s factor=1 fallback.
# (mode, label, color)
CONFIGS = [
    ("none", "No FF\n(feedback only)", "#c2410c"),
    ("analytical", "Cheeseman-Bennett\nFF", "#8892a6"),
    ("ml_pure", "Pure ML FF\n(20% trust band)", "#f59e0b"),
    ("ml_narrow", "ML FF, narrowed\n5% trust band", "#dc2626"),
    ("ml", "ML FF, hybrid\n(low-clearance cutover)", "#3b82f6"),
]


def main():
    out_dir = Path("ml/results/ff_mode_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    telemetry_dir = Path("ml/results/ff_mode_comparison_sqrt")
    labels = [c[1] for c in CONFIGS]
    colors = [c[2] for c in CONFIGS]
    means, stds = [], []
    for mode, _, _ in CONFIGS:
        values = []
        for seed in range(1, 9):
            df = pd.read_csv(telemetry_dir / f"telemetry_seed{seed}_{mode}.csv")
            values.append(float(np.sqrt(np.mean((df["rangefinder_alt"] - 0.22) ** 2))))
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values, ddof=1)))
        print(f"[ablation] {mode:<10s} AGL RMS={means[-1]:.6f}+/-{stds[-1]:.6f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(CONFIGS))
    ax.barh(y_pos, means, xerr=stds, color=colors, capsize=5, height=0.6)
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("AGL tracking error RMS [m] (8-seed mean +/- std, lower = better)")
    ax.spines[["top", "right"]].set_visible(False)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.001, i, f"{m:.4f}", va="center", fontsize=9)

    ax.set_title("Closed-loop feedforward comparison\n"
                 "(corrected quadratic actuator mapping; 8 terrain seeds)",
                  fontsize=12)
    fig.tight_layout()
    path = out_dir / "ff_ablation_summary.png"
    fig.savefig(path, dpi=150)
    print(f"[ablation] wrote {path}")


if __name__ == "__main__":
    main()
