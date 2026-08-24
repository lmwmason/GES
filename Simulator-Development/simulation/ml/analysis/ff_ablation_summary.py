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

# (label, agl_err_rms mean, std, color, report citation)
CONFIGS = [
    ("No FF\n(feedback only)", 0.07004, 0.01050, "#c2410c", "Sec. 5.10"),
    ("ML FF, naive\n(bias-corrected retrain)", 0.07069, 0.00984, "#f59e0b", "Sec. 5.10"),
    ("ML FF, tightened\ntrust band (failed)", 0.07387, 0.01123, "#dc2626", "Sec. 5.12"),
    ("Cheeseman-Bennett\nFF", 0.05950, 0.00570, "#8892a6", "Sec. 5.10"),
    ("ML FF, hybrid\n(low-clearance cutover)", 0.06096, 0.00579, "#3b82f6", "Sec. 5.12"),
]


def main():
    out_dir = Path("ml/results/ff_mode_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [c[0] for c in CONFIGS]
    means = [c[1] for c in CONFIGS]
    stds = [c[2] for c in CONFIGS]
    colors = [c[3] for c in CONFIGS]
    cites = [c[4] for c in CONFIGS]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(CONFIGS))
    ax.barh(y_pos, means, xerr=stds, color=colors, capsize=5, height=0.6)
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("AGL tracking error RMS [m] (8-seed mean +/- std, lower = better)")
    ax.spines[["top", "right"]].set_visible(False)

    for i, (m, s, cite) in enumerate(zip(means, stds, cites)):
        ax.text(m + s + 0.001, i, f"{m:.4f} ({cite})", va="center", fontsize=9)

    ax.set_title("The full search for a working ML feedforward\n"
                  "(each attempt tried to fix the one before it -- see docs/research_report.md Sec. 5.7-5.12)",
                  fontsize=12)
    fig.tight_layout()
    path = out_dir / "ff_ablation_summary.png"
    fig.savefig(path, dpi=150)
    print(f"[ablation] wrote {path}")


if __name__ == "__main__":
    main()
