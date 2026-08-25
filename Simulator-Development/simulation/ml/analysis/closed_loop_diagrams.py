#!/usr/bin/env python3
"""Regenerate paper diagrams affected by the actuator-mapping correction."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PAPER = Path("/home/just_grassy/Desktop/01.Home/07.Journal/springer-nature/GES__Ground_Effect_Stabilizer")
BG, BOX, EDGE, TEXT = "#000000", "#404040", "#aaaaaa", "#f4f4f4"


def box(ax, xy, wh, text, dashed=False, fs=11):
    x, y = xy; w, h = wh
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                       facecolor=BOX, edgecolor=EDGE, linewidth=1.3,
                       linestyle="--" if dashed else "-")
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", color=TEXT, fontsize=fs)
    return p


def arrow(ax, a, b, label=None, dashed=False):
    p = FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14,
                        color=EDGE, linewidth=1.2,
                        linestyle="--" if dashed else "-")
    ax.add_patch(p)
    if label:
        ax.text((a[0]+b[0])/2, (a[1]+b[1])/2, label, color=TEXT,
                fontsize=9, ha="center", va="center", backgroundcolor=BG)


def canvas(figsize):
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(BG); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


def feedback():
    fig, ax = canvas((10, 11))
    box(ax, (.30,.86), (.40,.09), "Clearance drops\n(e.g. terrain disturbance)")
    box(ax, (.56,.65), (.38,.12), "ML model OVERPREDICTS\nthe GE thrust factor")
    box(ax, (.56,.43), (.38,.13), "throttle_cmd = throttle_raw / sqrt(predicted factor)\n"
        "Overprediction => excessive compensation")
    box(ax, (.30,.22), (.40,.11), "Actual thrust falls short\n=> further clearance loss")
    box(ax, (.03,.43), (.38,.12), "Lower clearance enters\nthe next prediction")
    box(ax, (.02,.67), (.35,.10), "corr(prediction - truth,\nnext clearance change) = -0.246", dashed=True, fs=10)
    box(ax, (.30,.03), (.40,.10), "Safety guard: use analytical estimate\nwhen clearance < 0.15 m", dashed=True)
    arrow(ax,(.70,.86),(.75,.77)); arrow(ax,(.75,.65),(.75,.56))
    arrow(ax,(.75,.43),(.60,.33)); arrow(ax,(.30,.275),(.22,.43))
    arrow(ax,(.22,.55),(.40,.86)); arrow(ax,(.50,.22),(.50,.13),dashed=True)
    fig.savefig(PAPER/"fig_diagram_feedback_loop.png", dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def mitigation():
    fig, ax = canvas((9, 10))
    box(ax,(.30,.88),(.40,.08),"Each control tick (GE_FF_MODE = ml)")
    box(ax,(.30,.72),(.40,.08),"Read rangefinder_alt")
    box(ax,(.30,.54),(.40,.09),"clearance < 0.15 m ?")
    box(ax,(.04,.34),(.38,.11),"YES: use flat-ground\nCheeseman-Bennett estimate")
    box(ax,(.58,.34),(.38,.11),"NO: run MLP and\napply trust-band selection")
    box(ax,(.27,.15),(.46,.10),"Apply actuator correction\nthrottle_cmd = throttle_raw / sqrt(GE factor)")
    box(ax,(.23,.01),(.54,.07),"Corrected 8-seed AGL RMS:\nAnalytical 0.0632 m; ML hybrid 0.0644 m")
    arrow(ax,(.50,.88),(.50,.80)); arrow(ax,(.50,.72),(.50,.63))
    arrow(ax,(.36,.54),(.23,.45),"yes"); arrow(ax,(.64,.54),(.77,.45),"no")
    arrow(ax,(.23,.34),(.42,.25)); arrow(ax,(.77,.34),(.58,.25)); arrow(ax,(.50,.15),(.50,.08))
    fig.savefig(PAPER/"fig_diagram_mitigation_flow.png", dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def architecture():
    fig, ax = canvas((13, 7))
    box(ax,(.02,.80),(.24,.11),"Terrain generator\n(seed-controlled irregular terrain)", fs=10)
    box(ax,(.38,.80),(.24,.11),"Physics engine\nquadratic RPM--thrust x true GE", fs=10)
    box(ax,(.74,.80),(.24,.11),"Onboard sensors\nrangefinder, gyro, RPM/current", fs=10)
    box(ax,(.08,.46),(.25,.12),"Analytical estimator\nflat Cheeseman-Bennett only", fs=10)
    box(ax,(.39,.46),(.22,.12),"ML estimator (MLP)\n13 -> 64 -> 32 -> 1", fs=10)
    box(ax,(.70,.46),(.25,.12),"Flight controller\nPID + selected GE estimate", fs=10)
    box(ax,(.63,.16),(.30,.12),"Rotor throttle commands\nthrottle_raw / sqrt(GE estimate)", fs=10)
    arrow(ax,(.26,.855),(.38,.855)); arrow(ax,(.62,.855),(.74,.855))
    arrow(ax,(.80,.80),(.27,.58),"rangefinder")
    arrow(ax,(.88,.80),(.52,.58),"features")
    arrow(ax,(.33,.52),(.70,.52),"selected factor"); arrow(ax,(.61,.52),(.70,.52))
    arrow(ax,(.825,.46),(.78,.28)); arrow(ax,(.63,.22),(.50,.80),"closed loop",dashed=True)
    fig.savefig(PAPER/"fig_diagram_architecture.png", dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    architecture(); feedback(); mitigation()
    print(f"[diagrams] wrote corrected architecture/feedback/mitigation diagrams to {PAPER}")
