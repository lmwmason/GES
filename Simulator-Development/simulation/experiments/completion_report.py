#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def save(fig,path): fig.tight_layout(); fig.savefig(path,dpi=300); plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True); plt.rcParams.update({"font.size":9})
    d=pd.read_csv(a.root/"oracle_timing_30seed/per_seed_metrics.csv")
    s=d.groupby(["oracle_horizon","oracle_aggregation"]).agl_rms_primary.agg(["mean","sem"]).reset_index()
    fig,ax=plt.subplots()
    for name,g in s.groupby("oracle_aggregation"):
        ax.errorbar(
            g["oracle_horizon"] * 1000,
            g["mean"],
            yerr=1.96 * g["sem"],
            marker="o",
            capsize=3,
            label=name,
        )
    ax.set(xlabel="Oracle preview horizon (ms)",ylabel="AGL RMS (m)"); ax.legend(); save(fig,a.out_dir/"oracle_timing.png")

    d=pd.read_csv(a.root/"base_model_30seed_v2/per_seed_metrics.csv")
    s=d.groupby(["ge_base_model","mode"]).agl_rms_primary.mean().unstack()
    ax=s.plot.bar(y=["analytical","oracle","ml","ml_asym_clamp"],rot=0)
    ax.set(xlabel="Ground-effect base model",ylabel="AGL RMS (m)"); plt.tight_layout(); plt.savefig(a.out_dir/"base_model_sensitivity.png",dpi=300); plt.close()

    d=pd.read_csv(a.root/"frequency_stability_v2/delay_gain_margins.csv")
    fig,ax=plt.subplots()
    for factor,g in d[d.altitude_gain_scale==1].groupby("ge_factor"):
        ax.plot(g.delay_s*1000,g.minimum_phase_margin_deg,marker="o",label=f"GE={factor:g}")
    ax.axhline(0,color="k",lw=.8); ax.set(xlabel="Delay (ms)",ylabel="Minimum phase margin (deg)"); ax.legend(); save(fig,a.out_dir/"delay_phase_margin.png")

    d=pd.read_csv(a.root/"ml_active_pulse_analysis/pulse_response_per_seed.csv")
    s=d.groupby(["duration_s","bias"]).adverse_peak_clearance_m.mean().unstack()
    ax=s.plot.bar(rot=0); ax.set(xlabel="Pulse duration (s)",ylabel="Mean adverse peak clearance (m)")
    plt.tight_layout(); plt.savefig(a.out_dir/"active_ml_pulse_peak.png",dpi=300); plt.close()

    horizon=a.root/"horizon_summary/horizon_metrics.csv"
    if horizon.exists():
        d=pd.read_csv(horizon); fig,ax=plt.subplots()
        for model,g in d.groupby("model"): ax.plot(g.lead_s*1000,g.RMSE,marker="o",label=model)
        ax.set(xlabel="Prediction horizon (ms)",ylabel="Held-out-seed RMSE"); ax.legend(); save(fig,a.out_dir/"horizon_rmse.png")

    d=pd.read_csv(a.root/"terrain_decomposition/terrain_summary.csv")
    d=d[(d.metric=="rmse")&(d.clearance_band=="0.15-0.25")&(d.model.isin(["Analytical","MLP","LSTM","TRANSFORMER","Rangefinder_only_MLP"]))]
    ax=d.pivot(index="terrain_type",columns="model",values="mean").plot.bar(rot=0)
    ax.set(xlabel="Terrain class",ylabel="Seed-mean RMSE (0.15--0.25 m clearance)")
    plt.tight_layout(); plt.savefig(a.out_dir/"terrain_decomposition.png",dpi=300); plt.close()
if __name__=="__main__": main()
