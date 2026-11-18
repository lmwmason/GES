#!/usr/bin/env python3
"""Paired response analysis for actively deployed learned-controller pulses."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-dir",type=Path,required=True); p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_json(a.run_dir/"run_manifest.json")
    frames={(int(r.seed),float(r.pulse_duration),float(r.prediction_bias)):pd.read_csv(r.raw_csv,compression="gzip") for _,r in manifest.iterrows()}
    rows=[]; windows=[]
    for seed in sorted(manifest.seed.unique()):
      for duration in sorted(manifest.pulse_duration.unique()):
        control=frames[(int(seed),float(duration),0.)]
        start=float(manifest.pulse_start.iloc[0]); end=start+float(duration)
        for bias in (-.02,-.01,.01,.02):
          treated=frames[(int(seed),float(duration),bias)]
          if not np.allclose(treated.t,control.t): raise ValueError("unaligned trajectories")
          time=treated.t.to_numpy(); post=time>=end; pt=time[post]-end
          dc=(treated.min_rotor_clearance-control.min_rotor_clearance).to_numpy()[post]
          natural_t=np.abs(treated.ge_ff_ai-treated.ge_factor_truth_mean).to_numpy()[post]
          natural_c=np.abs(control.ge_ff_ai-control.ge_factor_truth_mean).to_numpy()[post]
          de=natural_t-natural_c
          adverse=-dc if bias>0 else dc
          peak_i=int(np.argmax(adverse)); peak=float(adverse[peak_i]); threshold=max(.1*abs(peak),1e-5)
          settle=np.nan
          for i in range(peak_i,len(adverse)-60):
            if np.all(np.abs(dc[i:i+60])<=threshold): settle=float(pt[i]); break
          crossings=int(np.sum(np.diff(np.signbit(dc[peak_i:]))!=0))
          rows.append({"seed":seed,"duration_s":duration,"bias":bias,"adverse_peak_clearance_m":peak,
                       "peak_time_s":float(pt[peak_i]),"settling_time_s":settle,"zero_crossings_after_peak":crossings,
                       "post10s_clearance_area_m_s":float(np.trapezoid(dc[pt<=10],pt[pt<=10])),
                       "post10s_natural_abs_error_delta":float(de[pt<=10].mean())})
          for lo,hi in ((0,.5),(.5,1),(1,2),(2,5),(5,10)):
            mask=(pt>=lo)&(pt<hi)
            windows.append({"seed":seed,"duration_s":duration,"bias":bias,"window_start_s":lo,"window_end_s":hi,
                            "clearance_delta_m":float(dc[mask].mean()),"natural_abs_error_delta":float(de[mask].mean())})
    raw=pd.DataFrame(rows); raw.to_csv(a.out_dir/"pulse_response_per_seed.csv",index=False)
    win=pd.DataFrame(windows); win.to_csv(a.out_dir/"pulse_windows_per_seed.csv",index=False)
    summary=[]
    for keys,g in win.groupby(["duration_s","bias","window_start_s","window_end_s"]):
      for metric in ("clearance_delta_m","natural_abs_error_delta"):
        x=g[metric].to_numpy(); sem=stats.sem(x); lo,hi=stats.t.interval(.95,len(x)-1,loc=x.mean(),scale=sem)
        summary.append({"duration_s":keys[0],"bias":keys[1],"window_start_s":keys[2],"window_end_s":keys[3],
                        "metric":metric,"mean":x.mean(),"ci_low":lo,"ci_high":hi,"p":stats.ttest_1samp(x,0).pvalue,"n":len(x)})
    pd.DataFrame(summary).to_csv(a.out_dir/"pulse_window_summary.csv",index=False)
if __name__=="__main__": main()
