#!/usr/bin/env python3
"""Decompose common-support test errors by preregistered terrain generator type."""
from __future__ import annotations
import argparse, bisect, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path[:0]=[str(ROOT),str(ROOT/"viz")]
from terrain_ref import BUMP,DIP,RIPPLE,generate_features

NAMES={-1:"flat",BUMP:"bump",DIP:"dip",RIPPLE:"ripple"}

def classify(seed,x,cache):
    if seed not in cache:
        features=generate_features(int(seed)); cache[seed]=(features,[f[1] for f in features])
    features,starts=cache[seed]; i=bisect.bisect_right(starts,float(x))-1
    if i<0: return "flat"
    f=features[i]
    return NAMES[f[0]] if f[1] <= x <= f[1]+f[2] else "flat"

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--offline-dir",type=Path,required=True); p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=20000); p.add_argument("--seed",type=int,default=20260825)
    a=p.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    split=json.loads((a.offline_dir/"seed_split.json").read_text()); test=set(map(int,split["test"]))
    pred=pd.read_csv(a.offline_dir/"pointwise_predictions.csv.gz")
    if set(map(int,pred.seed.unique())) != test: raise ValueError("prediction seeds do not equal recorded test seeds")
    telemetry=pd.read_csv(a.dataset,usecols=["seed","t","x","y","rangefinder_alt"])
    telemetry=telemetry[telemetry.seed.isin(test)].copy()
    merged=pred.merge(telemetry,left_on=["seed","source_t"],right_on=["seed","t"],how="left",validate="one_to_one")
    if merged.x.isna().any(): raise ValueError("failed to align prediction rows with telemetry")
    for name in ("lstm","transformer"):
        path=a.offline_dir/f"predictions_{name}_auditable.csv.gz"
        seq=pd.read_csv(path)
        if not np.array_equal(seq.seed.to_numpy(),merged.seed.to_numpy()) or not np.allclose(seq.target_t,merged.target_t):
            raise ValueError(f"{name} support mismatch")
        merged[f"prediction_{name.upper()}"]=seq.prediction.to_numpy()
    cache={}; merged["terrain_type"]=[classify(s,x,cache) for s,x in zip(merged.seed,merged.x)]
    merged["clearance_band"]=pd.cut(merged.rangefinder_alt,[-np.inf,.15,.25,.40,np.inf],
        labels=["<=0.15","0.15-0.25","0.25-0.40",">0.40"],include_lowest=True).astype(str)
    prediction_cols=[c for c in merged if c.startswith("prediction_")]
    per=[]
    for (terrain,band,seed),g in merged.groupby(["terrain_type","clearance_band","seed"]):
        for col in prediction_cols:
            e=g[col]-g.truth
            per.append({"terrain_type":terrain,"clearance_band":band,"seed":seed,"model":col[11:],
                        "n":len(g),"rmse":float(np.sqrt(np.mean(e*e))),"mae":float(np.mean(np.abs(e)))})
    per=pd.DataFrame(per); per.to_csv(a.out_dir/"per_seed_terrain_metrics.csv",index=False)
    rng=np.random.default_rng(a.seed); rows=[]
    for keys,g in per.groupby(["terrain_type","clearance_band","model"]):
        for metric in ("rmse","mae"):
            values=g[metric].to_numpy(); boot=values[rng.integers(0,len(values),(a.bootstrap,len(values)))].mean(1)
            rows.append({"terrain_type":keys[0],"clearance_band":keys[1],"model":keys[2],"metric":metric,
                         "n_seeds":len(values),"mean":values.mean(),"ci_low":np.quantile(boot,.025),"ci_high":np.quantile(boot,.975)})
    pd.DataFrame(rows).to_csv(a.out_dir/"terrain_summary.csv",index=False)
    merged.to_csv(a.out_dir/"classified_predictions.csv.gz",index=False,compression="gzip")
    (a.out_dir/"config.json").write_text(json.dumps({"test_seeds":sorted(test),"bootstrap":a.bootstrap,"seed":a.seed,
        "terrain_classes":["flat","bump","dip","ripple"],"clearance_edges_m":[.15,.25,.40]},indent=2)+"\n")
if __name__=="__main__": main()
