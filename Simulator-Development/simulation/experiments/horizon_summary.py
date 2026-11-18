#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True); rows=[]; splits=[]
    for path in sorted(a.root.glob("horizon_*s")):
        lead=float(path.name.removeprefix("horizon_").removesuffix("s"))
        metrics=pd.read_csv(path/"offline_metrics.csv"); metrics["lead_s"]=lead; rows.append(metrics)
        splits.append(json.loads((path/"seed_split.json").read_text()))
    if not rows: raise ValueError("no completed horizon directories")
    if any(s!=splits[0] for s in splits[1:]): raise ValueError("horizon seed splits differ")
    out=pd.concat(rows,ignore_index=True); out.to_csv(a.out_dir/"horizon_metrics.csv",index=False)
    (a.out_dir/"seed_split.json").write_text(json.dumps(splits[0],indent=2)+"\n")
    fig,ax=plt.subplots()
    for model,g in out.groupby("model"):
        ax.plot(g.lead_s*1000,g.RMSE,marker="o",label=model)
    ax.set(xlabel="Prediction horizon (ms)",ylabel="Held-out-seed RMSE"); ax.legend(); fig.tight_layout()
    fig.savefig(a.out_dir/"horizon_rmse.png",dpi=300); plt.close(fig)
if __name__=="__main__": main()
