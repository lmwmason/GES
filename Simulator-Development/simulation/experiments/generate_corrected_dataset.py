#!/usr/bin/env python3
"""Generate immutable per-seed telemetry in parallel and merge deterministically."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out-dir",required=True); p.add_argument("--start-seed",type=int,default=1)
    p.add_argument("--n-seeds",type=int,default=300); p.add_argument("--duration",type=float,default=60); p.add_argument("--workers",type=int,default=min(32,os.cpu_count() or 1))
    p.add_argument("--mode",default="analytical",choices=["none","analytical"]); a=p.parse_args()
    out=Path(a.out_dir).resolve(); episodes=out/"episodes"
    if out.exists() and any(out.iterdir()): raise SystemExit(f"refusing to overwrite non-empty output: {out}")
    episodes.mkdir(parents=True); build=ROOT/"build"
    subprocess.run(["cmake","-S",str(ROOT),"-B",str(build),"-DCMAKE_BUILD_TYPE=RelWithDebInfo"],check=True)
    subprocess.run(["cmake","--build",str(build),"-j",str(os.cpu_count() or 1)],check=True)
    simulator=build/"bin/simulator"; fc=build/"bin/libfc.so"; seeds=list(range(a.start_seed,a.start_seed+a.n_seeds)); started=time.time()
    def run(seed):
        dest=episodes/f"seed_{seed:04d}.csv"; env=os.environ.copy(); env["GE_FF_MODE"]=a.mode
        cmd=[str(simulator),"--fc",str(fc),"--seed",str(seed),"--duration",str(a.duration),"--fast","--no-telemetry","--csv",str(dest)]
        proc=subprocess.run(cmd,env=env,text=True,capture_output=True)
        if proc.returncode or not dest.exists() or dest.stat().st_size==0: raise RuntimeError(f"seed {seed}: rc={proc.returncode} stderr={proc.stderr[-500:]}")
        return seed,dest,sha256(dest)
    records=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(run,s):s for s in seeds}
        for i,f in enumerate(as_completed(futures),1):
            seed,path,digest=f.result(); records.append({"seed":seed,"file":str(path.relative_to(out)),"sha256":digest})
            if i%10==0 or i==len(seeds): print(f"[dataset] {i}/{len(seeds)} complete",flush=True)
    merged=out/"dataset_all.csv"
    with merged.open("w",newline="") as dst:
        writer=None
        for rec in sorted(records,key=lambda x:x["seed"]):
            with (out/rec["file"]).open(newline="") as src:
                reader=csv.reader(src); header=next(reader)
                if writer is None: writer=csv.writer(dst); writer.writerow(header)
                for row in reader: writer.writerow(row)
    try:
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        git_head=os.environ.get("GES_SOURCE_REVISION","unavailable-exported-tree")
    manifest={"schema_version":1,"created_utc":datetime.now(timezone.utc).isoformat(),"purpose":"reviewer corrected-cruise-dynamics regeneration",
              "precommitted":{"seeds":seeds,"duration_s":a.duration,"controller_mode":a.mode,"workers":a.workers},
              "runtime":{"python":sys.version,"platform":platform.platform()},"source":{"git_head":git_head},
              "episodes":sorted(records,key=lambda x:x["seed"]),"merged":{"file":"dataset_all.csv","sha256":sha256(merged)},"elapsed_s":time.time()-started}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"[dataset] wrote {merged} sha256={manifest['merged']['sha256']}")
if __name__=="__main__": main()
