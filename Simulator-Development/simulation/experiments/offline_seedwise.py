#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml")); sys.path.insert(0, str(ROOT))
from data import (BASELINE_COL, FEATURE_COLS, TARGET_COL, add_block_id,
                  build_common_support, lead_s_to_rows, load_dataset, seed_group_split)
from models_torch import LSTMNet, MLPNet, TransformerNet
from experiments.validity import load_config, write_json


def metrics(y, pred):
    return {"MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(mean_squared_error(y, pred) ** .5),
            "R2": float(r2_score(y, pred))}


class SeedWindowDataset(torch.utils.data.Dataset):
    """Lazy windows: stores each seed once rather than materializing N×W×F."""
    def __init__(self, df, scaler, window, lead_rows, feature_cols=FEATURE_COLS):
        self.window, self.lead_rows = window, lead_rows
        self.groups, self.ends = [], []
        total = 0
        for _, group in df.groupby("seed", sort=True):
            x = scaler.transform(group[feature_cols].to_numpy(np.float32)).astype(np.float32)
            y = group[TARGET_COL].to_numpy(np.float32)
            count = max(0, len(group) - window - lead_rows + 1)
            if count:
                self.groups.append((x, y, count)); total += count; self.ends.append(total)
    def __len__(self): return self.ends[-1] if self.ends else 0
    def __getitem__(self, idx):
        gi = bisect.bisect_right(self.ends, idx)
        base = 0 if gi == 0 else self.ends[gi - 1]
        x, y, _ = self.groups[gi]; start = idx - base
        return torch.from_numpy(x[start:start + self.window]), torch.tensor(y[start + self.window - 1 + self.lead_rows])


WINDOW = 32
LEAD_ROWS = 6


def analytical_from_clearance(z):
    z = np.maximum(np.asarray(z, dtype=float), 0.127 * .5)
    factor = 1.0 / (1.0 - (0.127 / (4.0 * z)) ** 2)
    return np.clip(factor, .85, 1.6)


def predictive_analytical(df, support, history_rows=6):
    """Constant-vertical-velocity rangefinder projection fixed a priori."""
    projected = np.empty(len(support["y"]), dtype=float)
    for seed, group in df.groupby("seed", sort=True):
        take = np.flatnonzero(support["seed"] == int(seed))
        indices = group.index.to_numpy(); positions = np.searchsorted(indices, support["source_index"][take])
        earlier = np.maximum(0, positions - history_rows)
        t = group["t"].to_numpy(float); z = group["rangefinder_alt"].to_numpy(float)
        dt = t[positions] - t[earlier]
        vz = np.divide(z[positions] - z[earlier], dt, out=np.zeros_like(dt), where=dt > 0)
        horizon = support["target_t"][take] - support["source_t"][take]
        projected[take] = z[positions] + vz * horizon
    return analytical_from_clearance(projected)


def train_torch(name, model, train_ds, val_ds, epochs, device):
    generator = torch.Generator().manual_seed(20260825)
    workers = min(4, max(0, (os.cpu_count() or 1) - 1))
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, generator=generator,
                                           num_workers=workers, persistent_workers=workers > 0)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=2048, num_workers=workers,
                                         persistent_workers=workers > 0)
    model.to(device); opt = torch.optim.Adam(model.parameters(), lr=1e-3); loss_fn = torch.nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device); opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        y, pred = predict_loader(model, val_dl, device); rmse = float(np.sqrt(np.mean((y-pred)**2)))
        history.append({"epoch": epoch+1, "validation_rmse": rmse}); print(f"[offline] {name} epoch {epoch+1}/{epochs} val_rmse={rmse:.6f}")
        if rmse < best - 1e-7:
            best, bad = rmse, 0; best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: bad += 1
        if bad >= 6: break
    if best_state: model.load_state_dict(best_state)
    return history


def predict_loader(model, loader, device):
    ys, ps = [], []; model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            ys.append(yb.numpy()); ps.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    global WINDOW, LEAD_ROWS
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset", help="override config dataset path (recorded in output metadata by the caller)")
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--models", default="all")
    ap.add_argument("--window", type=int, default=32); ap.add_argument("--lead-s", type=float, default=.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); WINDOW = args.window
    random.seed(20260825); np.random.seed(20260825); torch.manual_seed(20260825)
    cfg = load_config(Path(args.config)); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset) if args.dataset else ROOT / cfg["dataset"]
    digest = hashlib.sha256()
    with dataset_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""): digest.update(chunk)
    write_json(out / "metadata.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "dataset": str(dataset_path.resolve()),
        "dataset_sha256": digest.hexdigest(), "arguments": vars(args), "config": cfg,
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}})
    df = add_block_id(load_dataset(dataset_path)); LEAD_ROWS = lead_s_to_rows(df, args.lead_s)
    counts = cfg["offline_split"]
    train, val, test, split = seed_group_split(df, counts["train"], counts["validation"], counts["test"], cfg["random_seed"])
    write_json(out / "seed_split.json", split)
    selected = {"analytical","predictive_analytical","range_mlp","linear","tree","rf","gb","mlp","lstm","transformer"} if args.models == "all" else set(args.models.split(","))
    tr_support = build_common_support(train, WINDOW, LEAD_ROWS, extra_cols=[BASELINE_COL])
    va_support = build_common_support(val, WINDOW, LEAD_ROWS, extra_cols=[BASELINE_COL])
    te_support = build_common_support(test, WINDOW, LEAD_ROWS, extra_cols=[BASELINE_COL])
    Xtr, ytr = tr_support["X"], tr_support["y"]
    Xv, yv = va_support["X"], va_support["y"]
    Xte, yte, extra = te_support["X"], te_support["y"], te_support["extra"]
    results, predictions, artifacts = [], {}, {}
    def checkpoint():
        pd.DataFrame(results).sort_values("RMSE").to_csv(out/"offline_metrics.csv", index=False)
        write_json(out/"progress.json", {"completed_models": [r["model"] for r in results]})
    if "analytical" in selected:
        pred=extra[BASELINE_COL]; results.append({"model":"Analytical","train_s":0, "n_test":len(yte),**metrics(yte,pred)}); predictions["Analytical"]=pred; checkpoint()
    if "predictive_analytical" in selected:
        pred = predictive_analytical(test, te_support)
        results.append({"model":"Predictive analytical","train_s":0,"n_test":len(yte),**metrics(yte,pred)})
        predictions["Predictive_analytical"] = pred; checkpoint()
    classical = {
      "linear": LinearRegression(), "tree": DecisionTreeRegressor(max_depth=10,min_samples_leaf=5,random_state=20260825),
      "rf": RandomForestRegressor(n_estimators=100,max_depth=14,min_samples_leaf=3,
          max_samples=.5,n_jobs=-1,random_state=20260825),
      "gb": HistGradientBoostingRegressor(max_depth=6,max_iter=300,random_state=20260825)}
    scaler = StandardScaler().fit(Xtr)
    for key, model in classical.items():
        if key not in selected: continue
        a,b = (scaler.transform(Xtr),scaler.transform(Xte)) if key=="linear" else (Xtr,Xte)
        t=time.time(); model.fit(a,ytr); pred=model.predict(b); elapsed=time.time()-t
        results.append({"model":key,"train_s":elapsed,"n_test":len(yte),**metrics(yte,pred)}); predictions[key]=pred; artifacts[key]=model
        checkpoint(); print(f"[offline] {key} {results[-1]}")
    if "mlp" in selected:
        class PointDS(torch.utils.data.TensorDataset): pass
        tr=PointDS(torch.from_numpy(scaler.transform(Xtr).astype(np.float32)),torch.from_numpy(ytr)); va=PointDS(torch.from_numpy(scaler.transform(Xv).astype(np.float32)),torch.from_numpy(yv))
        te=PointDS(torch.from_numpy(scaler.transform(Xte).astype(np.float32)),torch.from_numpy(yte)); model=MLPNet(len(FEATURE_COLS)); t=time.time()
        hist=train_torch("MLP",model,tr,va,args.epochs,args.device); yt,pred=predict_loader(model,torch.utils.data.DataLoader(te,batch_size=2048),args.device)
        results.append({"model":"MLP","train_s":time.time()-t,"n_test":len(yt),**metrics(yt,pred)}); predictions["MLP"]=pred; artifacts["MLP_state"]={k:v.detach().cpu() for k,v in model.state_dict().items()}; write_json(out/"mlp_history.json",hist); checkpoint()
    if "range_mlp" in selected:
        range_cols = ["rangefinder_alt"]
        range_scaler = StandardScaler().fit(train[range_cols].to_numpy(np.float32))
        trr=SeedWindowDataset(train,range_scaler,WINDOW,LEAD_ROWS,range_cols)
        var=SeedWindowDataset(val,range_scaler,WINDOW,LEAD_ROWS,range_cols)
        ter=SeedWindowDataset(test,range_scaler,WINDOW,LEAD_ROWS,range_cols)
        class RangeMLP(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.net=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(WINDOW,64),torch.nn.ReLU(),torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
            def forward(self,x): return self.net(x).squeeze(-1)
        model=RangeMLP(); t=time.time(); hist=train_torch("range_mlp",model,trr,var,args.epochs,args.device)
        yt,pred=predict_loader(model,torch.utils.data.DataLoader(ter,batch_size=2048),args.device)
        np.testing.assert_allclose(yt, yte, rtol=0, atol=0, err_msg="range model test support differs")
        results.append({"model":"Rangefinder-only MLP","train_s":time.time()-t,"n_test":len(yt),**metrics(yt,pred)})
        predictions["Rangefinder_only_MLP"]=pred; artifacts["range_mlp_state"]={k:v.detach().cpu() for k,v in model.state_dict().items()}; artifacts["range_scaler"]=range_scaler
        write_json(out/"range_mlp_history.json",hist); checkpoint()
    sequence_scaler = None
    if selected & {"lstm", "transformer"}:
        sequence_scaler=StandardScaler().fit(train[FEATURE_COLS].to_numpy(np.float32))
        trw=SeedWindowDataset(train,sequence_scaler,WINDOW,LEAD_ROWS); vaw=SeedWindowDataset(val,sequence_scaler,WINDOW,LEAD_ROWS); tew=SeedWindowDataset(test,sequence_scaler,WINDOW,LEAD_ROWS)
    for key, model in (("lstm",LSTMNet(len(FEATURE_COLS))),("transformer",TransformerNet(len(FEATURE_COLS),WINDOW))):
        if key not in selected: continue
        t=time.time(); hist=train_torch(key,model,trw,vaw,args.epochs,args.device); yt,pred=predict_loader(model,torch.utils.data.DataLoader(tew,batch_size=2048),args.device)
        np.testing.assert_allclose(yt, yte, rtol=0, atol=0, err_msg=f"{key} test support differs")
        results.append({"model":key.upper(),"train_s":time.time()-t,"n_test":len(yt),**metrics(yt,pred)}); artifacts[f"{key}_state"]={k:v.detach().cpu() for k,v in model.state_dict().items()}; write_json(out/f"{key}_history.json",hist)
        pd.DataFrame({"seed":te_support["seed"], "source_t":te_support["source_t"],
                      "target_t":te_support["target_t"], "truth":yt,
                      "prediction":pred}).to_csv(out/f"predictions_{key}.csv.gz",index=False,compression="gzip")
        checkpoint()
    pd.DataFrame(results).sort_values("RMSE").to_csv(out/"offline_metrics.csv",index=False)
    point = {"seed":te_support["seed"],"source_t":te_support["source_t"],"target_t":te_support["target_t"],"truth": yte}; point.update({f"prediction_{k}":v for k,v in predictions.items()})
    pd.DataFrame(point).to_csv(out/"pointwise_predictions.csv.gz",index=False,compression="gzip")
    joblib.dump({"artifacts":artifacts,"pointwise_scaler":scaler,"sequence_scaler":sequence_scaler,
                 "mlp_model_state":artifacts.get("MLP_state"),"mlp_scaler":scaler,"features":FEATURE_COLS,
                 "common_support":{"window":WINDOW,"lead_rows":LEAD_ROWS,"n_test":len(yte)}},out/"models.joblib")
    print(pd.DataFrame(results).sort_values("RMSE").to_string(index=False))

if __name__ == "__main__": main()
