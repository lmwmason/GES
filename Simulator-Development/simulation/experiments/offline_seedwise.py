#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import random
import sys
import time
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
                  build_pointwise_xy, lead_s_to_rows, load_dataset, seed_group_split)
from models_torch import LSTMNet, MLPNet, TransformerNet
from experiments.validity import load_config, write_json


def metrics(y, pred):
    return {"MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(mean_squared_error(y, pred) ** .5),
            "R2": float(r2_score(y, pred))}


class SeedWindowDataset(torch.utils.data.Dataset):
    """Lazy windows: stores each seed once rather than materializing N×W×F."""
    def __init__(self, df, scaler, window, lead_rows):
        self.groups, self.ends = [], []
        total = 0
        for _, group in df.groupby("seed", sort=True):
            x = scaler.transform(group[FEATURE_COLS].to_numpy(np.float32)).astype(np.float32)
            y = group[TARGET_COL].to_numpy(np.float32)
            count = max(0, len(group) - window - lead_rows + 1)
            if count:
                self.groups.append((x, y, count)); total += count; self.ends.append(total)
    def __len__(self): return self.ends[-1] if self.ends else 0
    def __getitem__(self, idx):
        gi = bisect.bisect_right(self.ends, idx)
        base = 0 if gi == 0 else self.ends[gi - 1]
        x, y, _ = self.groups[gi]; start = idx - base
        return torch.from_numpy(x[start:start + WINDOW]), torch.tensor(y[start + WINDOW - 1 + LEAD_ROWS])


WINDOW = 32
LEAD_ROWS = 6


def train_torch(name, model, train_ds, val_ds, epochs, device):
    generator = torch.Generator().manual_seed(20260825)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, generator=generator)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=2048)
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
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--models", default="all")
    ap.add_argument("--window", type=int, default=32); ap.add_argument("--lead-s", type=float, default=.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); WINDOW = args.window
    random.seed(20260825); np.random.seed(20260825); torch.manual_seed(20260825)
    cfg = load_config(Path(args.config)); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = add_block_id(load_dataset(ROOT / cfg["dataset"])); LEAD_ROWS = lead_s_to_rows(df, args.lead_s)
    counts = cfg["offline_split"]
    train, val, test, split = seed_group_split(df, counts["train"], counts["validation"], counts["test"], cfg["random_seed"])
    write_json(out / "seed_split.json", split)
    selected = {"analytical","linear","tree","rf","gb","mlp","lstm","transformer"} if args.models == "all" else set(args.models.split(","))
    Xtr, ytr, _, _ = build_pointwise_xy(train, LEAD_ROWS); Xv, yv, _, _ = build_pointwise_xy(val, LEAD_ROWS)
    Xte, yte, _, extra = build_pointwise_xy(test, LEAD_ROWS, extra_cols=[BASELINE_COL])
    results, predictions, artifacts = [], {}, {}
    def checkpoint():
        pd.DataFrame(results).sort_values("RMSE").to_csv(out/"offline_metrics.csv", index=False)
        write_json(out/"progress.json", {"completed_models": [r["model"] for r in results]})
    if "analytical" in selected:
        pred=extra[BASELINE_COL]; results.append({"model":"Analytical","train_s":0, "n_test":len(yte),**metrics(yte,pred)}); predictions["Analytical"]=pred; checkpoint()
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
        results.append({"model":"MLP","train_s":time.time()-t,"n_test":len(yt),**metrics(yt,pred)}); predictions["MLP"]=pred; artifacts["MLP_state"]=model.state_dict(); write_json(out/"mlp_history.json",hist); checkpoint()
    sequence_scaler = None
    if selected & {"lstm", "transformer"}:
        sequence_scaler=StandardScaler().fit(train[FEATURE_COLS].to_numpy(np.float32))
        trw=SeedWindowDataset(train,sequence_scaler,WINDOW,LEAD_ROWS); vaw=SeedWindowDataset(val,sequence_scaler,WINDOW,LEAD_ROWS); tew=SeedWindowDataset(test,sequence_scaler,WINDOW,LEAD_ROWS)
    for key, model in (("lstm",LSTMNet(len(FEATURE_COLS))),("transformer",TransformerNet(len(FEATURE_COLS),WINDOW))):
        if key not in selected: continue
        t=time.time(); hist=train_torch(key,model,trw,vaw,args.epochs,args.device); yt,pred=predict_loader(model,torch.utils.data.DataLoader(tew,batch_size=2048),args.device)
        results.append({"model":key.upper(),"train_s":time.time()-t,"n_test":len(yt),**metrics(yt,pred)}); artifacts[f"{key}_state"]=model.state_dict(); write_json(out/f"{key}_history.json",hist)
        pd.DataFrame({"truth":yt,"prediction":pred}).to_csv(out/f"predictions_{key}.csv.gz",index=False,compression="gzip")
        checkpoint()
    pd.DataFrame(results).sort_values("RMSE").to_csv(out/"offline_metrics.csv",index=False)
    point = {"truth": yte}; point.update({f"prediction_{k}":v for k,v in predictions.items()})
    pd.DataFrame(point).to_csv(out/"pointwise_predictions.csv.gz",index=False,compression="gzip")
    joblib.dump({"artifacts":artifacts,"pointwise_scaler":scaler,"sequence_scaler":sequence_scaler,"features":FEATURE_COLS},out/"models.joblib")
    print(pd.DataFrame(results).sort_values("RMSE").to_string(index=False))

if __name__ == "__main__": main()
