#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml")); sys.path.insert(0, str(ROOT))
from data import add_block_id, load_dataset, seed_group_split
from experiments.validity import (load_config, paired_statistics, run_simulation,
                                  runtime_metadata, utc_run_id, write_json)


def prepare_run(args, cfg):
    out = ROOT / args.out_root / (args.run_id or utc_run_id(args.command))
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", cfg)
    write_json(out / "metadata.json", runtime_metadata(ROOT, cfg))
    return out


def split_command(cfg, out):
    df = add_block_id(load_dataset(ROOT / cfg["dataset"]))
    counts = cfg["offline_split"]
    train, val, test, seeds = seed_group_split(
        df, counts["train"], counts["validation"], counts["test"], cfg["random_seed"])
    write_json(out / "seed_split.json", seeds)
    pd.DataFrame([(name, seed) for name, values in seeds.items() for seed in values],
                 columns=["split", "seed"]).to_csv(out / "seed_split.csv", index=False)
    pd.DataFrame({"split": ["train", "validation", "test"],
                  "rows": [len(train), len(val), len(test)],
                  "seeds": [train.seed.nunique(), val.seed.nunique(), test.seed.nunique()]}).to_csv(
                      out / "split_summary.csv", index=False)
    print(json.dumps(seeds, indent=2))


def run_records(cfg, out, specs, workers=1):
    records = []
    def execute(spec):
        print(f"[experiment] {spec}", flush=True)
        return run_simulation(ROOT, out, cfg=cfg, **spec)
    if workers == 1:
        completed = map(execute, specs)
    else:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        completed = (future.result() for future in
                     concurrent.futures.as_completed([pool.submit(execute, spec) for spec in specs]))
    for record in completed:
        records.append(record)
        pd.DataFrame(records).to_csv(out / "per_seed_metrics.csv", index=False)
        write_json(out / "run_manifest.json", records)
    if workers != 1:
        pool.shutdown()
    records.sort(key=lambda row: (row["slope_exponent"], row["curvature_gain"],
                                  row["seed"], row["mode"]))
    pd.DataFrame(records).to_csv(out / "per_seed_metrics.csv", index=False)
    write_json(out / "run_manifest.json", records)
    failures = [r for r in records if r["status"] != "complete"]
    if failures:
        write_json(out / "failures.json", failures)
        raise RuntimeError(f"{len(failures)} simulation runs failed; see {out/'failures.json'}")
    metrics = pd.DataFrame(records)
    comparison = metrics.copy()
    if "oracle_aggregation" in comparison and comparison.oracle_aggregation.nunique() > 1:
        comparison["mode"] = comparison["mode"] + "_" + comparison["oracle_aggregation"]
    if "oracle_horizon" in comparison and comparison.oracle_horizon.nunique() > 1:
        comparison["mode"] = comparison["mode"] + "_h" + comparison["oracle_horizon"].map(lambda x:f"{x:.3f}")
    if "prediction_bias" in comparison and comparison.prediction_bias.nunique() > 1:
        comparison["mode"] = comparison["mode"] + "_pb" + comparison["prediction_bias"].map(lambda x:f"{x:+.3f}")
    if "pulse_duration" in comparison and comparison.pulse_duration.nunique() > 1:
        comparison["mode"] = comparison["mode"] + "_d" + comparison["pulse_duration"].map(lambda x:f"{x:.4f}")
    paired_statistics(comparison, cfg).to_csv(out / "paired_statistics.csv", index=False)
    metrics.groupby("mode").agg({
        "agl_rms_primary": ["mean", "std"], "low_clearance_duration_s": ["mean", "std"],
        "low_clearance_events": ["mean", "std"], "gyro_rms": ["mean", "std"],
        "minimum_clearance_m": ["mean", "std"], "p05_clearance_m": ["mean", "std"]
    }).to_csv(out / "summary_by_mode.csv")
    return metrics


def closed_loop_specs(cfg, validation=False, clamp_delta=None):
    seeds = cfg["controller_validation_seeds"] if validation else cfg["controller_test_seeds"]
    modes = cfg["controller_modes"] + (["ml_asym_clamp"] if clamp_delta is not None else [])
    return [{"mode": mode, "seed": seed, "clamp_delta": clamp_delta} for seed in seeds for mode in modes]


def clamp_command(cfg, out, workers):
    rows = []
    for delta in cfg["asymmetric_clamp_deltas"]:
        sub = out / f"delta_{delta:g}"; sub.mkdir(parents=True, exist_ok=True)
        metrics = run_records(cfg, sub, [{"mode": "ml_asym_clamp", "seed": seed, "clamp_delta": delta}
                                         for seed in cfg["controller_validation_seeds"]], workers=workers)
        rows.append({"delta": delta,
                     "agl_rms_primary": metrics.agl_rms_primary.mean(),
                     "low_clearance_duration_s": metrics.low_clearance_duration_s.mean()})
    table = pd.DataFrame(rows).sort_values(
        ["agl_rms_primary", "low_clearance_duration_s", "delta"], ascending=[True, True, True])
    table.to_csv(out / "clamp_validation.csv", index=False)
    selected = float(table.iloc[0].delta)
    write_json(out / "selected_clamp.json", {"delta": selected,
        "selection_rule": "minimum mean validation AGL RMS; tie-break violation duration then tighter cap"})
    print(f"[experiment] selected asymmetric clamp delta={selected:g}")


def bias_command(cfg, out, workers):
    specs = [{"mode": "oracle", "seed": seed, "oracle_bias": bias}
             for seed in cfg["controller_test_seeds"] for bias in cfg["oracle_biases"]]
    metrics = run_records(cfg, out, specs, workers=workers)
    bias_modes = metrics.copy()
    bias_modes["mode"] = bias_modes["oracle_bias"].map(lambda value: f"bias_{value:+.2f}")
    paired_statistics(bias_modes, cfg).to_csv(out / "bias_paired_statistics.csv", index=False)
    metrics.groupby("oracle_bias").agg({"agl_rms_primary": ["mean", "std"],
        "low_clearance_duration_s": ["mean", "std"], "minimum_clearance_m": ["mean", "std"],
        "p05_clearance_m": ["mean", "std"]}).to_csv(out / "bias_summary.csv")
    pairs = []
    for magnitude in (0.01, 0.02):
        pos = metrics[metrics["oracle_bias"] == magnitude]
        neg = metrics[metrics["oracle_bias"] == -magnitude]
        merged = pos.merge(neg, on="seed", suffixes=("_positive", "_negative"))
        for metric in ("agl_rms_primary", "low_clearance_duration_s", "minimum_clearance_m", "p05_clearance_m"):
            d = merged[f"{metric}_positive"] - merged[f"{metric}_negative"]
            pairs.append({"bias_magnitude": magnitude, "metric": metric,
                          "mean_positive_minus_negative": d.mean(), "n": len(d)})
    pd.DataFrame(pairs).to_csv(out / "symmetric_bias_pairs.csv", index=False)


def pulse_command(cfg, out, workers):
    pulse = cfg["oracle_pulse"]
    specs = [{"mode": "oracle", "seed": seed, "oracle_bias": bias,
              "pulse_start": pulse["start_s"], "pulse_duration": pulse["duration_s"]}
             for seed in cfg["controller_test_seeds"] for bias in pulse["biases"]]
    run_records(cfg, out, specs, workers=workers)


def ml_pulse_command(cfg,out,workers):
    pulse=cfg["ml_pulse_grid"]
    specs=[{"mode":"ml","seed":seed,"prediction_bias":bias,
            "pulse_start":pulse["start_s"],"pulse_duration":duration}
           for seed in cfg["controller_test_seeds"] for duration in pulse["durations_s"]
           for bias in pulse["biases"]]
    run_records(cfg,out,specs,workers=workers)


def robustness_command(cfg, out, clamp_delta, workers):
    specs = []
    for slope in cfg["robustness"]["slope_exponents"]:
        for curvature in cfg["robustness"]["curvature_gains"]:
            for seed in cfg["controller_test_seeds"]:
                for mode in cfg["controller_modes"] + ["ml_asym_clamp"]:
                    specs.append({"mode": mode, "seed": seed, "slope_exponent": slope,
                                  "curvature_gain": curvature,
                                  "clamp_delta": clamp_delta if mode == "ml_asym_clamp" else None})
    metrics = run_records(cfg, out, specs, workers=workers)
    metrics.groupby(["slope_exponent", "curvature_gain", "mode"]).agg({
        "agl_rms_primary": ["mean", "std"], "low_clearance_duration_s": ["mean", "std"],
        "minimum_clearance_m": ["mean", "std"]}).to_csv(out / "robustness_summary.csv")


def bandwidth_command(cfg, out, workers):
    modes = ["none", "analytical", "oracle", "ml"]
    specs = [{"mode": mode, "seed": seed, "altitude_gain_scale": scale}
             for scale in cfg["feedback_bandwidth_gain_scales"]
             for seed in cfg["controller_test_seeds"] for mode in modes]
    metrics = run_records(cfg, out, specs, workers=workers)
    metrics.groupby(["altitude_gain_scale", "mode"]).agg({
        "agl_rms_primary": ["mean", "std"], "minimum_clearance_m": ["mean", "std"],
        "p05_clearance_m": ["mean", "std"]}).to_csv(out / "bandwidth_summary.csv")


def oracle_aggregation_command(cfg, out, workers):
    specs = [{"mode":"oracle", "seed":seed, "oracle_aggregation":aggregation}
             for seed in cfg["controller_test_seeds"]
             for aggregation in ("arithmetic", "thrust-weighted")]
    metrics = run_records(cfg, out, specs, workers=workers)
    metrics.groupby("oracle_aggregation").agg({
        "agl_rms_primary":["mean","std"], "minimum_clearance_m":["mean","std"],
        "p05_clearance_m":["mean","std"]}).to_csv(out/"oracle_aggregation_summary.csv")


def oracle_timing_command(cfg, out, workers):
    specs = [{"mode":"oracle", "seed":seed, "oracle_horizon":horizon,
              "oracle_aggregation":aggregation}
             for seed in cfg["controller_test_seeds"]
             for horizon in (0.0, 0.1)
             for aggregation in ("arithmetic", "thrust-weighted")]
    metrics = run_records(cfg, out, specs, workers=workers)
    metrics.groupby(["oracle_horizon","oracle_aggregation"]).agg({
        "agl_rms_primary":["mean","std"], "minimum_clearance_m":["mean","std"],
        "p05_clearance_m":["mean","std"]}).to_csv(out/"oracle_timing_summary.csv")


def base_model_command(cfg, out, workers):
    specs = [{"mode":mode,"seed":seed,"ge_base_model":base,
              "clamp_delta":0.0 if mode == "ml_asym_clamp" else None}
             for base in ("cheeseman","hayden")
             for seed in cfg["controller_test_seeds"]
             for mode in cfg["controller_modes"] + ["ml_asym_clamp"]]
    metrics=run_records(cfg,out,specs,workers=workers)
    metrics.groupby(["ge_base_model","mode"]).agg({
        "agl_rms_primary":["mean","std"],"minimum_clearance_m":["mean","std"],
        "p05_clearance_m":["mean","std"],"low_clearance_duration_s":["mean","std"]
    }).to_csv(out/"base_model_summary.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["split", "offline", "clamp-validation", "closed-loop", "bias", "pulse", "ml-pulse", "robustness", "bandwidth", "oracle-aggregation", "oracle-timing", "base-model"])
    ap.add_argument("--config", default="experiments/configs/ijira_validity.json")
    ap.add_argument("--out-root", default="experiments/results/ijira_validity")
    ap.add_argument("--run-id")
    ap.add_argument("--clamp-delta", type=float)
    ap.add_argument("--offline-epochs", type=int, default=25)
    ap.add_argument("--offline-models", default="all")
    ap.add_argument("--workers", type=int, default=1,
                    help="independent simulator workers (robustness only)")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    out = prepare_run(args, cfg)
    if args.command == "split": split_command(cfg, out)
    elif args.command == "offline":
        cmd = [sys.executable, str(ROOT / "experiments/offline_seedwise.py"), "--config", str(ROOT / args.config),
               "--out-dir", str(out), "--epochs", str(args.offline_epochs), "--models", args.offline_models]
        subprocess.run(cmd, cwd=ROOT, check=True)
    elif args.command == "clamp-validation": clamp_command(cfg, out, args.workers)
    elif args.command == "closed-loop":
        run_records(cfg, out, closed_loop_specs(cfg, clamp_delta=args.clamp_delta), workers=args.workers)
    elif args.command == "bias": bias_command(cfg, out, args.workers)
    elif args.command == "pulse": pulse_command(cfg, out, args.workers)
    elif args.command == "ml-pulse": ml_pulse_command(cfg,out,args.workers)
    elif args.command == "robustness":
        if args.clamp_delta is None: ap.error("robustness requires frozen --clamp-delta")
        robustness_command(cfg, out, args.clamp_delta, args.workers)
    elif args.command == "bandwidth": bandwidth_command(cfg, out, args.workers)
    elif args.command == "oracle-aggregation": oracle_aggregation_command(cfg, out, args.workers)
    elif args.command == "oracle-timing": oracle_timing_command(cfg, out, args.workers)
    elif args.command == "base-model": base_model_command(cfg, out, args.workers)
    print(f"[experiment] outputs: {out}")


if __name__ == "__main__": main()
