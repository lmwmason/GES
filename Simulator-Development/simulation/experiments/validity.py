from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REQUIRED_TELEMETRY = {
    "seed", "t", "gyro_x", "gyro_y", "gyro_z", "true_agl",
    "min_rotor_clearance", "ge_ff_applied", "ge_oracle_truth"
}


def load_config(path: Path) -> dict:
    with path.open() as f:
        cfg = json.load(f)
    required = {"random_seed", "controller_validation_seeds", "controller_test_seeds",
                "duration_s", "analysis_start_s", "target_agl_m", "low_clearance_m"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"configuration missing keys: {sorted(missing)}")
    validation = list(map(int, cfg["controller_validation_seeds"]))
    test = list(map(int, cfg["controller_test_seeds"]))
    offline_count = sum(map(int, cfg.get("offline_split", {}).values()))
    if len(validation) != len(set(validation)) or len(test) != len(set(test)):
        raise ValueError("controller seed lists contain duplicates")
    if set(validation) & set(test):
        raise ValueError("controller validation/test seed lists overlap")
    if any(seed <= offline_count for seed in validation + test):
        raise ValueError("controller seeds overlap the configured offline seed domain")
    return cfg


def utc_run_id(prefix: str) -> str:
    return datetime.now(timezone.utc).strftime(f"%Y%m%dT%H%M%SZ_{prefix}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_metadata(root: Path, config: dict) -> dict:
    def cmd(*args):
        return subprocess.run(args, cwd=root, text=True, capture_output=True, check=False).stdout.strip()
    versions = {}
    for name in ("numpy", "pandas", "scipy", "sklearn", "torch", "matplotlib"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"unavailable: {exc}"
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": cmd("git", "rev-parse", "HEAD"),
        "git_branch": cmd("git", "branch", "--show-current"),
        "git_status_short": cmd("git", "status", "--short"),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "config": config,
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_telemetry(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        df = pd.read_csv(path, compression="infer")
    except Exception as exc:
        raise ValueError(f"malformed telemetry {path}: {exc}") from exc
    missing = REQUIRED_TELEMETRY - set(df.columns)
    if missing:
        raise ValueError(f"telemetry {path} missing columns: {sorted(missing)}")
    if df.empty or df[list(REQUIRED_TELEMETRY - {"seed"})].apply(pd.to_numeric, errors="coerce").isna().any().any():
        raise ValueError(f"telemetry {path} is empty or contains malformed required numeric data")
    if not df["t"].is_monotonic_increasing:
        raise ValueError(f"telemetry {path} timestamps are not monotonic")
    return df


def compute_metrics(df: pd.DataFrame, cfg: dict) -> dict:
    active = df[df["t"] >= cfg["analysis_start_s"]].copy()
    if active.empty:
        raise ValueError("no samples remain after analysis_start_s")
    agl = active["true_agl"].to_numpy(float)
    clearance = active["min_rotor_clearance"].to_numpy(float)
    gyro = np.sqrt((active[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(float) ** 2).sum(axis=1))
    low = clearance < cfg["low_clearance_m"]
    transitions = np.diff(np.r_[False, low].astype(int)) == 1
    dt = float(np.median(np.diff(active["t"]))) if len(active) > 1 else 1 / 60
    err = agl - cfg["target_agl_m"]
    full_err = df["true_agl"].to_numpy(float) - cfg["target_agl_m"]
    return {
        "agl_rms_primary": float(np.sqrt(np.mean(err ** 2))),
        "low_clearance_duration_s": float(low.sum() * dt),
        "low_clearance_events": int(transitions.sum()),
        "gyro_rms": float(np.sqrt(np.mean(gyro ** 2))),
        "minimum_clearance_m": float(clearance.min()),
        "p05_clearance_m": float(np.quantile(clearance, 0.05)),
        "legacy_full_agl_rms": float(np.sqrt(np.mean(full_err ** 2))),
        "n_samples_primary": int(len(active)),
    }


def run_simulation(root: Path, out_dir: Path, *, mode: str, seed: int, cfg: dict,
                   oracle_bias=0.0, slope_exponent=1.0, curvature_gain=0.6,
                   clamp_delta=None, pulse_start=None, pulse_duration=None,
                   altitude_gain_scale=1.0, oracle_aggregation="arithmetic",
                   oracle_horizon=0.0, ge_base_model="cheeseman",
                   prediction_bias=0.0) -> dict:
    raw_dir = out_dir / "raw"
    log_dir = out_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
    pulse_tag = "" if pulse_start is None else f"_p{pulse_start:.2f}x{pulse_duration:.2f}"
    aggregation_tag = "_ow" if oracle_aggregation == "thrust-weighted" else ""
    horizon_tag = f"_h{oracle_horizon:.3f}_{ge_base_model}_pb{prediction_bias:+.3f}"
    tag = f"seed{seed}_{mode}{aggregation_tag}{horizon_tag}_b{oracle_bias:+.3f}_s{slope_exponent:.2f}_c{curvature_gain:.2f}{pulse_tag}"
    tmp_csv = raw_dir / f"{tag}.csv"
    gz_csv = raw_dir / f"{tag}.csv.gz"
    cmd = [str(root / "build/bin/simulator"), "--fc", str(root / "build/bin/libfc.so"),
           "--seed", str(seed), "--duration", str(cfg["duration_s"]), "--fast", "--no-telemetry",
           "--csv", str(tmp_csv), "--oracle-bias", str(oracle_bias),
           "--ge-slope-exponent", str(slope_exponent), "--ge-curvature-gain", str(curvature_gain)]
    cmd += ["--oracle-aggregation", oracle_aggregation, "--oracle-horizon", str(oracle_horizon)]
    cmd += ["--ge-base-model", ge_base_model]
    cmd += ["--prediction-bias", str(prediction_bias)]
    if pulse_start is not None:
        cmd += ["--oracle-pulse-start", str(pulse_start),
                "--oracle-pulse-duration", str(pulse_duration)]
    env = os.environ.copy(); env["GE_FF_MODE"] = mode
    env["ALT_PID_GAIN_SCALE"] = str(altitude_gain_scale)
    if clamp_delta is not None: env["GE_ASYM_CLAMP_DELTA"] = str(clamp_delta)
    proc = subprocess.run(cmd, cwd=root, env=env, text=True, capture_output=True, check=False)
    (log_dir / f"{tag}.stdout.txt").write_text(proc.stdout)
    (log_dir / f"{tag}.stderr.txt").write_text(proc.stderr)
    record = {"tag": tag, "mode": mode, "seed": seed, "command": cmd,
              "oracle_bias": oracle_bias, "slope_exponent": slope_exponent,
              "curvature_gain": curvature_gain, "clamp_delta": clamp_delta,
              "altitude_gain_scale": altitude_gain_scale,
              "oracle_aggregation": oracle_aggregation,
              "oracle_horizon": oracle_horizon,
              "ge_base_model": ge_base_model,
              "prediction_bias": prediction_bias,
              "pulse_start": pulse_start, "pulse_duration": pulse_duration,
              "returncode": proc.returncode, "status": "failed" if proc.returncode else "complete"}
    if proc.returncode:
        return record
    with tmp_csv.open("rb") as src, gzip.open(gz_csv, "wb", compresslevel=6) as dst:
        for chunk in iter(lambda: src.read(1024 * 1024), b""): dst.write(chunk)
    tmp_csv.unlink()
    try:
        df = read_telemetry(gz_csv)
        record.update(compute_metrics(df, cfg)); record["raw_csv"] = str(gz_csv)
        record["raw_sha256"] = sha256(gz_csv)
    except Exception as exc:
        record.update(status="failed", error=str(exc))
    return record


def paired_statistics(metrics: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    keys = ["agl_rms_primary", "low_clearance_duration_s", "low_clearance_events",
            "gyro_rms", "minimum_clearance_m", "p05_clearance_m"]
    rows = []
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    n_boot = int(cfg["bootstrap"]["resamples"])
    design_columns = [name for name in ("slope_exponent", "curvature_gain", "altitude_gain_scale", "ge_base_model")
                      if name in metrics and metrics[name].nunique(dropna=False) > 1]
    grouped = metrics.groupby(design_columns, dropna=False) if design_columns else [((), metrics)]
    for design_values, group in grouped:
        if not isinstance(design_values, tuple):
            design_values = (design_values,)
        design = dict(zip(design_columns, design_values))
        modes = sorted(group["mode"].unique())
        for i, a in enumerate(modes):
          for b in modes[i + 1:]:
            merged = group[group["mode"] == a].merge(
                group[group["mode"] == b], on="seed", suffixes=("_a", "_b"))
            for key in keys:
                d = merged[f"{key}_a"].to_numpy(float) - merged[f"{key}_b"].to_numpy(float)
                n = len(d)
                if n < 2: continue
                mean, sd = float(d.mean()), float(d.std(ddof=1))
                sem = sd / math.sqrt(n)
                tcrit = float(stats.t.ppf(0.975, n - 1))
                boot = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
                rows.append({**design, "mode_a": a, "mode_b": b, "metric": key, "n": n,
                             "mean_difference_a_minus_b": mean,
                             "paired_t_p_raw": float(stats.ttest_1samp(d, 0.0).pvalue),
                             "t_ci_low": mean - tcrit * sem, "t_ci_high": mean + tcrit * sem,
                             "bootstrap_ci_low": float(np.quantile(boot, .025)),
                             "bootstrap_ci_high": float(np.quantile(boot, .975)),
                             "cohen_dz": mean / sd if sd > 0 else np.nan})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # Holm family-wise correction within each metric/design cell.  Raw and
    # adjusted p-values are both retained; no comparison is silently dropped.
    family = design_columns + ["metric"]
    result["paired_t_p_holm"] = np.nan
    grouper = family[0] if len(family) == 1 else family
    for _, idx in result.groupby(grouper, dropna=False).groups.items():
        ordered = result.loc[idx, "paired_t_p_raw"].sort_values().index.tolist()
        m = len(ordered); running = 0.0
        for rank, row_idx in enumerate(ordered):
            running = max(running, (m - rank) * float(result.at[row_idx, "paired_t_p_raw"]))
            result.at[row_idx, "paired_t_p_holm"] = min(1.0, running)
    return result
