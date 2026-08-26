#!/usr/bin/env python3
"""Lagged and pulse-intervention diagnostics for feedback-loop closure."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def mean_test(values):
    x = np.asarray(values, float); n = len(x); sem = stats.sem(x)
    ci = stats.t.interval(.95, n - 1, loc=x.mean(), scale=sem)
    test = stats.ttest_1samp(x, 0.)
    return x.mean(), ci[0], ci[1], test.statistic, test.pvalue, n


def standardized_beta(y, *xs):
    arrays = [np.asarray(v, float) for v in (y,) + xs]
    arrays = [(v - v.mean()) / v.std(ddof=0) for v in arrays]
    design = np.column_stack([np.ones(len(y))] + arrays[1:])
    return np.linalg.lstsq(design, arrays[0], rcond=None)[0][1:]


def lagged(args):
    manifest = pd.read_json(args.run_dir / "run_manifest.json")
    manifest = manifest[manifest["mode"] == "ml"]
    per_seed = []
    for _, record in manifest.iterrows():
        d = pd.read_csv(record.raw_csv, compression="gzip")
        d = d[d.t >= args.start_s].reset_index(drop=True)
        error = (d.ge_ff_ai - d.ge_factor_truth_mean).to_numpy()
        magnitude = np.abs(error); clearance = d.min_rotor_clearance.to_numpy()
        truth = d.ge_factor_truth_mean.to_numpy()
        dclear = np.r_[0., np.diff(clearance)]
        for lag in range(1, 11):
            sl = slice(None, -lag); future = slice(lag, None)
            # Forward: future clearance, controlling current clearance/truth.
            forward = standardized_beta(clearance[future], clearance[sl], truth[sl], error[sl])[-1]
            # Reverse: future |error|, controlling its current value, truth and clearance.
            reverse_clear = standardized_beta(magnitude[future], magnitude[sl], truth[sl], clearance[sl])[-1]
            reverse_dclear = standardized_beta(magnitude[future], magnitude[sl], truth[sl],
                                               clearance[sl], dclear[sl])[-1]
            per_seed.append({"seed": int(record.seed), "lag_ticks": lag,
                "forward_error_to_clearance_beta": forward,
                "reverse_clearance_to_abs_error_beta": reverse_clear,
                "reverse_dclear_to_abs_error_beta": reverse_dclear})
    raw = pd.DataFrame(per_seed); raw.to_csv(args.out_dir / "lagged_per_seed.csv", index=False)
    rows = []
    for lag, group in raw.groupby("lag_ticks"):
        for metric in raw.columns[2:]:
            mean, lo, hi, t, p, n = mean_test(group[metric])
            rows.append({"lag_ticks": lag, "metric": metric, "mean_standardized_beta": mean,
                         "ci_low": lo, "ci_high": hi, "t": t, "p": p, "n_seeds": n})
    pd.DataFrame(rows).to_csv(args.out_dir / "lagged_summary.csv", index=False)


def pulse(args):
    manifest = pd.read_json(args.run_dir / "run_manifest.json")
    pulse_cfg = manifest.iloc[0]
    start = float(pulse_cfg.pulse_start); end = start + float(pulse_cfg.pulse_duration)
    records = []
    frames = {}
    for _, r in manifest.iterrows():
        frames[(int(r.seed), float(r.oracle_bias))] = pd.read_csv(r.raw_csv, compression="gzip")
    windows = [(0., .5), (.5, 1.), (1., 2.), (2., 5.), (5., 10.)]
    window_rows = []
    for seed in sorted(manifest.seed.unique()):
        control = frames[(int(seed), 0.)]
        for bias in (-.02, .02):
            treated = frames[(int(seed), bias)]
            if not np.allclose(treated.t, control.t): raise ValueError("unaligned pulse trajectories")
            time = treated.t.to_numpy(); post = time >= end
            dc = treated.min_rotor_clearance.to_numpy() - control.min_rotor_clearance.to_numpy()
            de = np.abs(treated.ge_ff_ai - treated.ge_factor_truth_mean).to_numpy() - np.abs(
                 control.ge_ff_ai - control.ge_factor_truth_mean).to_numpy()
            post_dc = dc[post]; post_t = time[post] - end
            peak_i = int(np.argmin(post_dc)) if bias > 0 else int(np.argmax(post_dc))
            peak = float(post_dc[peak_i]); threshold = max(abs(peak) * .1, 1e-5)
            settle = np.nan
            for i in range(peak_i, len(post_dc)):
                horizon = post_dc[i:i + 60]
                if len(horizon) == 60 and np.all(np.abs(horizon) <= threshold):
                    settle = float(post_t[i]); break
            after_peak = post_dc[peak_i:]
            crossings = int(np.sum(np.diff(np.signbit(after_peak)) != 0))
            records.append({"seed": seed, "bias": bias, "peak_clearance_delta_m": peak,
                "peak_time_after_pulse_s": float(post_t[peak_i]), "settling_time_s": settle,
                "zero_crossings_after_peak": crossings,
                "mean_post10s_clearance_delta_m": float(post_dc[post_t <= 10].mean())})
            for lo, hi in windows:
                mask = (time >= end + lo) & (time < end + hi)
                window_rows.append({"seed": seed, "bias": bias, "window_start_s": lo,
                    "window_end_s": hi, "clearance_delta_m": float(dc[mask].mean()),
                    "shadow_abs_error_delta": float(de[mask].mean())})
    per = pd.DataFrame(records); per.to_csv(args.out_dir / "pulse_per_seed.csv", index=False)
    win = pd.DataFrame(window_rows); win.to_csv(args.out_dir / "pulse_windows_per_seed.csv", index=False)
    summaries = []
    for keys, group in win.groupby(["bias", "window_start_s", "window_end_s"]):
        for metric in ("clearance_delta_m", "shadow_abs_error_delta"):
            mean, lo, hi, t, p, n = mean_test(group[metric])
            summaries.append({"bias": keys[0], "window_start_s": keys[1], "window_end_s": keys[2],
                "metric": metric, "mean": mean, "ci_low": lo, "ci_high": hi, "t": t, "p": p, "n": n})
    pd.DataFrame(summaries).to_csv(args.out_dir / "pulse_window_summary.csv", index=False)


def linearized(args):
    # Flat-hover small-signal approximation. These constants mirror
    # drone_params.h and controller.c; saturation, terrain motion, the mixer,
    # and attitude-estimator dynamics are deliberately excluded.
    mass, gravity, kt, rpm_max = 1.50, 9.81, 1.472e-7, 9000.0
    motor_tau, deriv_tau = .030, .15
    kp, ki, kd = .60, .15, .25
    hover = np.sqrt((mass * gravity / 4.) / kt) / rpm_max
    rows = []
    for factor in (.85, 1., 1.2, 1.6):
        command = hover / np.sqrt(factor)
        plant_gain = 8 * kt * rpm_max ** 2 * command * factor / mass
        numerator = plant_gain * np.array([kp * deriv_tau + kd,
                                            kp + ki * deriv_tau, ki])
        characteristic = np.array([deriv_tau * motor_tau, deriv_tau + motor_tau,
                                   1., 0., 0., 0.])
        characteristic[-3:] += numerator
        for pole in np.roots(characteristic):
            rows.append({"ge_factor": factor, "hover_command": command,
                         "small_signal_accel_per_command": plant_gain,
                         "pole_real": pole.real, "pole_imag": pole.imag,
                         "stable": bool(pole.real < 0)})
    pd.DataFrame(rows).to_csv(args.out_dir / "linearized_closed_loop_poles.csv", index=False)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="command", required=True)
    lag = sub.add_parser("lagged"); lag.add_argument("--run-dir", type=Path, required=True)
    lag.add_argument("--out-dir", type=Path, required=True); lag.add_argument("--start-s", type=float, default=10.)
    pul = sub.add_parser("pulse"); pul.add_argument("--run-dir", type=Path, required=True)
    pul.add_argument("--out-dir", type=Path, required=True)
    lin = sub.add_parser("linearized"); lin.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "lagged": lagged(args)
    elif args.command == "pulse": pulse(args)
    else: linearized(args)


if __name__ == "__main__": main()
