#!/usr/bin/env python3
"""Lagged and pulse-intervention diagnostics for feedback-loop closure."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


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
    rows, coefficients, matrices = [], [], []
    for factor in (.85, 1., 1.2, 1.6):
        command = hover / np.sqrt(factor)
        plant_gain = 8 * kt * rpm_max ** 2 * command * factor / mass
        numerator = plant_gain * np.array([kp * deriv_tau + kd,
                                            kp + ki * deriv_tau, ki])
        characteristic = np.array([deriv_tau * motor_tau, deriv_tau + motor_tau,
                                   1., 0., 0., 0.])
        characteristic[-3:] += numerator
        normalized = characteristic / characteristic[0]
        companion = np.zeros((5, 5)); companion[:-1, 1:] = np.eye(4)
        companion[-1, :] = -normalized[:0:-1]
        coefficients.append({"ge_factor":factor, **{f"s_power_{5-i}":float(v) for i,v in enumerate(characteristic)}})
        for i in range(5):
            for j in range(5): matrices.append({"ge_factor":factor,"row":i,"column":j,"A":companion[i,j]})
        for pole in np.roots(characteristic):
            rows.append({"ge_factor": factor, "hover_command": command,
                         "small_signal_accel_per_command": plant_gain,
                         "pole_real": pole.real, "pole_imag": pole.imag,
                         "stable": bool(pole.real < 0)})
    pd.DataFrame(rows).to_csv(args.out_dir / "linearized_closed_loop_poles.csv", index=False)
    pd.DataFrame(coefficients).to_csv(args.out_dir / "characteristic_polynomials.csv", index=False)
    pd.DataFrame(matrices).to_csv(args.out_dir / "companion_state_matrices.csv", index=False)
    (args.out_dir / "MODEL_SCOPE.md").write_text(
        "# Local vertical-axis model\n\n"
        "Operating point: level hover, zero vertical velocity, unsaturated actuators, "
        "constant GE factor. Motor dynamics are `tau_m r_dot + r = u`; vertical "
        "dynamics are `z_ddot = K_m r`; the altitude PID is "
        "`u = (Kp + Ki/s + Kd*s/(1+tau_d*s)) (z_ref-z)`. "
        "Parameters: mass=1.50 kg, gravity=9.81 m/s^2, KT=1.472e-7, "
        "rpm_max=9000, tau_m=0.030 s, Kp=0.60, Ki=0.15, Kd=0.25, "
        "tau_d=0.15 s. The CSV companion matrices reproduce every reported pole. "
        "This is local evidence only and excludes switching, clamps, mixer saturation, "
        "attitude estimation, terrain motion, and learned-model feedback; it does not "
        "prove nonlinear closed-loop stability.\n")


def frequency(args):
    """Exact-frequency local loop with transport delay; no Padé approximation."""
    mass, gravity, kt, rpm_max = 1.50, 9.81, 1.472e-7, 9000.0
    motor_tau, deriv_tau = .030, .15
    kp, ki, kd = .60, .15, .25
    hover = np.sqrt((mass * gravity / 4.) / kt) / rpm_max
    omega = np.logspace(-3, 3, 12000)
    rows, responses = [], []
    for factor in (.85, 1., 1.2, 1.6):
      command = hover / np.sqrt(factor)
      plant_gain = 8 * kt * rpm_max ** 2 * command * factor / mass
      for gain_scale in (1., .5, .25):
       for delay in (0., .03, .10):
        s = 1j * omega
        controller = gain_scale * (kp + ki/s + kd*s/(1 + deriv_tau*s))
        plant = plant_gain / (s*s*(1 + motor_tau*s))
        loop = controller * plant * np.exp(-s*delay)
        magnitude = np.abs(loop)
        phase = np.unwrap(np.angle(loop))
        if phase[0] > 0: phase -= 2*np.pi
        # All negative-real-axis crossings; scaling r changes magnitude only.
        crossing_gains = []
        for level in -np.pi * np.arange(1, 21, 2):
            hit = np.flatnonzero((phase[:-1]-level)*(phase[1:]-level) <= 0)
            for i in hit:
                frac = (level-phase[i])/(phase[i+1]-phase[i]) if phase[i+1] != phase[i] else 0
                logmag = np.log(magnitude[i]) + frac*(np.log(magnitude[i+1])-np.log(magnitude[i]))
                crossing_gains.append(float(np.exp(-logmag)))
        gain_hits = np.flatnonzero((magnitude[:-1]-1)*(magnitude[1:]-1) <= 0)
        phase_margins=[]
        for i in gain_hits:
            frac=(1-magnitude[i])/(magnitude[i+1]-magnitude[i]) if magnitude[i+1] != magnitude[i] else 0
            ph=phase[i]+frac*(phase[i+1]-phase[i]); phase_margins.append(np.degrees(np.pi+ph))
        minimum_pm=min(phase_margins) if phase_margins else np.nan
        nominal_stable=bool(np.isfinite(minimum_pm) and minimum_pm > 0)
        admissible=min((g for g in crossing_gains if g > 1.0),default=np.nan) if nominal_stable else np.nan
        rows.append({"ge_factor":factor,"altitude_gain_scale":gain_scale,"delay_s":delay,
                     "plant_gain":plant_gain,"gain_margin_r_upper":admissible,
                     "minimum_phase_margin_deg":minimum_pm,"nominal_locally_stable":nominal_stable,
                     "negative_axis_crossings":len(crossing_gains)})
        if factor == 1. and gain_scale == 1.:
            responses.extend({"delay_s":delay,"omega_rad_s":w,"magnitude_db":20*np.log10(m),
                              "phase_deg":np.degrees(p)} for w,m,p in zip(omega,magnitude,phase))
    pd.DataFrame(rows).to_csv(args.out_dir/"delay_gain_margins.csv",index=False)
    pd.DataFrame(responses).to_csv(args.out_dir/"default_bode_response.csv",index=False)
    frame=pd.DataFrame(responses); fig,(ax1,ax2)=plt.subplots(2,1,sharex=True,figsize=(6,5))
    for delay,g in frame.groupby("delay_s"):
        ax1.semilogx(g.omega_rad_s,g.magnitude_db,label=f"delay={delay:.2f}s")
        ax2.semilogx(g.omega_rad_s,g.phase_deg)
    ax1.axhline(0,color="0.5",lw=.8); ax2.axhline(-180,color="0.5",lw=.8)
    ax1.set(ylabel="Magnitude (dB)"); ax1.legend(); ax2.set(xlabel="Frequency (rad/s)",ylabel="Phase (deg)")
    fig.tight_layout(); fig.savefig(args.out_dir/"bode_with_delay.png",dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="command", required=True)
    lag = sub.add_parser("lagged"); lag.add_argument("--run-dir", type=Path, required=True)
    lag.add_argument("--out-dir", type=Path, required=True); lag.add_argument("--start-s", type=float, default=10.)
    pul = sub.add_parser("pulse"); pul.add_argument("--run-dir", type=Path, required=True)
    pul.add_argument("--out-dir", type=Path, required=True)
    lin = sub.add_parser("linearized"); lin.add_argument("--out-dir", type=Path, required=True)
    freq = sub.add_parser("frequency"); freq.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "lagged": lagged(args)
    elif args.command == "pulse": pulse(args)
    elif args.command == "linearized": linearized(args)
    else: frequency(args)


if __name__ == "__main__": main()
