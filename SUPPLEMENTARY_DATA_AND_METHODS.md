# GES Supplementary Data and Methods Record

This document identifies the exact verification datasets and records the
implementation details needed to reproduce the Methods section. Values are
transcribed from the cited source files; none were inferred.

## Verification datasets

| Required evidence | Authoritative file | Contents |
|---|---|---|
| Six-controller paired comparisons | [`paired_statistics.csv`](Simulator-Development/simulation/experiments/results/reviewer_revision_20261119/closed_loop_retrained_30seed/paired_statistics.csv) | Paired mean differences, t-test p-values, t and deterministic-bootstrap 95% CIs, standardized effects, Holm-adjusted p-values |
| Controller seed metrics | [`per_seed_metrics.csv`](Simulator-Development/simulation/experiments/results/reviewer_revision_20261119/closed_loop_retrained_30seed/per_seed_metrics.csv) | AGL RMS, gyro RMS, minimum/P05 clearance, low-clearance duration and event count for every controller and seed |
| Common-support offline metrics | [`offline_metrics.csv`](Simulator-Development/simulation/experiments/results/reviewer_revision_20261119/offline_common_support/offline_metrics.csv) | MAE, RMSE, R2, training time, and identical test sample count for every model |
| Active-ML pulse windows | [`pulse_window_summary.csv`](Simulator-Development/simulation/experiments/results/reviewer_completion_20261119/ml_active_pulse_analysis/pulse_window_summary.csv) | Mean response, 95% CI, p-value, and n by pulse duration, sign, time window, and response variable |
| Linearized poles | [`linearized_closed_loop_poles.csv`](Simulator-Development/simulation/experiments/results/reviewer_revision_20261119/linearized_model/linearized_closed_loop_poles.csv) | Hover command, small-signal gain, and five poles at each GE operating point |
| Terrain decomposition | [`terrain_summary.csv`](Simulator-Development/simulation/experiments/results/reviewer_completion_20261119/terrain_decomposition/terrain_summary.csv) | Seed-level n and summarized metrics by terrain class, clearance band, and model |
| Complete 1,620-run manifest | [`run_manifest.json`](Simulator-Development/simulation/experiments/results/reviewer_revision_20261119/robustness_retrained_1620/run_manifest.json) | Commands, settings, status, telemetry paths, six modes, nine correction cells, and 30 seeds |
| Exact offline split | [`seed_split.json`](Simulator-Development/simulation/experiments/results/reviewer_revision_20261119/offline_common_support/seed_split.json) | Exact 240/30/30 train/validation/test seed lists |

The final controller seeds are 331--360, and clamp-selection seeds are
301--330. The full experiment configuration is
[`ijira_validity.json`](Simulator-Development/simulation/experiments/configs/ijira_validity.json).

## Ground-effect equations

Let `z` denote rotor clearance and `R=0.127 m` rotor radius. Clearance is
regularized as

```text
z_c = max(z, 0.5 R).
```

The default flat-ground Cheeseman--Bennett thrust factor is

```text
f_flat = 1 / (1 - (R / (4 z_c))^2).
```

For the Hayden sensitivity model, with `D=2R`,

```text
q = max(z_c / D, 0.15)
f_g = min(1 / (0.9926 + 0.03794 / q^2), 1)
f_flat = f_g^(-2/3).
```

The simulator-only heuristic terrain corrections are

```text
slope = atan(sqrt((dh/dx)^2 + (dh/dy)^2))
augmentation = (f_flat - 1) max(0, cos(slope))^p
c = clamp(1 + g kappa R^2, 0.5, 1.8)
f_true = clamp(1 + augmentation c, 0.85, 1.60),
```

where `kappa` is terrain curvature. The default values are `p=1.0` and
`g=0.6`; the robustness grid uses `p={0.5,1.0,2.0}` and
`g={0.3,0.6,0.9}`. These corrections are explicitly heuristic and are not
used by the analytical controller. Source:
[`aero_ground_effect.c`](Simulator-Development/simulation/physics/aero_ground_effect.c).

Because motor RPM is proportional to command and thrust is proportional to
RPM squared, all active feedforward modes use

```text
throttle_command = throttle_raw / sqrt(f_hat).
```

Consequently, ignoring saturation and transients,
`T_actual = T_raw f_true / f_hat`.

## Predictive analytical baseline

The predictive baseline is fixed before evaluation and uses only rangefinder
history. For each source sample it estimates vertical range rate over six
telemetry rows,

```text
v_z = (z[t] - z[t-6]) / (time[t] - time[t-6]),
z_preview = z[t] + v_z (target_time - source_time),
```

then evaluates the flat Cheeseman formula at `z_preview` and clips the result
to `[0.85,1.60]`. It has no slope or curvature correction. Source:
[`offline_seedwise.py`](Simulator-Development/simulation/experiments/offline_seedwise.py).

## Machine-learning inputs and hyperparameters

The 13 inputs are three gyroscope axes, rangefinder altitude, four rotor RPMs,
four motor currents, and normalized thrust command. The target is mean
ground-truth GE factor. Inputs are standardized using training seeds only.

Common settings:

- Forecast horizon: 0.1 s (six rows at 60 Hz); history window: 32 rows.
- Random seed: `20260825`; maximum epochs: 25.
- Optimizer/loss: Adam, learning rate `1e-3`, mean squared error.
- Training/validation batch sizes: 256/2048.
- Early stopping: validation RMSE, minimum improvement `1e-7`, patience 6;
  best-validation weights are restored.
- MLP: 13--64--32--1 with ReLU.
- Range-only MLP: flattened 32-sample range history, 32--64--32--1 with ReLU.
- LSTM: one layer, hidden size 48; 48--32--1 ReLU head.
- Transformer: model width 32, four heads, two encoder layers, feed-forward
  width 128, dropout 0.1; 32--1 head via a 32-unit ReLU layer.
- Random forest: 100 trees, maximum depth 14, minimum leaf size 3,
  `max_samples=0.5`.
- Gradient boosting: histogram gradient boosting, maximum depth 6,
  maximum 300 iterations.
- Regression tree: maximum depth 10, minimum leaf size 5.

Architectures are defined in
[`models_torch.py`](Simulator-Development/simulation/ml/models_torch.py), and
the exact training procedure is in
[`offline_seedwise.py`](Simulator-Development/simulation/experiments/offline_seedwise.py).

## Controller thresholds and safety-event definition

- GE factor clamp: `[0.85,1.60]`.
- Standard trust band: ±20% relative to analytical prediction.
- Narrow ablation trust band: ±5%.
- Hybrid low-clearance analytical cutover: rangefinder altitude below 0.15 m.
- Safety threshold: minimum per-rotor clearance below 0.15 m.
- Primary analysis excludes the first 10 s and targets AGL 0.22 m.
- An event begins at each false-to-true transition of the Boolean low-clearance
  signal. Event duration equals the number of low samples times the median
  telemetry interval (nominally 1/60 s). Consecutive low samples constitute one
  event; a return to at least 0.15 m closes it.
- Minimum clearance is the minimum per-rotor clearance over the analysis
  interval; P05 clearance is its empirical fifth percentile.
- Gyro RMS is the RMS of the Euclidean gyroscope magnitude.

The metric implementation is
[`validity.py`](Simulator-Development/simulation/experiments/validity.py).

## Statistical procedure

All controller contrasts are paired by terrain seed. Reported outputs include
the paired mean, two-sided paired t-test, t-based 95% CI, deterministic
20,000-resample paired bootstrap CI (seed `20260825`), standardized paired
effect, and Holm-adjusted p-value within the reported comparison family.

## Stability scope

The local vertical model assumes level hover, zero vertical velocity,
unsaturated actuators, constant GE factor, and no learned feedback. Its
parameters, companion matrices, characteristic polynomials, and numerical
poles are preserved under
[`linearized_model`](Simulator-Development/simulation/experiments/results/reviewer_revision_20261119/linearized_model/MODEL_SCOPE.md).
It excludes switching, clamps, mixer saturation, attitude estimation, terrain
motion, and nonlinear delay effects; negative pole real parts therefore do not
prove stability of the complete controller.
