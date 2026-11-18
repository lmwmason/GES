# Reviewer-completion experiments (2026-11-19)

This directory preserves the raw, per-seed, model, log, summary, and figure
outputs for the experiments added in response to peer review. No failed runs
were omitted. Seed lists and runtime metadata are stored with each experiment.
The archive copied from the GPU host had SHA-256
`a11f40c571bb5fc8679057822f5e3b441a08f21a5d8394576c130803f9716978`.

## Main findings

- **Prediction horizon:** on the held-out seed set, LSTM RMSE increased from
  0.001859 at 0 ms to 0.005251, 0.007613, and 0.010013 at 50, 100, and
  200 ms. It remained the best tested predictor at every horizon. The 100 ms
  analytical/RF/MLP/LSTM RMSE values were 0.024537/0.013927/0.014022/0.007613.
- **Oracle timing and aggregation:** current-state arithmetic oracle AGL RMS
  was 0.060578 m; thrust-weighted aggregation was 0.060794 m. A kinematic
  100 ms preview worsened these to 0.064186 and 0.064189 m. This is a timing
  diagnostic, not an exact future-state oracle.
- **Ground-effect base-model sensitivity:** under Cheeseman, analytical/oracle/
  hybrid-ML AGL RMS was 0.060436/0.060578/0.061283 m. Under the independently
  implemented Hayden base model it was 0.061573/0.060107/0.062475 m. Thus ML
  remained worse than analytical, but the claim that the analytical controller
  is at the oracle ceiling did **not** generalize: under Hayden, oracle improved
  AGL RMS by 0.001466 m (95% t CI 0.001185--0.001747, Holm p=6.05e-11).
- **Active learned-controller pulses:** signed prediction pulses caused an
  immediate clearance response whose peak increased with duration (mean peak
  for magnitude 0.02 was approximately 0.000815, 0.00284, and 0.00651 m for
  1-tick, 0.1 s, and 0.25 s pulses). Natural prediction-error magnitude could
  increase during the first 0.5 s for either sign, but no persistent significant
  error increase remained in later windows. The data support transient
  trajectory coupling, not a sustained self-reinforcing prediction loop.
- **Delay sensitivity:** for GE=1 and nominal altitude gains, the exact-delay
  linear model had phase margins 14.15 degrees (0 ms), 0.80 degrees (30 ms),
  and -30.37 degrees (100 ms). A blanket local-stability claim at 100 ms delay
  is therefore not supported by this linearized diagnostic.

## Directory guide

- `horizon_*` and `horizon_summary`: models, predictions, split, metrics, and
  cross-horizon summary.
- `oracle_timing_30seed`: current/100-ms-preview and arithmetic/thrust-weighted
  oracle comparisons on the same 30 seeds.
- `base_model_30seed_v2`: authoritative predeclared Cheeseman/Hayden comparison.
  The directory without `_v2` is retained as a superseded diagnostic in which
  the deployment clamp was not frozen.
- `ml_active_pulse_grid` and `ml_active_pulse_analysis`: raw telemetry and
  per-seed/window summaries for pulses applied to the active learned controller.
- `frequency_stability_v2`: authoritative exact-delay frequency analysis. The
  directory without `_v2` is retained as the superseded first calculation.
- `terrain_decomposition`: independent-test-seed terrain-class and clearance-
  band metrics.
- `publication_figures`: regenerated figures from the authoritative summaries.

See `experiments/README.md` for exact reproduction commands.
