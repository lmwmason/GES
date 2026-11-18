# 2026-11-19 reviewer-revision experiment bundle

This directory is an immutable copy of the successful GPU/CPU experiment run.
No failed runs were omitted; no `failures.json` was produced.  The archive
transport SHA-256 was
`3860bfd76aa935641f070f3cdb07b224aee74c22f33b10d331eff3168eeb8eda`.

The corrected-dynamics dataset contains seeds 1--300 (60 s each, analytical
data-collection controller).  Its merged CSV SHA-256 is
`1be2d7bc285efe15e3cb513822d691f494f488f05fe1bb3db434c0e3501674ad`.
The deterministic split and exact seed lists are in
`offline_common_support/seed_split.json`; final controller seeds are 331--360.

Key directories:

- `corrected_dataset`: merged raw offline dataset and generation manifest.
- `offline_common_support`: models, common-row predictions and metrics.
- `clamp_validation_retrained`: validation-only threshold selection (delta 0).
- `closed_loop_retrained_30seed`: six-controller raw telemetry and statistics.
- `bias_retrained_30seed`: five signed oracle-bias conditions.
- `oracle_aggregation_30seed`: arithmetic versus thrust-weighted oracle.
- `feedback_bandwidth_30seed`: altitude-gain scales 1, 0.5 and 0.25.
- `robustness_retrained_1620`: all 1,620 preregistered robustness runs.
- `linearized_model`: equations, matrices, polynomials and numerical poles.
- `publication_figures_v2`: regenerated tables and 95%-CI figures.

Main findings: LSTM is the best common-support offline model (RMSE 0.007417,
R2 0.91993), versus analytical RMSE 0.024537.  On 30 unseen controller seeds,
analytical AGL RMS is 0.060436 m, hybrid MLP 0.061283 m, asymmetric clamp
0.060676 m, and oracle 0.060578 m.  The analytical--oracle paired difference
is -0.000142 m (95% CI -0.000284 to -0.00000048); its Holm-adjusted p-value is
0.0986, so this run does not establish an oracle advantage.

Positive oracle bias reduces AGL RMS slightly but worsens safety-oriented
clearance measures: +0.02 versus -0.02 changes 5th-percentile clearance by
-0.002571 m (95% CI -0.003422 to -0.001721) and low-clearance duration by
+0.325833 s (95% CI +0.207320 to +0.444347).  Interpret the intervention as a
safety trade-off, not as evidence that every outcome metric worsens.

Hybrid MLP has higher mean AGL RMS than analytical in all nine correction
cells (difference 0.000845--0.000868 m).  These are synthetic-model results;
they are not independent aerodynamic or hardware validation.
