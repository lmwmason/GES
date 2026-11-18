# IJIRA experimental-validity suite

This suite writes every run to a new timestamped directory under
`experiments/results/ijira_validity/`. It never edits the manuscript or the
legacy block-split results. The legacy procedure remains available through
the original `ml/train_models.py` and `ml/train_all_models.py`; all outputs
from this suite are explicitly seed-wise.

The registered configuration is `configs/ijira_validity.json`. Its seed lists
are chosen before outcomes are inspected: offline seeds 1--300 are shuffled
with Python `random.Random(20260825)` into 240/30/30 groups, controller clamp
validation uses 301--330, and the final controller set uses 331--360. Thus no
closed-loop seed participates in model training, validation, or offline test.
The oracle is diagnostic and cannot be deployed because it receives the
simulator's ground-truth GE multiplier.

## Major-revision corrected-dynamics rerun

The reviewer rerun is written to a new directory. Dataset generation refuses
to reuse a non-empty output directory. The data-collection controller is fixed
before outcomes are inspected (`analytical`), seeds 1--300 are offline-only,
and controller seeds 331--360 remain unseen.

```bash
python experiments/generate_corrected_dataset.py \
  --out-dir dataset/reviewer_corrected_20261119 \
  --start-seed 1 --n-seeds 300 --duration 60 --workers 32 --mode analytical
python experiments/offline_seedwise.py \
  --config experiments/configs/ijira_validity.json \
  --dataset dataset/reviewer_corrected_20261119/dataset_all.csv \
  --out-dir experiments/results/reviewer_revision_20261119/offline_common_support \
  --epochs 25 --models all --device cuda
```

Every model in this rerun uses the same 32-tick warm-up and identical 100
ms-ahead target rows. Instantaneous analytical, constant-velocity predictive
analytical, rangefinder-only MLP, full-sensor models, LSTM and Transformer are
therefore scored against one truth vector. `pointwise_predictions.csv.gz`
records seed, source time and target time for auditability.

```bash
python ml/export_mlp_to_c.py \
  --joblib experiments/results/reviewer_revision_20261119/offline_common_support/models.joblib \
  --out firmware/ge_ff_ml_weights.h
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
ctest --test-dir build --output-on-failure
python experiments/run.py closed-loop --run-id reviewer_corrected_closed_loop_20261119
python experiments/run.py bias --run-id reviewer_corrected_bias_20261119
python experiments/run.py oracle-aggregation --run-id reviewer_oracle_aggregation_20261119 --workers 16
python experiments/run.py bandwidth --run-id reviewer_feedback_bandwidth_20261119 --workers 16
```

Run `clamp-validation` on seeds 301--330 and freeze its selected value before
any final-seed clamp or robustness run. Never tune that value on seeds 331--360.

Additional reviewer diagnostics:

```bash
python experiments/run.py oracle-timing --run-id oracle_timing_30seed --workers 20
python experiments/run.py base-model --run-id base_model_30seed_v2 --workers 30
python experiments/run.py ml-pulse --run-id ml_active_pulse_grid --workers 30
python experiments/ml_pulse_analysis.py \
  --run-dir experiments/results/ijira_validity/ml_active_pulse_grid \
  --out-dir experiments/results/ijira_validity/ml_active_pulse_analysis
python experiments/loop_closure.py frequency \
  --out-dir experiments/results/ijira_validity/frequency_stability_v2
python experiments/terrain_decomposition.py \
  --dataset dataset/reviewer_corrected_20261119/dataset_all.csv \
  --offline-dir experiments/results/reviewer_revision_20261119/offline_common_support \
  --out-dir experiments/results/ijira_validity/terrain_decomposition
for horizon in 0 0.05 0.10 0.20; do
  python experiments/offline_seedwise.py \
    --config experiments/configs/ijira_validity.json \
    --dataset dataset/reviewer_corrected_20261119/dataset_all.csv \
    --out-dir "experiments/results/ijira_validity/horizon_${horizon}s" \
    --models rf mlp lstm --lead-s "$horizon" --epochs 25
done
python experiments/horizon_summary.py \
  --root experiments/results/ijira_validity \
  --out-dir experiments/results/ijira_validity/horizon_summary
python experiments/completion_report.py \
  --root experiments/results/ijira_validity \
  --out-dir experiments/results/ijira_validity/publication_figures
```

`oracle-timing` compares current-factor cancellation with a diagnostic 100 ms
constant-velocity terrain preview; the latter is not claimed to be exact future
state knowledge. `base-model` compares the original Cheeseman--Bennett base
against the NASA NDARC Hayden empirical induced-power model at the same default
slope/curvature correction (NASA TP-20220000355, Section 12.4.1.2).
`ml-pulse` perturbs the actively
deployed learned factor, not a non-actuating shadow predictor.

## Environment and tests

Run from `Simulator-Development/simulation`:

```sh
python3 -m venv .venv
.venv/bin/pip install -r ml/requirements.txt
cmake -S . -B build
cmake --build build -j2
ctest --test-dir build --output-on-failure
.venv/bin/python -m unittest discover -s tests -v
```

## Registered experiment order

Use explicit run IDs to make paths easy to cite. Do not tune the asymmetric
clamp on seeds 331--360.

```sh
.venv/bin/python experiments/run.py split --run-id seed_split_20260825
.venv/bin/python experiments/run.py offline --run-id offline_seedwise_20260825 --offline-models all --offline-epochs 25
.venv/bin/python experiments/run.py clamp-validation --run-id clamp_validation_20260825
# Read selected_clamp.json, freeze it, then substitute D below.
.venv/bin/python experiments/run.py closed-loop --run-id closed_loop_30seed_20260825 --clamp-delta D
.venv/bin/python experiments/run.py bias --run-id oracle_bias_30seed_20260825
.venv/bin/python experiments/run.py robustness --run-id robustness_30seed_20260825_complete --clamp-delta D --workers 4
```

The closed-loop command always runs none, analytical, oracle, direct/pure MLP,
and the existing hybrid MLP on exactly the same 30 seeds. Supplying the frozen
clamp delta adds the asymmetric-clamp controller. A failed simulator process
is retained in `failures.json` and aborts aggregation; it is never silently
dropped. Each successful raw telemetry CSV is gzip-compressed and SHA-256
recorded in `run_manifest.json`.

Generate publication-quality derivative artifacts without changing LaTeX:

```sh
.venv/bin/python experiments/collect_offline.py --runs \
  experiments/results/ijira_validity/offline_seedwise_analytical_gb_20260825 \
  experiments/results/ijira_validity/offline_seedwise_rf_20260825 \
  experiments/results/ijira_validity/offline_seedwise_linear_tree_20260826 \
  experiments/results/ijira_validity/offline_seedwise_mlp_25epoch_20260826 \
  experiments/results/ijira_validity/offline_seedwise_lstm_25epoch_20260826 \
  experiments/results/ijira_validity/offline_seedwise_transformer_25epoch_20260826 \
  --out-dir experiments/results/ijira_validity/offline_seedwise_final_20260826
.venv/bin/python experiments/report.py \
  --offline experiments/results/ijira_validity/offline_seedwise_final_20260826 \
  --closed-loop experiments/results/ijira_validity/closed_loop_30seed_20260825 \
  --bias experiments/results/ijira_validity/oracle_bias_30seed_20260825 \
  --robustness experiments/results/ijira_validity/robustness_30seed_20260825_complete \
  --out-dir experiments/results/ijira_validity/publication_20260825
```

Primary outcomes are post-transient body-center AGL RMS and low-clearance
violation duration/event count. Safety clearance is the minimum of four rotor
clearances. Gyroscope RMS, minimum/p05 clearance, and the legacy full-duration
AGL RMS are exploratory. Paired differences are `mode_a - mode_b`; tables
include t intervals, deterministic 20,000-resample bootstrap intervals, and
paired Cohen's dz. Robustness comparisons remain separated by their registered
slope-exponent/curvature-gain condition.

## Loop-closure diagnostic

This uses the existing hybrid trajectories and a predefined 0.25 s oracle
bias pulse at 30 s. The learned model runs in shadow mode during the oracle
experiment, so its error is observable but cannot affect the pulse response.

```sh
.venv/bin/python experiments/run.py pulse --run-id oracle_pulse_loop_closure_20260826
.venv/bin/python experiments/loop_closure.py lagged \
  --run-dir experiments/results/ijira_validity/closed_loop_30seed_20260825 \
  --out-dir experiments/results/ijira_validity/loop_closure_lagged_20260826
.venv/bin/python experiments/loop_closure.py pulse \
  --run-dir experiments/results/ijira_validity/oracle_pulse_loop_closure_20260826 \
  --out-dir experiments/results/ijira_validity/loop_closure_pulse_analysis_20260826
.venv/bin/python experiments/loop_closure.py linearized \
  --out-dir experiments/results/ijira_validity/loop_closure_linearized_20260826
```
