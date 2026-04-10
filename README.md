# Reefer Load Forecasting for the Eurogate Challenge

Forecasting hourly terminal reefer electricity demand with a leak-safe pipeline, a calibrated `p90` forecast, and a gated blend of two complementary point models.

## Current Best Result
The current primary submission is:

- `peakprob_gate`
- `--residual-training-policy nov_dec_only`
- file: [predictions.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/predictions.csv)

Public-window score:

| Metric | Value |
| --- | ---: |
| `mae_all` | `35.805733` |
| `mae_peak` | `34.918464` |
| `pinball_p90` | `7.173262` |
| `score` | `29.813058` |

Safer backup:

- `peakprob_gate`
- `--residual-training-policy jan_nov_dec`
- file: [outputs/predictions_peakprob_gate_jan_nov_dec.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/predictions_peakprob_gate_jan_nov_dec.csv)
- public score: `31.693414`

Incremental milestone kept for reference:

- `peakprob_gate`
- `--residual-training-policy full_year`
- file: [outputs/predictions_peakprob_gate_full_year.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/predictions_peakprob_gate_full_year.csv)
- public score: `34.180486`

Scoring formula:

```text
score = 0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball_p90
```

Lower is better.

## What The Solution Predicts
The challenge requires hourly forecasts for the timestamps in [target_timestamps.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/target_timestamps.csv).

The model predicts:
- `pred_power_kw`
- `pred_p90_kw`

The public evaluation window is:
- `2026-01-01 00:00 UTC` through `2026-01-10 06:00 UTC`

## Core Idea
The final solution keeps one shared feature pipeline and combines three pieces:

1. A `residual` model for stable normal-hour behavior.
2. A `direct` model for sharper high-load behavior.
3. A learned `peakprob_gate` that blends the two hour by hour.

The last major gain came from changing only what the `residual` branch trains on:
- not all of 2025
- only the late-year winter-like portion of 2025

That lets the normal-hours model specialize for early January conditions, while the direct model and gate still benefit from full-year training data.

## Data Used
Only the supplied challenge files are used:

- [reefer_release.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/reefer_release.csv)
- [Wetterdaten Okt 25 - 23 Feb 26](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/Wetterdaten%20Okt%2025%20-%2023%20Feb%2026)
- [target_timestamps.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/target_timestamps.csv)

The hourly target is aggregated as:
- `y_kw = sum(AvPowerCons) / 1000`

## Leak Safety
The pipeline is designed to stay compliant with the challenge constraints:

- training labels only come from 2025
- reefer-state features are only used through lagged values
- weather is shifted by `24h` before use
- no January 2026 actuals are used for training
- folds are strictly time-ordered
- the post-holiday release flag is calendar-known in advance
- residual training-policy filters only choose subsets of 2025 rows

Leak checks are implemented in:
- [check_leak_safety.py](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/check_leak_safety.py)

## Feature Highlights
The strongest feature groups are:

- load lags:
  - `feat_lag_24`
  - `feat_lag_48`
  - `feat_lag_72`
  - `feat_lag_168`
- rolling lag summaries:
  - `feat_roll24_*`
  - `feat_roll72_*`
  - `feat_roll168_*`
- calendar features
- lagged reefer fleet summaries
- lagged weather summaries

The most useful engineered additions were the thermal-burden features:

- `feat_thermal_lift_ambient_setpoint_pos_lag24`
- `feat_aggregate_thermal_load_ambient_lag24`
- `feat_thermal_lift_return_setpoint_pos_lag24`
- `feat_aggregate_thermal_load_return_lag24`
- `feat_thermal_lift_weather_setpoint_pos_lag24`
- `feat_aggregate_thermal_load_weather_lag24`

These encode how much cooling effort the active fleet is likely to need.

There is also one sparse calendar feature kept only for the residual branch:
- `feat_post_holiday_release_flag`

This is not a generic holiday flag. It is only active on a very small set of known post-holiday fleet-release dates.

## Model Modes
The training entry point is [train_and_predict.py](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/train_and_predict.py).

Supported modes:
- `residual`
- `direct`
- `peakprob_gate`

### `residual`
- baseline:
  - `0.7 * feat_lag_24 + 0.3 * feat_lag_168`
- target:
  - `y_kw - baseline`

### `direct`
- target:
  - `y_kw`

### `peakprob_gate`
Trains:
- one residual model
- one direct model
- one gate model predicting whether an hour is likely to be high-load

The gate turns predicted peak probability into a direct-model blend weight:
- `direct_weight = 0.3 + 0.7 * peak_prob`

Final point forecast:
- `pred_power = direct_weight * direct_pred + (1 - direct_weight) * residual_pred`

## Residual Training Policy
The main final improvement is controlled by:
- `--residual-training-policy`

Supported values:
- `full_year`
- `jan_nov_dec`
- `jan_oct_nov_dec`
- `nov_dec_only`

Only the `residual` branch is filtered by this policy.
The `direct` branch and `gate` still train on full 2025.

Benchmark summary:

| Residual Training Policy | Public Score | `mae_all` | `mae_peak` | Mean Fold Score |
| --- | ---: | ---: | ---: | ---: |
| `nov_dec_only` | `29.813058` | `35.805733` | `34.918464` | `95.526099` |
| `jan_nov_dec` | `31.693414` | `36.551358` | `40.135490` | `93.600879` |
| `full_year` | `34.180486` | `51.129222` | `23.177656` | `95.850613` |

The full benchmark files are:
- [outputs/residual_training_policy_benchmark.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/residual_training_policy_benchmark.csv)
- [outputs/residual_training_policy_benchmark.json](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/residual_training_policy_benchmark.json)

## Repository Layout

```text
.
├── README.md
├── approach.md
├── worked_summary.md
├── build_features.py
├── check_leak_safety.py
├── score_public_submission.py
├── train_and_predict.py
├── pipeline/
│   └── reefer_pipeline.py
├── outputs/
│   ├── hourly_feature_table.csv
│   ├── predictions_peakprob_gate.csv
│   ├── predictions_peakprob_gate_full_year.csv
│   ├── predictions_peakprob_gate_jan_nov_dec.csv
│   ├── predictions_peakprob_gate_nov_dec_only.csv
│   ├── backtest_metrics_peakprob_gate.json
│   ├── model_ready_feature_table_peakprob_gate.csv
│   ├── preprocessing_summary_peakprob_gate.json
│   ├── best_model_peakprob_gate.pkl
│   ├── residual_training_policy_benchmark.csv
│   └── residual_training_policy_benchmark.json
└── predictions.csv
```

## How To Run
Install the core Python dependencies:

```bash
pip install numpy pandas scikit-learn
```

Build the canonical feature table:

```bash
/usr/bin/python3 build_features.py --root .
```

Run leak checks:

```bash
/usr/bin/python3 check_leak_safety.py
```

Train the final primary model:

```bash
/usr/bin/python3 train_and_predict.py --root . --model-mode peakprob_gate --residual-training-policy nov_dec_only
```

Train the safer backup:

```bash
/usr/bin/python3 train_and_predict.py --root . --model-mode peakprob_gate --residual-training-policy jan_nov_dec
```

Score kept submissions on the public window:

```bash
/usr/bin/python3 score_public_submission.py predictions.csv outputs/predictions_peakprob_gate_jan_nov_dec.csv outputs/predictions_peakprob_gate_full_year.csv
```

## Additional Notes
- [approach.md](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/approach.md) contains the full technical walkthrough.
- [worked_summary.md](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/worked_summary.md) records the ideas that actually improved the solution.
- The repo intentionally keeps only the final and incrementally useful outputs, not the failed branches.

## License
This project is released under the [MIT License](LICENSE).
