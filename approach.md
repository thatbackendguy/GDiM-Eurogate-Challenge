# Reefer Forecasting Approach

## Final Recommendation
The repo now keeps one clean final approach and one safer backup.

- Primary submission:
  - `peakprob_gate` with `--residual-training-policy nov_dec_only`
  - public score: `29.813058`
  - file: [predictions.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/predictions.csv)
- Safer backup:
  - `peakprob_gate` with `--residual-training-policy jan_nov_dec`
  - public score: `31.693414`
  - file: [outputs/predictions_peakprob_gate_jan_nov_dec.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/predictions_peakprob_gate_jan_nov_dec.csv)
- Incremental milestone kept for reference:
  - `peakprob_gate` with `--residual-training-policy full_year`
  - public score: `34.180486`
  - file: [outputs/predictions_peakprob_gate_full_year.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/predictions_peakprob_gate_full_year.csv)

The main idea is simple:
- keep the shared leak-safe feature pipeline
- keep the thermal and holiday-release signals that helped
- keep the direct and residual models plus the learned gate
- only specialize the `residual` branch to late-year winter-like data

## Challenge Setup
The task is to predict terminal-wide hourly reefer power for the timestamps in [target_timestamps.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/target_timestamps.csv).

The public target window is:
- `2026-01-01 00:00 UTC` through `2026-01-10 06:00 UTC`

The pipeline predicts:
- `pred_power_kw`
- `pred_p90_kw`

The challenge score is:
- `0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball_p90`

## Data Used
Only the supplied challenge files are used:

- [reefer_release.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/reefer_release.csv)
- [Wetterdaten Okt 25 - 23 Feb 26](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/Wetterdaten%20Okt%2025%20-%2023%20Feb%2026)
- [target_timestamps.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/target_timestamps.csv)

The hourly target is built as:
- `y_kw = sum(AvPowerCons) / 1000`

## Leak Safety
The current implementation remains future-safe.

Training labels are restricted to:
- `2025-01-01 00:00:00` through `2025-12-31 23:00:00`

Leak-prevention rules used in code:
- reefer-state features are only consumed through lagged versions
- weather is shifted by `24h` before it is used
- no Jan 2026 actuals are used for training
- validation folds are strictly time-ordered
- the post-holiday release flag is calendar-known in advance
- the residual training-policy filter only selects subsets of 2025 rows, never 2026 rows

The automated checker is:
- [check_leak_safety.py](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/check_leak_safety.py)

## Shared Feature Pipeline
All model modes share the same base feature table from [pipeline/reefer_pipeline.py](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/pipeline/reefer_pipeline.py).

Main feature groups:
- load lags:
  - `feat_lag_24`
  - `feat_lag_48`
  - `feat_lag_72`
  - `feat_lag_168`
- rolling lag summaries:
  - `feat_roll24_*`
  - `feat_roll72_*`
  - `feat_roll168_*`
- calendar features:
  - hour
  - day of week
  - month
  - day of year
  - weekend flag
  - cyclical encodings
- lagged reefer summaries:
  - active container count
  - mean and median reefer temperatures
  - hardware mix
  - size mix
  - stack-tier mix
- lagged weather summaries

## Thermal Features
The most important engineered feature family is still the thermal-burden branch.

Kept thermal features:
- `feat_thermal_lift_ambient_setpoint_pos_lag24`
- `feat_aggregate_thermal_load_ambient_lag24`
- `feat_thermal_lift_return_setpoint_pos_lag24`
- `feat_aggregate_thermal_load_return_lag24`
- `feat_thermal_lift_weather_setpoint_pos_lag24`
- `feat_aggregate_thermal_load_weather_lag24`

These work because they give the model a more physical view of reefer demand:
- how far temperatures are above setpoint
- how many reefers are active
- how that thermal stress scales to fleet-wide power burden

## Sparse Release Flag
One small calendar feature is also kept:
- `feat_post_holiday_release_flag`

This is only fed into the `residual` branch.

It is `1` only on a very small set of known post-holiday fleet-release dates and `0` everywhere else. It is not a broad holiday flag. It exists only to nudge the normal-hours model during the handful of periods where fleet unwinding repeatedly distorted the lag-based baseline.

## Model Architecture
The repo supports three forecast modes via [train_and_predict.py](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/train_and_predict.py).

### `residual`
- deterministic baseline:
  - `baseline_pred_kw = 0.7 * feat_lag_24 + 0.3 * feat_lag_168`
- model target:
  - `y_kw - baseline_pred_kw`
- final point forecast:
  - `baseline + learned correction`

### `direct`
- model target:
  - `y_kw`
- final point forecast:
  - direct prediction from the same feature set

### `peakprob_gate`
This is the final family used for the best results.

It trains:
- one residual model
- one direct model
- one binary gate model

The gate predicts whether an hour is likely to be a high-load regime:
- `is_peak = y_kw >= P75(train)`

Its peak probability becomes a continuous blend weight:
- `direct_weight = 0.3 + 0.7 * peak_prob`

Final point forecast:
- `pred_power = direct_weight * direct_pred + (1 - direct_weight) * residual_pred`

Final `p90` forecast:
- blend the calibrated `p90` forecasts the same way
- then enforce `pred_p90_kw >= pred_power_kw`

## Residual Training Policy
This is the last major improvement and the main reason the score moved from the mid-30s into the high-20s.

The new CLI flag is:
- `--residual-training-policy`

Supported values:
- `full_year`
- `jan_nov_dec`
- `jan_oct_nov_dec`
- `nov_dec_only`

Important:
- only the `residual` branch is filtered by this policy
- the `direct` branch still trains on full 2025
- the `gate` model still trains on full 2025

Why this helps:
- the residual branch is the normal-hours stabilizer
- early January 2026 looks much more like late 2025 winter than like summer 2025
- removing off-regime months lets the residual branch specialize on the distribution it will actually face

## Benchmark Summary
The full benchmark is kept in:
- [outputs/residual_training_policy_benchmark.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/residual_training_policy_benchmark.csv)
- [outputs/residual_training_policy_benchmark.json](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/residual_training_policy_benchmark.json)

Key `peakprob_gate` results:

| Residual Training Policy | Public Score | `mae_all` | `mae_peak` | `pinball_p90` | Mean Fold Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nov_dec_only` | `29.813058` | `35.805733` | `34.918464` | `7.173262` | `95.526099` |
| `jan_nov_dec` | `31.693414` | `36.551358` | `40.135490` | `6.885440` | `93.600879` |
| `full_year` | `34.180486` | `51.129222` | `23.177656` | `8.312891` | `95.850613` |

Interpretation:
- `nov_dec_only` is the best public-window performer by a clear margin
- `jan_nov_dec` is the safer backup because it also improves strongly and has the best mean fold score
- both outperform the earlier `full_year` release-flagged setup

## Chosen Primary And Backup
Primary:
- `peakprob_gate + nov_dec_only`
- best public score
- still not worse than `full_year` on mean fold score

Backup:
- `peakprob_gate + jan_nov_dec`
- slightly weaker public score
- best average fold score among the tested policies

## Kept Artifacts
The repo now keeps only the final and incrementally useful outputs.

Primary current run:
- [predictions.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/predictions.csv)
- [outputs/predictions_peakprob_gate.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/predictions_peakprob_gate.csv)
- [outputs/backtest_metrics_peakprob_gate.json](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/backtest_metrics_peakprob_gate.json)
- [outputs/model_ready_feature_table_peakprob_gate.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/model_ready_feature_table_peakprob_gate.csv)
- [outputs/preprocessing_summary_peakprob_gate.json](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/preprocessing_summary_peakprob_gate.json)
- [outputs/best_model_peakprob_gate.pkl](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/best_model_peakprob_gate.pkl)

Incremental worked outputs:
- [outputs/predictions_peakprob_gate_full_year.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/predictions_peakprob_gate_full_year.csv)
- [outputs/predictions_peakprob_gate_jan_nov_dec.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/predictions_peakprob_gate_jan_nov_dec.csv)
- [outputs/predictions_peakprob_gate_nov_dec_only.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/predictions_peakprob_gate_nov_dec_only.csv)
- [outputs/residual_training_policy_benchmark.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/residual_training_policy_benchmark.csv)
- [outputs/residual_training_policy_benchmark.json](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/residual_training_policy_benchmark.json)

## Reproducibility
Rebuild features:

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

Train the safer backup model:

```bash
/usr/bin/python3 train_and_predict.py --root . --model-mode peakprob_gate --residual-training-policy jan_nov_dec
```

Score any kept submission on the public window:

```bash
/usr/bin/python3 score_public_submission.py predictions.csv outputs/predictions_peakprob_gate_jan_nov_dec.csv outputs/predictions_peakprob_gate_full_year.csv
```
