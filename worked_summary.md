# What Worked

This file records only the ideas that materially improved the solution and are still worth keeping.

## Score Progression
The strongest improvements came in three clear steps:

| Step | What Changed | Public Score |
| --- | --- | ---: |
| 1 | `peakprob_gate` with full-year residual training and sparse post-holiday release flag | `34.180486` |
| 2 | Same system, but residual branch trained on `jan_nov_dec` only | `31.693414` |
| 3 | Same system, but residual branch trained on `nov_dec_only` only | `29.813058` |

This is the core story of the final solution:
- the architecture mattered
- the sparse release signal mattered
- the biggest last gain came from changing what the residual branch learned from

## Shared Foundations That Kept Helping

### 1. Strict leak-safe setup
These rules stayed intact through every successful version:
- training labels only from 2025
- reefer-state features only through lagged values
- weather shifted by `24h`
- no Jan 2026 actuals in training
- time-ordered folds only

That means every kept gain was made without breaking the challenge rules.

### 2. Strong lag backbone
The consistently useful lag family was:
- `feat_lag_24`
- `feat_lag_48`
- `feat_lag_72`
- `feat_lag_168`
- `feat_roll24_*`
- `feat_roll72_*`
- `feat_roll168_*`

These features remained the backbone of all strong models.

### 3. Thermal-burden features
The most useful engineered feature family was the thermal branch:
- `feat_thermal_lift_ambient_setpoint_pos_lag24`
- `feat_aggregate_thermal_load_ambient_lag24`
- `feat_thermal_lift_return_setpoint_pos_lag24`
- `feat_aggregate_thermal_load_return_lag24`
- `feat_thermal_lift_weather_setpoint_pos_lag24`
- `feat_aggregate_thermal_load_weather_lag24`

These mattered because they gave the model a physical explanation for reefer demand:
- higher temperature lift above setpoint means more cooling effort
- more active containers means more total cooling burden

This was the first feature direction that consistently helped rather than adding noise.

## Structural Changes That Worked

### 1. Direct model
Moving beyond the forced `baseline + residual` architecture was important.

The `direct` model:
- trains directly on `y_kw`
- uses the same leak-safe feature table
- is much better at peaks than the residual model

This did not win alone, but it created a strong peak specialist that the final system depends on.

### 2. Learned gate
The best architecture family became `peakprob_gate`.

It combines:
- a residual branch
- a direct branch
- a learned gate that predicts peak likelihood

Why this worked:
- residual is better at stable hours
- direct is better at high-load hours
- a learned gate routes between them better than any fixed blend

This was the first structural step that clearly improved the combined score in a durable way.

## Targeted Calendar Signal That Worked

### Sparse post-holiday release flag
Generic holiday features hurt, but one very small holiday-related signal helped:
- `feat_post_holiday_release_flag`

Important details:
- binary
- only active on a very small set of known post-holiday release dates
- only fed into the `residual` branch

Why this worked:
- it nudged the normal-hours model during the few fleet-unwind periods that repeatedly caused lag-based overprediction
- it did so without disturbing the rest of the forecast space

This moved the best gated score from `34.916991` to `34.180486`.

## What Finally Unlocked The Big Gain

### Residual-only winter training
The biggest late improvement came from changing the residual branch’s training window, not from adding more features.

The key idea:
- keep `direct` on full 2025
- keep `gate` on full 2025
- only filter the `residual` branch’s training rows

Why that makes sense:
- the residual branch is the normal-hours stabilizer
- early January 2026 is much more like late 2025 winter than summer 2025
- training the residual branch on off-regime months forces it to compromise between very different demand patterns

### Benchmark results
The benchmark results are stored in:
- [outputs/residual_training_policy_benchmark.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/residual_training_policy_benchmark.csv)
- [outputs/residual_training_policy_benchmark.json](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/residual_training_policy_benchmark.json)

Useful comparison:

| Residual Training Policy | Public Score | `mae_all` | `mae_peak` | Mean Fold Score |
| --- | ---: | ---: | ---: | ---: |
| `full_year` | `34.180486` | `51.129222` | `23.177656` | `95.850613` |
| `jan_nov_dec` | `31.693414` | `36.551358` | `40.135490` | `93.600879` |
| `nov_dec_only` | `29.813058` | `35.805733` | `34.918464` | `95.526099` |

Interpretation:
- `jan_nov_dec` is the safer seasonal compromise
- `nov_dec_only` is the strongest overall performer
- `nov_dec_only` is not worse than `full_year` on mean fold score, so the gain is not just an obvious public-only overfit

## Final Model Choice

### Primary
Use:
- `peakprob_gate`
- `--residual-training-policy nov_dec_only`

Why:
- best public score
- much better `mae_all`
- still acceptable fold behavior

Primary submission file:
- [predictions.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/predictions.csv)

### Backup
Use:
- `peakprob_gate`
- `--residual-training-policy jan_nov_dec`

Why:
- second-best public score
- best mean fold score among the tested policies
- best “safer” seasonal alternative

Backup submission file:
- [outputs/predictions_peakprob_gate_jan_nov_dec.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/predictions_peakprob_gate_jan_nov_dec.csv)

## Kept Incremental Outputs
The repo keeps the outputs that represent the winning path, not the failed branches.

Kept prediction outputs:
- [predictions.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/predictions.csv)
- [outputs/predictions_peakprob_gate.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/predictions_peakprob_gate.csv)
- [outputs/predictions_peakprob_gate_full_year.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/predictions_peakprob_gate_full_year.csv)
- [outputs/predictions_peakprob_gate_jan_nov_dec.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/predictions_peakprob_gate_jan_nov_dec.csv)
- [outputs/predictions_peakprob_gate_nov_dec_only.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/predictions_peakprob_gate_nov_dec_only.csv)

Kept benchmark outputs:
- [outputs/residual_training_policy_benchmark.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/residual_training_policy_benchmark.csv)
- [outputs/residual_training_policy_benchmark.json](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/residual_training_policy_benchmark.json)

Kept primary run artifacts:
- [outputs/backtest_metrics_peakprob_gate.json](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/backtest_metrics_peakprob_gate.json)
- [outputs/model_ready_feature_table_peakprob_gate.csv](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/model_ready_feature_table_peakprob_gate.csv)
- [outputs/preprocessing_summary_peakprob_gate.json](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/preprocessing_summary_peakprob_gate.json)
- [outputs/best_model_peakprob_gate.pkl](/Users/ashutoshchatterjee/Documents/Projects/GDiM-Eurogate-Challenge-Yash/outputs/best_model_peakprob_gate.pkl)

## Bottom Line
The final breakthrough did not come from a new model class or from using future information.

It came from:
- a strong leak-safe shared pipeline
- thermal-burden features
- a direct model plus learned gate
- one sparse post-holiday release signal
- and, most importantly, training the residual branch on the part of 2025 that actually looks like early January 2026

That is why the solution moved from the mid-30s down to `29.813058`.
