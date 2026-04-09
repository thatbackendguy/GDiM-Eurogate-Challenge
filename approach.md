# Reefer Load Forecasting Approach

## Summary
This solution builds an hourly forecasting table from the provided reefer and weather files, trains a residual model on top of a strong lag baseline, calibrates an upper-risk estimate for `pred_p90_kw`, and exports a submission file that matches the required format.

Key implementation choices:
- Aggregate reefer load as hourly `sum(AvPowerCons) / 1000`.
- Create leak-safe features using lagged reefer load, lagged fleet-composition summaries, lagged hourly weather summaries, and calendar signals.
- Use a deterministic baseline of `0.7 * lag_24 + 0.3 * lag_168`.
- Train a gradient boosting residual model to correct that baseline, with extra weight on the top 10% highest-load training hours.
- Calibrate `pred_p90_kw` from positive validation residuals by hour-of-day and predicted-load bin, with fallbacks for sparse bins.

## Data Notes
The implementation follows the actual files in this package instead of the markdown descriptions when they differ:
- Reefer data comes from `reefer_release.csv`.
- Weather data is already unpacked under `Wetterdaten Okt 25 - 23 Feb 26/`.
- The real reefer schema uses columns such as `container_visit_uuid`, `AvPowerCons`, and `RemperatureSupply`.

The exported feature table is:
- `outputs/hourly_feature_table.csv`

The exported model-ready, preprocessed feature table is:
- `outputs/model_ready_feature_table.csv`

The preprocessing summary is:
- `outputs/preprocessing_summary.json`

The final submission file is:
- `predictions.csv`

## Features
The hourly feature table includes:
- Lag load features: 24h, 48h, 72h, and 168h.
- Rolling load statistics ending at `t-24h`.
- Delta and short-trend features from those lagged load values.
- Calendar features: hour, day of week, month, day of year, weekend flag, and cyclical hour/day encodings.
- Lagged reefer fleet summaries: active container count, mean/median temperature fields, and counts/shares by hardware type, container size, and stack tier.
- Lagged weather summaries: hourly mean/max/min for temperature and wind plus circular wind-direction features.

## Preprocessing
The model training workflow now includes a dedicated preprocessing stage fitted on the training split only in each backtest fold and on the full training history for the final model:
- replace `inf` and `-inf` with missing values
- median imputation per feature
- winsorization-style clipping at the 0.5th and 99.5th percentiles
- standardization of non-binary numeric features

This preprocessing is exported to `outputs/model_ready_feature_table.csv` so the model input can be inspected directly.

## Validation
The training script runs four rolling 7-day validation folds and scores:
- `mae_all`
- `mae_peak`
- `pinball_p90`
- combined score `0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball_p90`

Observed mean backtest metrics from `outputs/backtest_metrics.json`:
- `mae_all`: `117.63`
- `mae_peak`: `128.71`
- `pinball_p90`: `22.91`
- combined score: `102.01`

## Reproducibility
Rebuild the feature table:

```bash
python3 build_features.py --root .
```

Train, backtest, and export the submission:

```bash
python3 train_and_predict.py --root .
```

Outputs created by the pipeline:
- `outputs/hourly_feature_table.csv`
- `outputs/model_ready_feature_table.csv`
- `outputs/preprocessing_summary.json`
- `outputs/backtest_metrics.json`
- `predictions.csv`

## Model Note
The final implementation uses `GradientBoostingRegressor` for the residual model on top of the lag baseline. This keeps the leak-safe feature pipeline and calibrated `p90` workflow while relying on an explicit preprocessing step for model-ready numeric inputs.
