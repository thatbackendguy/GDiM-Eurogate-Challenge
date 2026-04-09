# Approach: Reefer Peak Load Forecasting

## Summary

We forecast hourly aggregate reefer electricity consumption (kW) for 223 target
hours spanning Jan 1-10, 2026, using a LightGBM + CatBoost ensemble trained on
12 months of historical reefer data (Jan-Dec 2025), weather data, and
reefer-mix features. Predictions are generated via recursive 24-hour-ahead
forecasting. All model development was assisted by an LLM (Claude, via Cursor
IDE).

## Data used

| Source | Period | Role |
|---|---|---|
| `reefer_release.csv` | Jan 2025 - Dec 2025 | Training target (hourly aggregated kW) + reefer-mix features |
| `wetterdaten/*.csv` | Oct 2025 - Feb 2026 | Hourly temperature + wind (lagged) |
| `target_timestamps.csv` | Jan 1-10 2026 | Prediction targets |

Training data is cut off at **Dec 31, 2025 23:00** (1 hour before the first
target). No actual Jan 2026 data is used in model training.

## Feature engineering (76 features)

- **Load lags**: 1, 2, 3, 6, 12, 24, 48, 72, 168 hours
- **Rolling statistics**: mean, std, max, min over 3/6/12/24/48-hour windows
  (shifted by 1h to avoid leakage)
- **Trend features**: diff_24, diff_168, ratio of lag_24 to lag_168
- **Calendar**: hour-of-day, day-of-week, month, weekend flag, hour x weekend
  interaction
- **Weather** (lagged 24/48/168h): mean temperature, mean wind speed, 24h
  temperature change
- **Reefer-mix** (lagged 1h and 24h): active container count, mean/std
  per-container power, setpoint mean/std, ambient/return/supply temperature,
  40ft container share, top-5 hardware type shares, container count change

## Model architecture

### Point forecast (`pred_power_kw`)
- **LightGBM** (MAE objective, 800 trees, lr=0.025) + **CatBoost** (MAE, 800
  iterations, lr=0.03)
- Both trained with **peak-weighted samples** (top 20% load hours get 3x
  weight)
- Blended at optimised weight (w=0.40 LightGBM, 0.60 CatBoost)
- **Peak correction**: +30 kW additive boost on hours predicted above the 80th
  percentile threshold

### Upper estimate (`pred_p90_kw`)
- **LightGBM** (quantile alpha=0.9) + **CatBoost** (quantile alpha=0.9)
- Blended at same weight, then calibrated: `p90 = max(point, 1.0 * q90 + 30)`
- Calibration parameters (a=1.0, b=30) selected on backtest to minimise
  pinball loss

### Recursive forecasting
Predictions are generated hour-by-hour. Each predicted value feeds back into
the feature vector as lag data for subsequent hours. This avoids using any
future actuals during the target period.

## Validation

- **14-day rolling backtest** (Dec 17-30, 2025): for each day, the model
  trains on all prior data and predicts 24 hours ahead
- **Organiser-style proxy score**: `0.5 * mae_all + 0.3 * mae_peak + 0.2 *
  pinball_p90`
- **Joint parameter optimisation**: blend weight, peak correction, and p90
  calibration are tuned together over a grid of 4536 combinations, directly
  minimising the weighted proxy score

### Backtest results

| Metric | Score |
|---|---|
| `mae_all` | 89.44 |
| `mae_peak` | 172.65 |
| `pinball_p90` | 22.61 |
| **score_proxy** | **101.04** |

## Data integrity

- No future data leakage: all lag features reference timestamps strictly before
  the target hour
- Lag features within the target period come from the model's own recursive
  predictions, not from actual reefer data
- Submission passes all organiser checks: 223 rows, no duplicates, non-negative
  values, `pred_p90_kw >= pred_power_kw`

## LLM usage

An LLM (Claude) was used throughout model development via the Cursor IDE:

1. **Data exploration**: profiling the reefer CSV schema, identifying column
   types, missingness, and the `RemperatureSupply` typo
2. **Feature engineering**: designing lag/rolling/interaction features, deciding
   which weather signals to include, creating reefer-mix aggregate features
3. **Model selection**: comparing LightGBM vs CatBoost trade-offs, choosing
   ensemble architecture
4. **Scoring analysis**: decomposing the weighted score to prioritise
   improvements (peak MAE dominated at 53% of total score)
5. **Code generation**: the full training/backtest/prediction pipeline was
   iteratively developed with LLM assistance
6. **Validation**: designing the leakage audit, date-range verification, and
   submission constraint checks

## How to reproduce

```bash
pip install pandas numpy lightgbm catboost
python starter/train_and_forecast_lightgbm.py
```

Output: `starter/output/predictions.csv`

The script automatically reads `reefer_release.csv`, `wetterdaten/`, and
`target_timestamps.csv` from the parent directory. To rerun on a different
target list, replace `target_timestamps.csv`.
