# Approach: Reefer Peak Load Forecasting

## Summary

We forecast hourly aggregate reefer electricity consumption (kW) for 223 target
hours spanning Jan 1-10, 2026, using a **residual-modelling** approach: a
deterministic baseline (`0.7 * lag_24 + 0.3 * lag_168`) plus a
**LightGBM + CatBoost ensemble** that predicts the residual. The model trains
exclusively on **Jan-Dec 2025** data. Predictions for the target period are
generated via **recursive forecasting** (each predicted hour feeds back as lag
data for subsequent hours). P90 estimates use flat calibration (`a * q90 + b`)
tuned on backtesting. All development was assisted by an LLM (Claude, via
Cursor IDE).

## Data used

| Source | Period used | Role |
|---|---|---|
| `reefer_release.csv` | Jan 1 - Dec 31, 2025 only | Training target (hourly kW) + fleet features |
| `wetterdaten/*.csv` | All available (Oct 2025 - Feb 2026) | Hourly temp + wind (lagged 24h+) |
| `target_timestamps.csv` | Jan 1-10, 2026 | Prediction targets |

**No Jan 2026 reefer data is used** for training or features. The reefer CSV is
explicitly filtered to `EventTime < 2026-01-01`. Weather data for the target
period is used only with 24h+ lags.

## Feature engineering (82 features)

- **Deterministic baseline**: `0.7 * lag_24 + 0.3 * lag_168`
- **Load lags**: 1, 2, 3, 6, 12, 24, 48, 72, 168 hours
- **Rolling statistics**: mean, std, max, min over 3/6/12/24/48-hour windows
  (anchored at t-1)
- **Trend features**: diff_24, diff_168, delta_24_48, delta_24_168,
  ratio_24_168
- **Calendar**: hour, day-of-week, month, weekend flag, hour x weekend,
  cyclical sin/cos encodings (hour, day-of-week)
- **Weather** (lagged 24h/48h/168h): mean temperature, mean wind speed, 24h
  temperature change
- **Fleet composition** (lagged 1h and 24h): active container count, mean/std
  per-container power, setpoint mean/std, ambient/return/supply temperature,
  temp spread, 40ft container share, top-5 hardware type shares, container
  count delta-24

## Model architecture

### Target variable
The ensemble predicts the **residual** (`y - baseline`), not the raw load.
This forces the models to focus on what the simple baseline gets wrong.

### Point forecast (`pred_power_kw`)
- **LightGBM** (MAE, 800 trees, lr=0.025) + **CatBoost** (MAE, 800 iters,
  lr=0.03), both trained with **peak-weighted samples** (top 10% hours get 4x
  weight)
- Blended: `point = baseline + w * lgbm_resid + (1-w) * cb_resid`
  (optimised w=0.70)
- **Peak correction**: optimised additive boost on predicted-high hours

### Upper estimate (`pred_p90_kw`)
- **LightGBM** (quantile a=0.9) + **CatBoost** (quantile a=0.9) for residual
- Flat calibration: `p90 = max(point, cal_a * q90_raw + cal_b)`
- Parameters (cal_a=1.0, cal_b=10) tuned to minimise pinball loss on backtest

### Recursive forecasting
For the target period, predictions are generated hour-by-hour. Each predicted
value is appended to the working series and used as lag data for subsequent
hours. This respects the constraint that no Jan 2026 reefer data is used.

## Validation

- **14-day rolling backtest** (Dec 17-30, 2025): each fold trains on all prior
  2025 data and predicts the next 24 hours
- **Organiser-style proxy score**: `0.5 * mae_all + 0.3 * mae_peak + 0.2 *
  pinball_p90`
- **Joint parameter optimisation**: blend weight (w), peak correction, and p90
  calibration (a, b) are tuned together over a grid, directly minimising the
  combined score

### Backtest results

| Metric | v5 (old ensemble) | **v7b (residual, 2025-only)** |
|---|---|---|
| `mae_all` | 89.44 | **19.93** |
| `mae_peak` | 172.65 | **28.75** |
| `pinball_p90` | 22.61 | **5.76** |
| **score_proxy** | **101.04** | **19.74** |

Note: backtest uses direct (non-recursive) evaluation with actual lag features
from within the 2025 validation windows. Production predictions use recursive
forecasting, which may produce somewhat higher errors due to compounding.

## Data integrity

- **No Jan 2026 reefer data**: the reefer CSV is filtered to `EventTime <
  2026-01-01` before any processing
- All lag features during the target period come from the model's own recursive
  predictions, not from actual reefer data
- Weather features use lags >= 24h
- Submission passes all organiser checks: 223 rows, no duplicates, non-negative
  values, `pred_p90_kw >= pred_power_kw`

## LLM usage

An LLM (Claude) was used throughout model development via the Cursor IDE:

1. **Data exploration**: profiling the reefer CSV schema, identifying column
   types, missingness, and the `RemperatureSupply` typo
2. **Feature engineering**: designing lag/rolling/interaction features, deciding
   which weather signals to include, creating fleet-composition features
3. **Model selection**: comparing LightGBM vs CatBoost trade-offs, choosing
   ensemble architecture, pivoting to residual modelling
4. **Scoring analysis**: decomposing the weighted score to prioritise
   improvements (peak MAE was the largest contributor)
5. **Code generation**: the full training/backtest/prediction pipeline was
   iteratively developed with LLM assistance across 7+ versions
6. **Validation**: leakage audit confirming no Jan 2026 data used, date-range
   verification, submission constraint checks
7. **Data integrity analysis**: verified that using actual target-period data
   gives unrealistically good scores; chose to use 2025-only data for honest
   forecasting

## How to reproduce

```bash
pip install pandas numpy lightgbm catboost
python starter/train_and_forecast_lightgbm.py
```

Output: `starter/output/predictions.csv`

The script reads `reefer_release.csv`, `wetterdaten/`, and
`target_timestamps.csv` from the parent directory. Runtime: ~20 minutes.
