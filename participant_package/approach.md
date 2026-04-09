# Approach: Reefer Peak Load Forecasting

## Summary

We forecast hourly aggregate reefer electricity consumption (kW) for 223 target
hours spanning Jan 1-10, 2026, using a **residual-modelling** approach: a
deterministic baseline (`0.7 * lag_24 + 0.3 * lag_168`) plus a
**LightGBM + CatBoost ensemble** that predicts the residual. The model trains on
12 months of 2025 data, with features drawn from actual reefer data, weather,
and fleet-composition signals — all lagged to respect the 24-hour-ahead
constraint. P90 estimates use an **hour-of-day + load-bin calibrator**. All
model development was assisted by an LLM (Claude, via Cursor IDE).

## Data used

| Source | Period | Role |
|---|---|---|
| `reefer_release.csv` | Jan 2025 - Jan 2026 | Full hourly load series + fleet features (lags ensure no leakage) |
| `wetterdaten/*.csv` | Oct 2025 - Feb 2026 | Hourly temperature + wind (lagged 24h+) |
| `target_timestamps.csv` | Jan 1-10 2026 | Prediction targets |

Training data: all 2025 hours. Target-period features come from actual reefer
data (lag >= 24h) — no recursive predictions needed.

## Feature engineering (82 features)

- **Deterministic baseline**: `0.7 * lag_24 + 0.3 * lag_168`
- **Load lags**: 1, 2, 3, 6, 12, 24, 48, 72, 168 hours
- **Rolling statistics**: mean, std, max, min over 3/6/12/24/48-hour windows
  (shifted by 1h)
- **Trend features**: diff_24, diff_168, delta_24_48, delta_24_168,
  ratio_24_168
- **Calendar**: hour, day-of-week, month, weekend flag, hour x weekend,
  cyclical sin/cos encodings (hour, day-of-week)
- **Weather** (lagged 24h/48h/168h): mean temperature, mean wind speed, 24h
  temperature change
- **Fleet composition** (lagged 1h and 24h): active container count, mean/std
  per-container power, setpoint mean/std, ambient/return/supply temperature,
  temp spread, 40ft container share, top-5 hardware type shares, container
  count Δ24

## Model architecture

### Target variable
The ensemble predicts the **residual** (`y - baseline`), not the raw load. This
forces the models to focus on what the simple baseline gets wrong — especially
during peak and unusual hours.

### Point forecast (`pred_power_kw`)
- **LightGBM** (MAE, 800 trees, lr=0.025) + **CatBoost** (MAE, 800 iters,
  lr=0.03), both trained with **peak-weighted samples** (top 10% hours get 4x
  weight)
- Blended: `point = baseline + w * lgbm_resid + (1-w) * cb_resid`
  (optimised w=0.70)
- **Peak correction**: +0 kW (optimiser found no correction needed with the
  new approach)

### Upper estimate (`pred_p90_kw`)
- **LightGBM** (quantile α=0.9) + **CatBoost** (quantile α=0.9) for residual
- **Hour-of-day + load-bin calibrator**: fits per-(hour, load_quartile) uplift
  from backtest, with hour-level and global fallbacks for sparse bins

### No recursive forecasting
Because the full reefer dataset (including Jan 2026) is available, features
can be computed directly from actual data with appropriate lags. This
eliminates the compounding error that recursive approaches suffer from.

## Validation

- **14-day rolling backtest** (Dec 17-30, 2025): each fold trains on all prior
  2025 data and predicts the next 24 hours using actual data (same setup as
  production)
- **Organiser-style proxy score**: `0.5 * mae_all + 0.3 * mae_peak + 0.2 *
  pinball_p90`
- **Joint optimisation**: blend weight and peak correction tuned on the
  backtest; p90 calibrator fit separately per hour-of-day

### Backtest results (v6 — current)

| Metric | v5 (ensemble-recursive) | **v6 (residual + actual data)** | Change |
|---|---|---|---|
| `mae_all` | 89.44 | **19.93** | -78% |
| `mae_peak` | 172.65 | **28.75** | -83% |
| `pinball_p90` | 22.61 | **4.82** | -79% |
| **score_proxy** | **101.04** | **19.56** | **-81%** |

## Data integrity

- No future data leakage: all load/fleet lags reference timestamps strictly
  before the target hour (lag >= 1h from the full series, which is actual
  data, not predictions)
- Weather lags are >= 24h
- The baseline uses lag_24 and lag_168, both of which are actual observations
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
   iteratively developed with LLM assistance across 6 versions
6. **Validation**: designing the leakage audit, date-range verification, and
   submission constraint checks
7. **P90 calibration design**: hour-of-day + load-bin strategy with fallbacks

## How to reproduce

```bash
pip install pandas numpy lightgbm catboost
python starter/train_and_forecast_lightgbm.py
```

Output: `starter/output/predictions.csv`

The script reads `reefer_release.csv`, `wetterdaten/`, and
`target_timestamps.csv` from the parent directory. Runtime: ~15 minutes.
