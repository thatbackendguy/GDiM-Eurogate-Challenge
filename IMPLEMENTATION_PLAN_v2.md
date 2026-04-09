# Reefer Power Forecasting — Implementation Plan v2

---

## What Actually Matters (Read This First)

The challenge docs change our priorities significantly. Here's what's real:

### Scoring Formula
```
SCORE = 0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball_p90
```
**Lower is better.** That's it. No bonus for dashboards, no bonus for LLM usage, no bonus for pretty charts. The leaderboard is purely this number.

### What We Actually Submit
```csv
timestamp_utc, pred_power_kw, pred_p90_kw
```
That's the entire deliverable. Plus `approach.md` and reproducible code/notebook.

### Critical Constraints
- **No future data leakage** — treat as 24h-ahead forecast. You cannot use any information from after the target hour.
- **Code must be re-runnable** — organizers rerun your code on hidden timestamps + full reefer data for private scoring. If it breaks, you lose.
- **pred_p90_kw >= pred_power_kw** always.
- Only supplied data allowed (reefer_release.zip, wetterdaten.zip).

### Data We Have (Not What We Assumed)
The reefer data is **per-container, per-hour**, not pre-aggregated terminal totals. Columns:
- `ContainerVisitID`, `ContainerIdentification` — individual container tracking
- `Power` (watts), `Energy` (watt-hours) — per container per hour
- `TemperatureSetPoint`, `TemperatureAmbient`, `TemperatureReturn`, `TemperatureSupply` — per container temps
- `HardwareType` — reefer controller model (ML2, ML3, etc.)
- `ContainerSize` — ISO 6346 size code
- `LocationRack` — position at terminal
- `EventTime` — hourly UTC timestamp

**This means Step 1 is aggregation.** We must sum individual container power to get total terminal demand per hour. The per-container features (setpoint, ambient temp, hardware type) become aggregation-level features.

Weather data is separate (`wetterdaten.zip`) — likely DWD/local station data.

### What We Do NOT Need
- ~~Streamlit dashboard~~ — not scored
- ~~Claude API integration~~ — not scored
- ~~What-if simulator~~ — not scored
- ~~Presentation slides~~ — approach.md is enough

**Every hour goes into prediction accuracy.**

---

## Scoring Deep Dive — Where Points Are Won and Lost

### Component 1: mae_all (weight 0.5)
Mean absolute error across ALL target hours. This is half your score. A model that's decent everywhere beats one that's great on easy hours but bad on a few.

### Component 2: mae_peak (weight 0.3)
MAE on high-load hours only. The organizers define which hours are "high-load" (likely top quartile or similar). Being wrong on peaks hurts 0.3/0.5 = 60% as much per-hour as being wrong overall. Since peaks are harder to predict, this component likely separates teams.

### Component 3: pinball_p90 (weight 0.2)
Pinball loss for the 90th quantile estimate. The pinball loss function:
```
If actual <= pred_p90: loss = 0.1 * (pred_p90 - actual)    # overshoot penalty (small)
If actual >  pred_p90: loss = 0.9 * (actual - pred_p90)    # undershoot penalty (large)
```
Being too low on P90 is **9x worse** than being too high. This means P90 should err on the side of being generous, but not absurdly so (overshoot still costs 10%).

### Optimal Strategy
1. Get P50 as accurate as possible everywhere (drives mae_all)
2. Get P50 especially accurate on high-load hours (drives mae_peak)
3. Set P90 slightly above where you'd expect — the asymmetric loss means overshooting is cheap

---

## The Plan: 7 Incremental Levels

Each level produces a valid submission. Each level strictly improves on the previous one. **Commit and submit after every level.**

```
Level 0 (Hr 0–2)    Naive baseline              → valid submission, terrible score
Level 1 (Hr 2–5)    Aggregation + EDA           → understand the data, informed baseline
Level 2 (Hr 5–9)    LightGBM with core features → competitive model
Level 3 (Hr 9–12)   Feature expansion           → squeeze out accuracy
Level 4 (Hr 12–15)  Peak-hour focus             → attack mae_peak component
Level 5 (Hr 15–18)  P90 optimization            → attack pinball_p90 component
Level 6 (Hr 18–20)  Ensemble + hardening        → final polish
```

---

## Level 0: Naive Baseline (Hours 0–2)

> **Goal:** Valid submission CSV in under 2 hours. Insurance policy.

### Hour 0–0.5: Unpack and Inventory

```python
import pandas as pd
import zipfile
import os

# Unpack everything
for f in ['reefer_release.zip', 'wetterdaten.zip']:
    with zipfile.ZipFile(f) as z:
        z.extractall('data/')

# See what we got
for root, dirs, files in os.walk('data/'):
    for file in files:
        path = os.path.join(root, file)
        size = os.path.getsize(path) / 1e6
        print(f"{path:60s} {size:.1f} MB")

# Load targets — this defines what we must predict
targets = pd.read_csv('target_timestamps.csv', parse_dates=['timestamp_utc'])
print(f"Target hours: {len(targets)}")
print(f"Range: {targets['timestamp_utc'].min()} → {targets['timestamp_utc'].max()}")
print(f"Unique dates: {targets['timestamp_utc'].dt.date.nunique()}")
```

**Immediately answer:**
- How many hours do we predict?
- What date range are targets?
- How far ahead from the latest reefer data are the targets? (This tells us the actual forecast horizon.)

### Hour 0.5–1: Load and Aggregate Reefer Data

```python
# Load reefer data — might be multiple CSVs, might be large
# Adapt based on what's in the zip
reefer = pd.read_csv('data/reefer_release/...', parse_dates=['EventTime'])

# CRITICAL FIRST STEP: aggregate per-container → terminal total
# Power is in watts — convert to kW
reefer['power_kw'] = reefer['Power'] / 1000

hourly_total = reefer.groupby('EventTime').agg(
    total_power_kw=('power_kw', 'sum'),
    n_reefers=('ContainerVisitID', 'nunique'),
    avg_ambient_temp=('TemperatureAmbient', 'mean'),
    avg_setpoint=('TemperatureSetPoint', 'mean'),
    avg_return_temp=('TemperatureReturn', 'mean'),
    avg_supply_temp=('TemperatureSupply', 'mean'),
).sort_index()

print(f"Aggregated hours: {len(hourly_total)}")
print(f"Range: {hourly_total.index.min()} → {hourly_total.index.max()}")
print(hourly_total.describe())
```

### Hour 1–1.5: Naive Baseline (Yesterday's Same Hour)

```python
# The challenge's own suggested baseline
# pred_power_kw = same hour, one day ago
# pred_p90_kw = 1.10 * pred_power_kw

def naive_baseline(hourly_total, target_timestamps):
    predictions = []
    for ts in target_timestamps:
        yesterday = ts - pd.Timedelta(hours=24)
        if yesterday in hourly_total.index:
            pred = hourly_total.loc[yesterday, 'total_power_kw']
        else:
            # Fallback: same hour, 2 days ago
            two_days = ts - pd.Timedelta(hours=48)
            if two_days in hourly_total.index:
                pred = hourly_total.loc[two_days, 'total_power_kw']
            else:
                # Last resort: overall hourly average
                hour = ts.hour
                pred = hourly_total.groupby(hourly_total.index.hour)['total_power_kw'].mean()[hour]
        
        predictions.append({
            'timestamp_utc': ts,
            'pred_power_kw': pred,
            'pred_p90_kw': pred * 1.10  # 10% uplift as challenge suggests
        })
    
    return pd.DataFrame(predictions)

submission = naive_baseline(hourly_total, targets['timestamp_utc'])
```

### Hour 1.5–2: Validate and Save

**Submission template format (must match exactly):**
```csv
timestamp_utc,pred_power_kw,pred_p90_kw
2026-01-01T00:00:00Z,1234.56,1360.00
2026-01-01T01:00:00Z,1225.10,1345.00
```

Key format requirements:
- `timestamp_utc` must be ISO 8601 with `T` separator and `Z` suffix
- `pred_power_kw` and `pred_p90_kw` as floats, 2 decimal places
- Column order: `timestamp_utc`, `pred_power_kw`, `pred_p90_kw`

```python
def format_submission(sub):
    """
    Enforce exact template format before saving.
    Pandas default datetime output uses space separator and no Z — fix this.
    """
    out = sub.copy()
    
    # Force ISO 8601 format: 2026-01-01T00:00:00Z
    out['timestamp_utc'] = pd.to_datetime(out['timestamp_utc']).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Round to 2 decimal places
    out['pred_power_kw'] = out['pred_power_kw'].round(2)
    out['pred_p90_kw'] = out['pred_p90_kw'].round(2)
    
    # Enforce column order
    out = out[['timestamp_utc', 'pred_power_kw', 'pred_p90_kw']]
    
    return out


def validate_submission(sub, targets):
    """
    Replicates every check the organizers will run.
    """
    # Parse both sides to datetime for comparison (handles format differences)
    sub_ts = set(pd.to_datetime(sub['timestamp_utc']))
    target_ts = set(pd.to_datetime(targets['timestamp_utc']))
    
    missing = target_ts - sub_ts
    extra = sub_ts - target_ts
    assert len(missing) == 0, f"Missing {len(missing)} timestamps: {list(missing)[:5]}"
    assert len(extra) == 0, f"Extra {len(extra)} timestamps not in target list"
    
    assert sub.duplicated(subset='timestamp_utc').sum() == 0, \
        "Duplicate timestamps found"
    assert (sub['pred_power_kw'].apply(float) >= 0).all(), \
        "Negative pred_power_kw values"
    assert (sub['pred_p90_kw'].apply(float) >= 0).all(), \
        "Negative pred_p90_kw values"
    assert (sub['pred_p90_kw'].apply(float) >= sub['pred_power_kw'].apply(float)).all(), \
        "pred_p90_kw < pred_power_kw violation"
    assert sub['pred_power_kw'].notna().all(), "NaN in pred_power_kw"
    assert sub['pred_p90_kw'].notna().all(), "NaN in pred_p90_kw"
    
    # Verify ISO format
    sample = sub['timestamp_utc'].iloc[0]
    assert 'T' in str(sample) and str(sample).endswith('Z'), \
        f"Timestamp format wrong: got '{sample}', expected '2026-01-01T00:00:00Z'"
    
    print(f"✓ All checks passed. {len(sub)} rows.")
    print(f"  Timestamp range: {sub['timestamp_utc'].iloc[0]} → {sub['timestamp_utc'].iloc[-1]}")
    print(f"  pred_power_kw:   {sub['pred_power_kw'].min():.2f} – {sub['pred_power_kw'].max():.2f}")
    print(f"  pred_p90_kw:     {sub['pred_p90_kw'].min():.2f} – {sub['pred_p90_kw'].max():.2f}")


# Format, validate, save
submission = format_submission(submission)
validate_submission(submission, targets)
submission.to_csv('outputs/predictions.csv', index=False)

# Verify the written file looks right
print("\nFirst 3 lines of saved CSV:")
with open('outputs/predictions.csv') as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i >= 3:
            break
```

### ✅ Level 0 Checkpoint
```
[x] Valid predictions.csv with every target timestamp
[x] pred_power_kw = yesterday's same hour
[x] pred_p90_kw = 1.10 * pred_power_kw
[x] All submission checks pass
>>> SUBMIT THIS NOW as insurance
```

---

## Level 1: Aggregation + EDA (Hours 2–5)

> **Goal:** Understand the data deeply. Build an informed mental model of what drives demand. Every insight here translates to better features later.

### Hour 2–3: Thorough EDA

```python
import matplotlib.pyplot as plt

# 1. Daily profile — what does a typical day look like?
hourly_avg = hourly_total.groupby(hourly_total.index.hour)['total_power_kw'].agg(['mean', 'std', 'min', 'max'])
hourly_avg[['mean']].plot(title='Average Hourly Power Profile')

# 2. Weekday vs weekend
hourly_total['dow'] = hourly_total.index.dayofweek
for dow in range(7):
    subset = hourly_total[hourly_total['dow'] == dow]
    profile = subset.groupby(subset.index.hour)['total_power_kw'].mean()
    plt.plot(profile, label=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow])
plt.legend()
plt.title('Daily Profile by Day of Week')

# 3. Reefer count vs power — how strong is this relationship?
plt.scatter(hourly_total['n_reefers'], hourly_total['total_power_kw'], alpha=0.1)
plt.xlabel('Reefer Count')
plt.ylabel('Total Power kW')
correlation = hourly_total[['n_reefers', 'total_power_kw']].corr().iloc[0,1]
plt.title(f'Count vs Power (r={correlation:.3f})')

# 4. Temperature effect
plt.scatter(hourly_total['avg_ambient_temp'], hourly_total['total_power_kw'], alpha=0.1)
plt.xlabel('Ambient Temperature °C')
plt.ylabel('Total Power kW')

# 5. What are "high load" hours? — understanding where mae_peak matters
p75 = hourly_total['total_power_kw'].quantile(0.75)
p90 = hourly_total['total_power_kw'].quantile(0.90)
print(f"75th percentile: {p75:.0f} kW")
print(f"90th percentile: {p90:.0f} kW")

high_load = hourly_total[hourly_total['total_power_kw'] > p75]
print(f"High-load hours by time of day:\n{high_load.index.hour.value_counts().sort_index()}")
print(f"High-load hours by day of week:\n{high_load['dow'].value_counts().sort_index()}")

# 6. Per-container analysis — are there distinct consumption profiles?
per_container = reefer.groupby('HardwareType')['power_kw'].agg(['mean', 'std', 'count'])
print(per_container.sort_values('count', ascending=False))

container_sizes = reefer.groupby('ContainerSize')['power_kw'].agg(['mean', 'std', 'count'])
print(container_sizes.sort_values('count', ascending=False))
```

### Hour 3–4: Weather Data Integration

```python
# Load weather data
weather = pd.read_csv('data/wetterdaten/...', parse_dates=['...'])
# Adapt column names based on actual file

# Align timestamps with reefer data
# Check: is weather data hourly? Same timezone (UTC)?
print(f"Weather range: {weather.index.min()} → {weather.index.max()}")
print(f"Weather columns: {weather.columns.tolist()}")

# Merge weather into hourly_total
hourly_total = hourly_total.merge(weather, left_index=True, right_index=True, how='left')

# Check alignment
print(f"Missing weather after merge: {hourly_total['temperature_c'].isna().sum()}")
```

### Hour 4–5: Smarter Baseline + Validation Framework

**Build the validation framework you'll use for all subsequent levels:**

```python
def evaluate_model(actual, predicted, predicted_p90, high_load_mask=None):
    """
    Replicate the exact scoring formula.
    """
    mae_all = np.mean(np.abs(actual - predicted))
    
    if high_load_mask is not None:
        mae_peak = np.mean(np.abs(actual[high_load_mask] - predicted[high_load_mask]))
    else:
        # Approximate: treat top 25% of actuals as high-load
        threshold = np.percentile(actual, 75)
        peak_mask = actual >= threshold
        mae_peak = np.mean(np.abs(actual[peak_mask] - predicted[peak_mask]))
    
    # Pinball loss for P90 (quantile 0.9)
    error = actual - predicted_p90
    pinball_p90 = np.mean(np.where(error > 0, 0.9 * error, -0.1 * error))
    
    score = 0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball_p90
    
    return {
        'mae_all': mae_all,
        'mae_peak': mae_peak,
        'pinball_p90': pinball_p90,
        'combined_score': score,
    }
```

**Walk-forward validation — simulate the actual competition:**
```python
def walk_forward_evaluate(hourly_total, predict_fn, n_days=14):
    """
    For each of the last n_days:
      - Pretend everything before that day is history
      - Predict the 24 hours of that day
      - Compare to actuals
    """
    results = []
    all_actuals, all_preds, all_p90s = [], [], []
    
    end_date = hourly_total.index.max().normalize()
    
    for day_offset in range(n_days, 0, -1):
        target_date = end_date - pd.Timedelta(days=day_offset)
        target_hours = pd.date_range(target_date, periods=24, freq='H')
        
        # History = everything before target_date
        history = hourly_total[hourly_total.index < target_date]
        
        # Get actuals for this day
        actuals_day = hourly_total.loc[
            hourly_total.index.isin(target_hours), 'total_power_kw'
        ]
        
        if len(actuals_day) < 24:
            continue
        
        # Predict
        preds, p90s = predict_fn(history, target_hours)
        
        all_actuals.extend(actuals_day.values)
        all_preds.extend(preds)
        all_p90s.extend(p90s)
    
    metrics = evaluate_model(
        np.array(all_actuals), np.array(all_preds), np.array(all_p90s)
    )
    return metrics
```

**Smarter naive baseline — average of yesterday + last week:**
```python
def predict_avg_naive(history, target_hours):
    preds, p90s = [], []
    for ts in target_hours:
        vals = []
        for lag in [24, 48, 168]:  # yesterday, 2 days ago, last week
            ref = ts - pd.Timedelta(hours=lag)
            if ref in history.index:
                vals.append(history.loc[ref, 'total_power_kw'])
        
        pred = np.mean(vals) if vals else history['total_power_kw'].mean()
        preds.append(pred)
        p90s.append(pred * 1.15)  # 15% uplift — slightly better than 10%
    
    return preds, p90s

metrics_naive = walk_forward_evaluate(hourly_total, predict_avg_naive)
print(f"Naive baseline score: {metrics_naive['combined_score']:.2f}")
print(metrics_naive)
```

### ✅ Level 1 Checkpoint
```
[x] Complete understanding of data structure and patterns
[x] Weather data loaded and aligned
[x] Validation framework matching competition scoring exactly
[x] Smarter naive baseline (avg of multiple lags)
[x] Baseline score established — all improvements measured against this
>>> Submit improved baseline if score beats Level 0
```

---

## Level 2: LightGBM Core Model (Hours 5–9)

> **Goal:** Proper ML model with core features. This is where the big accuracy jump happens.

### Hour 5–6.5: Feature Engineering — Aggregated Level

The key insight: we must build features on the **aggregated hourly total** since that's what we predict. But we can also derive aggregate features from the per-container data.

```python
def build_features(hourly_total):
    df = hourly_total.copy()
    
    # ── Calendar (always known for future hours) ──
    df['hour'] = df.index.hour
    df['dow'] = df.index.dayofweek
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    # ── Lags (known at forecast time: 24h+ ago) ──
    # CRITICAL: For 24h-ahead forecasting, we can ONLY use lags >= 24h
    # lag_1h, lag_2h are NOT available — they leak future info
    df['lag_24h'] = df['total_power_kw'].shift(24)
    df['lag_48h'] = df['total_power_kw'].shift(48)
    df['lag_168h'] = df['total_power_kw'].shift(168)
    df['lag_336h'] = df['total_power_kw'].shift(336)
    
    # ── Same-hour lags (compare this hour across days) ──
    # These are powerful because demand has strong hourly patterns
    df['lag_24h_48h_avg'] = (df['lag_24h'] + df['lag_48h']) / 2
    df['lag_week_avg'] = (df['lag_168h'] + df['lag_336h']) / 2
    
    # ── Rolling stats (must use shift(24) to avoid leakage) ──
    # These capture "what was the recent trend as of yesterday"
    df['rolling_mean_24h'] = df['total_power_kw'].shift(24).rolling(24, min_periods=12).mean()
    df['rolling_std_24h'] = df['total_power_kw'].shift(24).rolling(24, min_periods=12).std()
    df['rolling_mean_168h'] = df['total_power_kw'].shift(24).rolling(168, min_periods=48).mean()
    df['rolling_max_24h'] = df['total_power_kw'].shift(24).rolling(24, min_periods=12).max()
    
    # ── Reefer count (shift by 24h to avoid leakage) ──
    df['reefer_count_lag24'] = df['n_reefers'].shift(24)
    df['reefer_count_lag168'] = df['n_reefers'].shift(168)
    df['reefer_delta_24h'] = df['n_reefers'].shift(24) - df['n_reefers'].shift(48)
    
    # ── Temperature (from weather data — available as forecast for future) ──
    df['temp_ambient'] = df['avg_ambient_temp']  # or from weather file
    df['temp_squared'] = df['temp_ambient'] ** 2
    df['cooling_degree'] = (df['temp_ambient'] - 18).clip(lower=0)
    
    # ── Reefer-derived aggregates (shifted 24h) ──
    df['avg_setpoint_lag24'] = df['avg_setpoint'].shift(24)
    df['power_per_reefer_lag24'] = (
        df['total_power_kw'].shift(24) / df['n_reefers'].shift(24).replace(0, np.nan)
    ).fillna(0)
    
    return df
```

**LEAKAGE WARNING — critical for this competition:**
```
┌──────────────────────────────────────────────────────────┐
│ WHAT YOU KNOW AT FORECAST TIME (24h ahead):              │
│                                                          │
│ ✓ Everything ≥24h in the past                            │
│ ✓ Calendar features for the target hour (trivially)      │
│ ✓ Weather FORECAST for the target hour                   │
│                                                          │
│ WHAT YOU DO NOT KNOW:                                    │
│                                                          │
│ ✗ Actual reefer count at the target hour                 │
│ ✗ Anything <24h before the target                        │
│ ✗ Actual weather at the target hour (only forecast)      │
│                                                          │
│ Using lag_1h, lag_2h, or current reefer count as         │
│ features = DATA LEAKAGE = your model will fail on        │
│ private scoring when organizers rerun your code.         │
└──────────────────────────────────────────────────────────┘
```

### Hour 6.5–8: Train LightGBM

```python
from lightgbm import LGBMRegressor

TARGET = 'total_power_kw'

# Feature list — strict no-leakage set
FEATURES = [
    # Calendar
    'hour', 'dow', 'is_weekend', 'hour_sin', 'hour_cos',
    # Lags (all >=24h)
    'lag_24h', 'lag_48h', 'lag_168h', 'lag_336h',
    'lag_24h_48h_avg', 'lag_week_avg',
    # Rolling (shifted 24h)
    'rolling_mean_24h', 'rolling_std_24h', 'rolling_mean_168h', 'rolling_max_24h',
    # Reefer count (shifted 24h)
    'reefer_count_lag24', 'reefer_count_lag168', 'reefer_delta_24h',
    # Temperature
    'temp_ambient', 'temp_squared', 'cooling_degree',
    # Reefer-derived
    'avg_setpoint_lag24', 'power_per_reefer_lag24',
]

# Build feature matrix, drop warmup rows
df = build_features(hourly_total)
df = df.dropna(subset=['lag_336h'])  # need 2 weeks of history minimum

# Temporal split
split_date = df.index.max() - pd.Timedelta(days=14)
train = df[df.index <= split_date]
val = df[df.index > split_date]

# Train P50
model_p50 = LGBMRegressor(
    objective='quantile', alpha=0.5,
    n_estimators=500, learning_rate=0.05,
    num_leaves=31, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    verbose=-1
)
model_p50.fit(train[FEATURES], train[TARGET])

# Train P90
model_p90 = LGBMRegressor(
    objective='quantile', alpha=0.9,
    n_estimators=500, learning_rate=0.05,
    num_leaves=31, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    verbose=-1
)
model_p90.fit(train[FEATURES], train[TARGET])

# Validate
val_p50 = model_p50.predict(val[FEATURES])
val_p90 = model_p90.predict(val[FEATURES])
val_p90 = np.maximum(val_p90, val_p50)  # enforce ordering

metrics = evaluate_model(val[TARGET].values, val_p50, val_p90)
print(f"LightGBM score: {metrics['combined_score']:.2f}")
print(f"  mae_all:     {metrics['mae_all']:.2f}")
print(f"  mae_peak:    {metrics['mae_peak']:.2f}")
print(f"  pinball_p90: {metrics['pinball_p90']:.2f}")
```

### Hour 8–9: Generate Submission + Feature Importance

```python
def generate_submission(models, hourly_total, target_timestamps, features_list):
    """
    For each target hour, build feature row using available history.
    """
    df = build_features(hourly_total)
    
    predictions = []
    for ts in target_timestamps:
        if ts in df.index and df.loc[ts, FEATURES].notna().all():
            row = df.loc[[ts], FEATURES]
            p50 = models['p50'].predict(row)[0]
            p90 = models['p90'].predict(row)[0]
        else:
            # Fallback to naive for missing feature rows
            lag24 = ts - pd.Timedelta(hours=24)
            if lag24 in hourly_total.index:
                p50 = hourly_total.loc[lag24, 'total_power_kw']
            else:
                p50 = hourly_total['total_power_kw'].mean()
            p90 = p50 * 1.15
        
        p90 = max(p90, p50)  # enforce constraint
        p50 = max(p50, 0)    # enforce non-negative
        p90 = max(p90, 0)
        
        predictions.append({
            'timestamp_utc': ts,
            'pred_power_kw': round(p50, 2),
            'pred_p90_kw': round(p90, 2),
        })
    
    return pd.DataFrame(predictions)

submission = generate_submission(
    {'p50': model_p50, 'p90': model_p90},
    hourly_total,
    targets['timestamp_utc'],
    FEATURES
)
submission = format_submission(submission)  # ISO 8601 timestamps + 2dp rounding
validate_submission(submission, targets)
submission.to_csv('outputs/predictions.csv', index=False)

# Feature importance — save for approach.md
importance = pd.Series(
    model_p50.booster_.feature_importance(importance_type='gain'),
    index=FEATURES
).sort_values(ascending=False)
print("Top 10 features by gain:")
print(importance.head(10))
```

### ✅ Level 2 Checkpoint
```
[x] LightGBM P50 + P90 trained with no leakage
[x] Walk-forward validated with competition scoring formula
[x] Feature importance extracted
[x] Submission CSV generated and validated
[x] Score should be significantly better than naive baseline
>>> Submit. This is your competitive baseline.
```

---

## Level 3: Feature Expansion (Hours 9–12)

> **Goal:** Incrementally add features, keeping only those that improve the validation score.

### Hour 9–10: Per-Container Aggregation Features

The per-container data has rich information we haven't exploited. Aggregate it into hourly features:

```python
def build_container_features(reefer_df):
    """
    Aggregate per-container data into hourly terminal-level features.
    These capture the MIX of containers, not just the count.
    """
    hourly = reefer_df.groupby('EventTime').agg(
        # Distribution of setpoints — captures cargo mix
        setpoint_std=('TemperatureSetPoint', 'std'),
        setpoint_min=('TemperatureSetPoint', 'min'),
        setpoint_max=('TemperatureSetPoint', 'max'),
        n_frozen=('TemperatureSetPoint', lambda x: (x < -10).sum()),    # deep freeze
        n_chilled=('TemperatureSetPoint', lambda x: (x >= -10).sum()),  # chilled
        
        # Temperature gap = how hard compressors work
        avg_temp_gap=('TemperatureReturn', lambda x: x.mean()),  # return - setpoint
        
        # Hardware mix — different hardware draws differently
        n_ml2=('HardwareType', lambda x: (x == 'ML2').sum()),
        n_ml3=('HardwareType', lambda x: (x == 'ML3').sum()),
        
        # Container size mix
        n_40ft=('ContainerSize', lambda x: x.astype(str).str.startswith('4').sum()),
        n_20ft=('ContainerSize', lambda x: x.astype(str).str.startswith('2').sum()),
        
        # Power distribution — detect if a few containers are drawing heavily
        power_std=('power_kw', 'std'),
        power_max=('power_kw', 'max'),
        power_median=('power_kw', 'median'),
    ).fillna(0)
    
    return hourly
```

**Shift all these by 24h before using as features (no leakage).**

### Hour 10–11: Weather Feature Expansion

```python
def build_weather_features(weather_df):
    """
    Weather features — some known in advance (forecasts), some lagged.
    """
    df = weather_df.copy()
    
    # Direct (if using weather forecasts for target hour — no leakage)
    # Check if wetterdaten includes forecast data or only historical
    df['temp_c'] = df['temperature']
    df['humidity'] = df['humidity']  # if available
    df['wind_speed'] = df['wind_speed']  # if available
    
    # Derived
    df['temp_squared'] = df['temp_c'] ** 2
    df['cooling_degree'] = (df['temp_c'] - 18).clip(lower=0)
    df['heating_degree'] = (18 - df['temp_c']).clip(lower=0)
    
    # Temporal derivatives (from historical weather — shifted 24h)
    df['temp_change_24h'] = df['temp_c'] - df['temp_c'].shift(24)
    df['temp_rolling_max_24h'] = df['temp_c'].shift(24).rolling(24).max()
    
    # Day/night temperature swing (proxy for thermal stress)
    df['temp_daily_range'] = df.groupby(df.index.date)['temp_c'].transform('max') - \
                             df.groupby(df.index.date)['temp_c'].transform('min')
    
    return df
```

### Hour 11–12: Greedy Feature Selection

**Only keep features that actually improve the score:**

```python
def greedy_feature_selection(df, base_features, candidate_features, target):
    """
    Start with base_features. Try adding each candidate one at a time.
    Keep it only if validation score improves.
    """
    current_features = list(base_features)
    current_score = train_and_evaluate(df, current_features, target)
    print(f"Base score: {current_score:.4f}")
    
    for feat in candidate_features:
        if feat in current_features:
            continue
        if df[feat].isna().sum() / len(df) > 0.2:
            print(f"  Skip {feat}: too many NaN ({df[feat].isna().mean():.1%})")
            continue
        
        trial_features = current_features + [feat]
        trial_score = train_and_evaluate(df, trial_features, target)
        
        if trial_score < current_score:
            improvement = current_score - trial_score
            current_features.append(feat)
            current_score = trial_score
            print(f"  ✓ Add {feat}: score {trial_score:.4f} (Δ={improvement:.4f})")
        else:
            print(f"  ✗ Skip {feat}: score {trial_score:.4f} (no improvement)")
    
    return current_features, current_score

# Run it
final_features, final_score = greedy_feature_selection(
    df, base_features=FEATURES,
    candidate_features=[
        'setpoint_std_lag24', 'n_frozen_lag24', 'n_chilled_lag24',
        'n_ml2_lag24', 'n_ml3_lag24', 'n_40ft_lag24', 'n_20ft_lag24',
        'power_std_lag24', 'power_max_lag24',
        'temp_change_24h', 'temp_daily_range',
        'humidity', 'wind_speed',
    ],
    target=TARGET
)
```

### ✅ Level 3 Checkpoint
```
[x] Per-container aggregation features built
[x] Weather features expanded
[x] Greedy feature selection — only improvements kept
[x] Score measurably better than Level 2
>>> Submit updated predictions
```

---

## Level 4: Peak-Hour Focus (Hours 12–15)

> **Goal:** Directly attack the mae_peak component (30% of score). Being wrong on high-load hours hurts disproportionately.

### Hour 12–13: Analyze Peak Errors

```python
# Where does the P50 model fail worst?
val_df = val.copy()
val_df['pred'] = val_p50
val_df['error'] = val_df['total_power_kw'] - val_df['pred']
val_df['abs_error'] = val_df['error'].abs()

# High-load hours
threshold = val_df['total_power_kw'].quantile(0.75)
peak_hours = val_df[val_df['total_power_kw'] >= threshold]

print(f"Peak hours avg error: {peak_hours['error'].mean():.1f} (systematic bias?)")
print(f"Peak hours MAE: {peak_hours['abs_error'].mean():.1f}")
print(f"All hours MAE:  {val_df['abs_error'].mean():.1f}")

# Is the model systematically underpredicting peaks?
print(f"Peak hours: underpredicts {(peak_hours['error'] > 0).mean():.1%} of the time")
# If >60%, the model is biased low on peaks — common with MAE/quantile 0.5 loss
```

### Hour 13–14: Peak-Aware Training Strategies

**Strategy A: Sample weighting — give peak hours more influence:**
```python
# Weight peak hours higher during training
sample_weights = np.ones(len(train))
train_peaks = train['total_power_kw'] >= train['total_power_kw'].quantile(0.75)
sample_weights[train_peaks] = 2.0  # double weight on peaks

model_p50_weighted = LGBMRegressor(
    objective='quantile', alpha=0.5,
    n_estimators=500, learning_rate=0.05,
    num_leaves=31, min_child_samples=20,
    verbose=-1
)
model_p50_weighted.fit(train[FEATURES], train[TARGET], sample_weight=sample_weights)
```

**Strategy B: Two-model approach — separate models for peak/off-peak:**
```python
# If peak hours are predictably identifiable (e.g., certain times of day),
# train a specialist model on peak hours only
peak_train = train[train['total_power_kw'] >= train['total_power_kw'].quantile(0.6)]
offpeak_train = train[train['total_power_kw'] < train['total_power_kw'].quantile(0.6)]

model_peak = LGBMRegressor(objective='quantile', alpha=0.5, ...)
model_peak.fit(peak_train[FEATURES], peak_train[TARGET])

model_offpeak = LGBMRegressor(objective='quantile', alpha=0.5, ...)
model_offpeak.fit(offpeak_train[FEATURES], offpeak_train[TARGET])

# At prediction time: use a classifier or heuristic to route to the right model
# Simplest heuristic: if lag_24h > P75 threshold, use peak model
```

**Strategy C: Post-hoc peak adjustment:**
```python
# If peaks are systematically underpredicted, apply a correction
peak_bias = peak_hours['error'].mean()  # positive = underprediction
if peak_bias > 0:
    # For predicted high hours, add correction
    pred_high_mask = val_p50 > np.percentile(val_p50, 70)
    val_p50_adjusted = val_p50.copy()
    val_p50_adjusted[pred_high_mask] += peak_bias * 0.7  # partial correction
```

### Hour 14–15: Evaluate All Strategies, Pick Winner

```python
strategies = {
    'base': (val_p50_base, val_p90_base),
    'weighted': (val_p50_weighted, val_p90_weighted),
    'two_model': (val_p50_twomodel, val_p90_twomodel),
    'adjusted': (val_p50_adjusted, val_p90_adjusted),
}

for name, (p50, p90) in strategies.items():
    metrics = evaluate_model(val[TARGET].values, p50, p90)
    print(f"{name:12s} | score={metrics['combined_score']:.4f} | "
          f"mae_all={metrics['mae_all']:.2f} | mae_peak={metrics['mae_peak']:.2f}")

# Pick the strategy with best COMBINED score
# Don't pick based on mae_peak alone — it might hurt mae_all
```

### ✅ Level 4 Checkpoint
```
[x] Peak error analysis done — understand failure modes
[x] Multiple peak-aware strategies tested
[x] Best strategy selected by combined score
[x] mae_peak component meaningfully improved
>>> Submit updated predictions
```

---

## Level 5: P90 Optimization (Hours 15–18)

> **Goal:** Attack the pinball_p90 component (20% of score). The asymmetric loss means P90 strategy is different from P50.

### Hour 15–16: Understand the Pinball Loss

```python
# The pinball loss at quantile 0.9:
# If actual <= pred_p90: loss = 0.1 * (pred_p90 - actual)    # overshoot, small penalty
# If actual >  pred_p90: loss = 0.9 * (actual - pred_p90)    # undershoot, 9x penalty
#
# Optimal P90: should cover ~90% of actuals. If it covers more, you're
# paying unnecessary overshoot penalty. If less, undershoot kills you.

# Check current P90 coverage
coverage = np.mean(val[TARGET].values <= val_p90) * 100
print(f"Current P90 coverage: {coverage:.1f}% (target: ~90%)")
print(f"Current pinball loss: {metrics['pinball_p90']:.2f}")

if coverage < 85:
    print("→ P90 too aggressive (too low). Widen the band.")
elif coverage > 95:
    print("→ P90 too conservative (too high). Tighten the band.")
else:
    print("→ P90 coverage is reasonable.")
```

### Hour 16–17: Calibrate P90

**Approach A: Tune the quantile model separately:**
```python
# Try different alpha values for P90 model
for alpha in [0.85, 0.88, 0.90, 0.92, 0.95]:
    model_test = LGBMRegressor(
        objective='quantile', alpha=alpha,
        n_estimators=500, learning_rate=0.05,
        num_leaves=31, verbose=-1
    )
    model_test.fit(train[FEATURES], train[TARGET])
    p90_test = np.maximum(model_test.predict(val[FEATURES]), val_p50)
    
    coverage = np.mean(val[TARGET].values <= p90_test) * 100
    error = val[TARGET].values - p90_test
    pinball = np.mean(np.where(error > 0, 0.9 * error, -0.1 * error))
    
    print(f"alpha={alpha:.2f} | coverage={coverage:.1f}% | pinball={pinball:.2f}")
```

**Approach B: Conformal-style calibration:**
```python
# Use validation residuals to set P90 as: P50 + calibrated_margin
val_residuals = val[TARGET].values - val_p50
# Find the 90th percentile of residuals
margin = np.percentile(val_residuals, 90)
print(f"Conformal P90 margin: {margin:.2f} kW")

# Apply: P90 = P50 + margin (constant across all hours)
val_p90_conformal = val_p50 + margin

# Improvement: make margin hour-dependent
for hour in range(24):
    mask = val.index.hour == hour
    hour_residuals = val_residuals[mask]
    hour_margin = np.percentile(hour_residuals, 90)
    print(f"Hour {hour:02d}: margin = {hour_margin:.2f} kW")
    # Peaks need wider margins, off-peak need narrower
```

**Approach C: Hybrid — LightGBM P90 model with conformal correction:**
```python
# Train P90 model, then calibrate residuals
raw_p90 = model_p90.predict(val[FEATURES])
p90_residuals = val[TARGET].values - raw_p90

# If P90 systematically underpredicts, shift up
p90_bias = np.percentile(p90_residuals, 90) - np.percentile(p90_residuals, 50)
corrected_p90 = raw_p90 + max(0, np.percentile(p90_residuals, 15))
# Tune this offset to hit ~90% coverage and minimize pinball
```

### Hour 17–18: Final P90 Selection

```python
p90_strategies = {
    'lgbm_raw': val_p90_raw,
    'lgbm_tuned_alpha': val_p90_tuned,
    'conformal_constant': val_p90_conformal,
    'conformal_hourly': val_p90_hourly,
    'hybrid': val_p90_hybrid,
}

for name, p90 in p90_strategies.items():
    p90_clipped = np.maximum(p90, val_p50)  # enforce ordering
    error = val[TARGET].values - p90_clipped
    pinball = np.mean(np.where(error > 0, 0.9 * error, -0.1 * error))
    coverage = np.mean(val[TARGET].values <= p90_clipped) * 100
    print(f"{name:25s} | pinball={pinball:.4f} | coverage={coverage:.1f}%")

# Pick winner and regenerate submission
```

### ✅ Level 5 Checkpoint
```
[x] P90 coverage analyzed and calibrated
[x] Multiple P90 strategies tested
[x] Pinball loss minimized
[x] All three score components now optimized
>>> Submit updated predictions
```

---

## Level 6: Ensemble + Final Hardening (Hours 18–20)

> **Goal:** Squeeze out last bits of accuracy. Ensure code reproducibility for private scoring.

### Hour 18–19: Ensemble Methods

```python
# Try blending LightGBM with a simple second model for diversity

# Model 2: Ridge regression (simple, captures different patterns)
from sklearn.linear_model import QuantileRegressor

model_ridge = QuantileRegressor(quantile=0.5, alpha=1.0, solver='highs')
model_ridge.fit(train[FEATURES], train[TARGET])
val_ridge = model_ridge.predict(val[FEATURES])

# Model 3: LightGBM with different hyperparameters (depth/leaves)
model_deep = LGBMRegressor(
    objective='quantile', alpha=0.5,
    n_estimators=800, learning_rate=0.03,
    num_leaves=63, min_child_samples=10,
    verbose=-1
)
model_deep.fit(train[FEATURES], train[TARGET])
val_deep = model_deep.predict(val[FEATURES])

# Blend — find optimal weights
from scipy.optimize import minimize

def blend_score(weights):
    w1, w2, w3 = weights
    blended = w1 * val_p50 + w2 * val_ridge + w3 * val_deep
    return np.mean(np.abs(val[TARGET].values - blended))

result = minimize(blend_score, x0=[0.5, 0.25, 0.25],
                  constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1},
                  bounds=[(0,1)]*3, method='SLSQP')

print(f"Optimal weights: LightGBM={result.x[0]:.2f}, "
      f"Ridge={result.x[1]:.2f}, Deep={result.x[2]:.2f}")
print(f"Blended MAE: {result.fun:.2f} vs Single: {metrics['mae_all']:.2f}")

# Only use blend if it actually improves
```

### Hour 19–19.5: Code Reproducibility Check

```python
"""
CRITICAL: Organizers rerun your code on hidden timestamps.
Test that your pipeline works end-to-end with different targets.
"""

# Simulate organizer rerun:
# 1. Start from raw data files
# 2. Generate features
# 3. Train model (or load saved model — clarify with organizers)
# 4. Predict for any list of timestamps

def full_pipeline(reefer_path, weather_path, target_path, output_path):
    """
    Single entry point. Organizers run:
      python predict.py --reefer data/reefer_release/ \
                        --weather data/wetterdaten/ \
                        --targets target_timestamps.csv \
                        --output predictions.csv
    """
    # Load data
    hourly_total = load_and_aggregate(reefer_path)
    weather = load_weather(weather_path)
    targets = pd.read_csv(target_path, parse_dates=['timestamp_utc'])
    
    # Merge weather
    hourly_total = merge_weather(hourly_total, weather)
    
    # Build features
    df = build_features(hourly_total)
    
    # Train (using ALL available data — no validation split for final model)
    train = df.dropna(subset=FEATURES)
    model_p50 = LGBMRegressor(objective='quantile', alpha=0.5, ...)
    model_p50.fit(train[FEATURES], train[TARGET])
    model_p90 = LGBMRegressor(objective='quantile', alpha=0.9, ...)
    model_p90.fit(train[FEATURES], train[TARGET])
    
    # Predict
    submission = generate_submission(
        {'p50': model_p50, 'p90': model_p90},
        hourly_total, targets['timestamp_utc'], FEATURES
    )
    
    # Format to match template exactly, validate, and save
    submission = format_submission(submission)
    validate_submission(submission, targets)
    submission.to_csv(output_path, index=False)
    return submission

# Test with our public targets
full_pipeline(
    'data/reefer_release/', 'data/wetterdaten/',
    'target_timestamps.csv', 'outputs/predictions_final.csv'
)
```

### Hour 19.5–20: Write approach.md + Final Submission

```markdown
# approach.md

## Model
LightGBM quantile regression (P50 for best estimate, P90 for upper estimate).

## Feature Engineering
- Aggregated per-container reefer data to hourly terminal totals
- Calendar features (hour, day of week, weekend flag)
- Lag features at 24h, 48h, 168h, 336h (no leakage — all ≥24h)
- Rolling statistics (mean, std, max) shifted by 24h
- Reefer count and mix features (hardware type, container size, setpoint distribution)
- Weather features (temperature, cooling degree hours)
- All features validated for no future-data leakage

## Peak-Hour Handling
[describe which peak strategy won: weighting / two-model / adjustment]

## P90 Calibration
[describe which P90 strategy won: tuned alpha / conformal / hybrid]

## Validation
Walk-forward validation over last 14 days:
- mae_all:     XX kW
- mae_peak:    XX kW
- pinball_p90: XX kW
- combined:    XX

## Reproducibility
Run: `python predict.py --reefer <path> --weather <path> --targets <path> --output predictions.csv`
```

### ✅ Level 6 Final Checkpoint
```
[x] Ensemble tested (use only if it helps)
[x] Full pipeline tested end-to-end
[x] Code runs from scratch with different target timestamps
[x] approach.md written
[x] Final predictions.csv generated and validated

SUBMIT:
  1. predictions.csv
  2. approach.md
  3. predict.py (or notebook)
```

---

## Decision Log: What We Dropped and Why

| Dropped Item | Why |
|---|---|
| Streamlit dashboard | Not scored. Zero points. |
| Claude API integration | Not in scoring formula. Only matters for a presentation bonus if one exists. Can be described in approach.md verbally. |
| What-if simulator | Not scored. |
| P10 model | Challenge asks only for best + upper estimate. |
| Recursive forecasting for short lags | With 24h-ahead constraint, short lags (1h, 2h) are unavailable anyway — problem doesn't exist. |
| Cyclical encoding (sin/cos) | Test via feature selection. Only keep if it helps score. |
| Interaction features (temp × reefers) | Test in Level 3 greedy selection. Only keep if it helps. |

---

## Time Budget Summary

```
Hours 0–2    Level 0: Naive baseline                   → SUBMIT
Hours 2–5    Level 1: EDA + validation framework       → SUBMIT if improved
Hours 5–9    Level 2: LightGBM core model              → SUBMIT (big jump)
Hours 9–12   Level 3: Feature expansion                → SUBMIT if improved
Hours 12–15  Level 4: Peak-hour optimization            → SUBMIT if improved
Hours 15–18  Level 5: P90 calibration                   → SUBMIT if improved
Hours 18–20  Level 6: Ensemble + hardening + submit     → FINAL SUBMIT
```

Every level is independently submittable. If you run out of time at hour 9, you still have a competitive model. The later levels have diminishing returns but could separate you from teams that also built a LightGBM.

---

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Data leakage in features | Fatal — model fails on private scoring | Strict rule: ALL lags ≥ 24h. No exceptions. |
| Code not reproducible | Fatal — organizers can't score you | Test `full_pipeline()` from scratch at Hour 19 |
| Reefer data too large for memory | Blocks everything | Load in chunks, aggregate immediately, discard raw |
| Weather data format mismatch | Bad temperature features | Inspect `wetterdaten.zip` in first 30 min |
| NaN in target hour features | Missing predictions | Fallback to naive baseline for any missing row |
| Overfitting from too many features | Bad private score | Greedy selection in Level 3 acts as regularizer |
| P90 too wide or too narrow | Bad pinball loss | Explicit calibration in Level 5 |
| Peak hours defined differently than assumed | Bad mae_peak | Test multiple thresholds (P75, P80, P90) in validation |
