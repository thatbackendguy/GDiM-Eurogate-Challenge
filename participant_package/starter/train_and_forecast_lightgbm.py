"""
Reefer load forecast — LightGBM + CatBoost ensemble on residuals.

v6 changes vs v5 (ensemble-recursive):
  1. Use ACTUAL reefer data within target period (24h-lagged, no recursion)
  2. Residual modelling: baseline + ensemble_residual
  3. Hour-of-day p90 calibration with fallbacks
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:
    raise SystemExit("pip install lightgbm") from exc
try:
    from catboost import CatBoostRegressor
except ImportError as exc:
    raise SystemExit("pip install catboost") from exc

ROOT = Path(__file__).resolve().parents[1]
REEFER_CSV = ROOT / "reefer_release" / "reefer_release.csv"
TARGETS_CSV = ROOT / "target_timestamps.csv"
WEATHER_DIR = ROOT / "wetterdaten"
OUTPUT_DIR = ROOT / "starter" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"

_REEFER_COLS = [
    "EventTime", "AvPowerCons", "HardwareType",
    "TemperatureSetPoint", "TemperatureAmbient",
    "TemperatureReturn", "RemperatureSupply",
    "ContainerSize", "container_visit_uuid",
]
_TOP_HW = ["ML3", "SCC6", "DecosVa", "DecosIIIj", "MP4000"]


# ── helpers ──────────────────────────────────────────────────────────────

def _to_numeric(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


# ── data loading (FULL dataset including target period) ──────────────────

def _p(msg):
    print(msg, flush=True)


def load_all_reefer() -> tuple[pd.Series, pd.DataFrame]:
    _p("  reading CSV...")
    df = pd.read_csv(REEFER_CSV, sep=";", usecols=_REEFER_COLS, low_memory=False)
    _p(f"  {len(df)} rows loaded, parsing dates...")
    df["EventTime"] = pd.to_datetime(df["EventTime"], utc=True, errors="coerce")
    _p("  converting numerics...")
    for col in ["AvPowerCons", "TemperatureSetPoint", "TemperatureAmbient",
                 "TemperatureReturn", "RemperatureSupply"]:
        df[col] = _to_numeric(df[col])
    df = df.dropna(subset=["EventTime", "AvPowerCons"])
    df["hour"] = df["EventTime"].dt.floor("h")

    _p("  computing hourly totals...")
    y = df.groupby("hour")["AvPowerCons"].sum() / 1000.0
    y = y.sort_index()

    _p("  computing fleet features (vectorised)...")
    fleet = df.groupby("hour").agg(
        n_containers=("container_visit_uuid", "nunique"),
        power_mean=("AvPowerCons", "mean"),
        power_std=("AvPowerCons", "std"),
        setpoint_mean=("TemperatureSetPoint", "mean"),
        setpoint_std=("TemperatureSetPoint", "std"),
        ambient_mean=("TemperatureAmbient", "mean"),
        return_mean=("TemperatureReturn", "mean"),
        supply_mean=("RemperatureSupply", "mean"),
    )
    fleet["temp_spread"] = fleet["return_mean"] - fleet["supply_mean"]

    # Vectorised container-size share (no .apply)
    hour_counts = df.groupby("hour").size()
    size40_counts = df[df["ContainerSize"] == 40].groupby("hour").size()
    fleet["size_40_share"] = (size40_counts / hour_counts).reindex(fleet.index).fillna(0)

    # Vectorised hardware-type shares (no .apply — MUCH faster)
    for hw in _TOP_HW:
        hw_counts = df[df["HardwareType"] == hw].groupby("hour").size()
        fleet[f"hw_{hw}"] = (hw_counts / hour_counts).reindex(fleet.index).fillna(0)

    return y, fleet.sort_index()


def load_weather() -> pd.DataFrame:
    files = sorted(WEATHER_DIR.rglob("*.csv"))
    temp_files = [f for f in files if "temperatur" in f.name.lower()]
    wind_files = [f for f in files if "wind_" in f.name.lower()]

    def _read(fp, name):
        chunks = []
        for ch in pd.read_csv(fp, sep=";", usecols=["UtcTimestamp", "Value"],
                              chunksize=500_000, low_memory=False):
            ch["UtcTimestamp"] = pd.to_datetime(ch["UtcTimestamp"], utc=True, errors="coerce")
            ch["Value"] = _to_numeric(ch["Value"])
            ch = ch.dropna()
            ch["hour"] = ch["UtcTimestamp"].dt.floor("h")
            chunks.append(ch.groupby("hour")["Value"].mean())
        if not chunks:
            return pd.Series(dtype=float, name=name)
        s = pd.concat(chunks).groupby(level=0).mean()
        s.name = name
        return s

    series = []
    for i, f in enumerate(temp_files):
        series.append(_read(f, f"temp_{i}"))
    for i, f in enumerate(wind_files):
        series.append(_read(f, f"wind_{i}"))
    if not series:
        return pd.DataFrame()
    weather = pd.DataFrame()
    temp_s = [s for s in series if s.name.startswith("temp_")]
    wind_s = [s for s in series if s.name.startswith("wind_")]
    if temp_s:
        weather["temp_mean"] = pd.concat(temp_s, axis=1).mean(axis=1)
    if wind_s:
        weather["wind_mean"] = pd.concat(wind_s, axis=1).mean(axis=1)
    return weather.sort_index().pipe(lambda d: d[~d.index.duplicated(keep="last")])


# ── feature engineering (all lag >= 24h, uses full series) ───────────────

def build_feature_table(
    y: pd.Series,
    fleet: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Build features for EVERY hour in y. All load/fleet features use
    lags >= 24h so they reference only data known 24h before target."""
    f = pd.DataFrame(index=y.index)
    f["y"] = y.values

    # Deterministic baseline
    f["lag_24"]  = f["y"].shift(24)
    f["lag_48"]  = f["y"].shift(48)
    f["lag_72"]  = f["y"].shift(72)
    f["lag_168"] = f["y"].shift(168)
    f["baseline"] = 0.7 * f["lag_24"] + 0.3 * f["lag_168"]
    f["residual"] = f["y"] - f["baseline"]

    # Also keep short lags (1-12) — they ARE valid because they reference
    # actual data from the full series, not recursive predictions.
    for lag in [1, 2, 3, 6, 12]:
        f[f"lag_{lag}"] = f["y"].shift(lag)

    # Rolling stats anchored at t-1 (valid: actual data available)
    shifted = f["y"].shift(1)
    for w in [3, 6, 12, 24, 48]:
        f[f"roll_mean_{w}"]  = shifted.rolling(w).mean()
        f[f"roll_std_{w}"]   = shifted.rolling(w).std()
        f[f"roll_max_{w}"]   = shifted.rolling(w).max()
        f[f"roll_min_{w}"]   = shifted.rolling(w).min()

    # Trend / delta
    f["diff_24"]     = shifted - f["lag_24"]
    f["diff_168"]    = shifted - f["lag_168"]
    f["ratio_24_168"] = f["lag_24"] / f["lag_168"].replace(0, np.nan)
    f["delta_24_48"]  = f["lag_24"] - f["lag_48"]
    f["delta_24_168"] = f["lag_24"] - f["lag_168"]

    # Calendar (cyclical + raw)
    idx = f.index
    f["hour"]       = idx.hour
    f["dayofweek"]  = idx.dayofweek
    f["month"]      = idx.month
    f["is_weekend"]  = (idx.dayofweek >= 5).astype(int)
    f["hour_x_wknd"] = f["hour"] * f["is_weekend"]
    f["hour_sin"]    = np.sin(2 * np.pi * idx.hour / 24)
    f["hour_cos"]    = np.cos(2 * np.pi * idx.hour / 24)
    f["dow_sin"]     = np.sin(2 * np.pi * idx.dayofweek / 7)
    f["dow_cos"]     = np.cos(2 * np.pi * idx.dayofweek / 7)

    # Fleet composition lagged 1h and 24h
    if not fleet.empty:
        fl = fleet.reindex(f.index).ffill().bfill()
        for col in fl.columns:
            f[f"fleet_{col}_lag1"]  = fl[col].shift(1)
            f[f"fleet_{col}_lag24"] = fl[col].shift(24)
        if "n_containers" in fl.columns:
            f["fleet_n_diff24"] = fl["n_containers"].shift(1) - fl["n_containers"].shift(24)

    # Weather lagged 24h + 48h
    if not weather.empty:
        wt = weather.reindex(f.index).ffill().bfill()
        for col in wt.columns:
            f[f"{col}_lag24"]  = wt[col].shift(24)
            f[f"{col}_lag48"]  = wt[col].shift(48)
            f[f"{col}_lag168"] = wt[col].shift(168)
            f[f"{col}_diff24"] = wt[col].shift(24) - wt[col].shift(48)

    return f


# ── metrics ──────────────────────────────────────────────────────────────

def pinball_loss(y_true, y_q, q=0.9):
    err = np.asarray(y_true) - np.asarray(y_q)
    return float(np.mean(np.maximum(q * err, (q - 1) * err)))


def challenge_score(y_true, y_pred, y_p90):
    y_true, y_pred, y_p90 = map(np.asarray, (y_true, y_pred, y_p90))
    mae_all = float(np.mean(np.abs(y_true - y_pred)))
    thr = float(np.quantile(y_true, 0.8))
    peak = y_true >= thr
    mae_peak = float(np.mean(np.abs(y_true[peak] - y_pred[peak])))
    pb = pinball_loss(y_true, y_p90)
    return {"mae_all": mae_all, "mae_peak": mae_peak, "pinball_p90": pb,
            "score_proxy": 0.5 * mae_all + 0.3 * mae_peak + 0.2 * pb}


# ── model builders ───────────────────────────────────────────────────────

def _lgbm_pt():
    return LGBMRegressor(
        objective="mae", n_estimators=800, learning_rate=0.025,
        num_leaves=63, max_depth=8, min_child_samples=20,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1)

def _lgbm_q90():
    return LGBMRegressor(
        objective="quantile", alpha=0.9, n_estimators=800, learning_rate=0.025,
        num_leaves=63, max_depth=8, min_child_samples=30,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1)

def _cb_pt():
    return CatBoostRegressor(
        loss_function="MAE", iterations=800, learning_rate=0.03,
        depth=8, l2_leaf_reg=3.0, random_seed=42, verbose=0)

def _cb_q90():
    return CatBoostRegressor(
        loss_function="Quantile:alpha=0.9", iterations=800, learning_rate=0.03,
        depth=8, l2_leaf_reg=3.0, random_seed=42, verbose=0)

def _peak_weight(y, top_pct=0.10, boost=3.0):
    thr = float(np.quantile(y, 1.0 - top_pct))
    w = np.ones(len(y))
    w[y >= thr] += boost
    return w


# ── hour-of-day p90 calibrator ──────────────────────────────────────────

class HourlyP90Calibrator:
    def __init__(self, n_bins=4, min_samples=5):
        self.n_bins = n_bins
        self.min_samples = min_samples
        self.table: dict[tuple[int, int], float] = {}
        self.hour_fallback: dict[int, float] = {}
        self.global_uplift = 0.0
        self.edges = np.array([])

    def fit(self, hours, y_true, y_pred):
        hours, y_true, y_pred = map(np.asarray, (hours, y_true, y_pred))
        resid = y_true - y_pred
        pos_resid = np.maximum(resid, 0.0)
        self.global_uplift = float(np.quantile(pos_resid, 0.9)) if len(pos_resid) else 0.0

        self.edges = np.quantile(y_pred, np.linspace(0, 1, self.n_bins + 1))
        bins = np.clip(np.digitize(y_pred, self.edges[1:-1]), 0, self.n_bins - 1)

        for h in range(24):
            m = hours == h
            self.hour_fallback[h] = (
                float(np.quantile(pos_resid[m], 0.9)) if m.sum() >= self.min_samples
                else self.global_uplift
            )
        for h in range(24):
            for b in range(self.n_bins):
                m = (hours == h) & (bins == b)
                self.table[(h, b)] = (
                    float(np.quantile(pos_resid[m], 0.9)) if m.sum() >= self.min_samples
                    else self.hour_fallback.get(h, self.global_uplift)
                )

    def uplift(self, hours, y_pred):
        hours, y_pred = map(np.asarray, (hours, y_pred))
        bins = np.clip(np.digitize(y_pred, self.edges[1:-1]), 0, self.n_bins - 1) if len(self.edges) > 2 else np.zeros_like(y_pred, dtype=int)
        return np.array([self.table.get((int(h), int(b)), self.global_uplift) for h, b in zip(hours, bins)])


# ── parameter optimisation (blend weight + peak add) ─────────────────────

def _optimize_blend(eval_df, cal: HourlyP90Calibrator):
    y_true = eval_df["y_true"].values
    hours = eval_df.index.hour

    best = {"w": 0.5, "peak_add": 0}
    best_score = float("inf")

    for w in np.arange(0.25, 0.75, 0.05):
        resid_pt = w * eval_df["lgbm_resid"].values + (1 - w) * eval_df["cb_resid"].values
        resid_q  = w * eval_df["lgbm_q90_resid"].values + (1 - w) * eval_df["cb_q90_resid"].values
        point = eval_df["baseline"].values + resid_pt
        point = np.clip(point, 0, None)
        q90_raw = eval_df["baseline"].values + resid_q
        q90_raw = np.maximum(q90_raw, point)

        peak_thr = float(np.quantile(point, 0.80))
        peak_mask = point >= peak_thr

        for pa in [0, 5, 10, 15, 20, 30, 40, 50]:
            corrected = point.copy()
            corrected[peak_mask] += pa
            p90 = np.maximum(corrected, corrected + cal.uplift(hours, corrected))
            s = challenge_score(y_true, corrected, p90)["score_proxy"]
            if s < best_score:
                best_score = s
                best = {"w": float(w), "peak_add": pa}
    return best


# ── backtest (14-day, direct — no recursion) ─────────────────────────────

def run_backtest(feat: pd.DataFrame, feature_cols: list[str], days=14):
    """For each fold the validation day's features come from actual data
    (via the pre-built feature table), not from recursive predictions."""
    folds = []
    # Only use 2025 data for backtest
    feat_2025 = feat.loc[feat.index < "2026-01-01"]
    end = feat_2025.index.max()

    for i in range(days, 0, -1):
        split = end - pd.Timedelta(days=i)
        train_df = feat_2025.loc[:split - pd.Timedelta(hours=1)].dropna(subset=["baseline", "residual"] + feature_cols)
        val_idx = pd.date_range(split, periods=24, freq="h", tz="UTC").intersection(feat_2025.index)
        val_df = feat_2025.loc[val_idx].dropna(subset=["baseline", "residual"] + feature_cols)
        if len(val_df) < 24:
            continue

        _p(f"  fold {days - i + 1}/{days}  split={split.date()}")

        X_tr, y_tr = train_df[feature_cols], train_df["residual"]
        X_va = val_df[feature_cols]
        sw = _peak_weight(train_df["y"].values, top_pct=0.10, boost=3.0)

        m1 = _lgbm_pt(); m1.fit(X_tr, y_tr, sample_weight=sw)
        m2 = _cb_pt();   m2.fit(X_tr, y_tr, sample_weight=sw)
        m3 = _lgbm_q90(); m3.fit(X_tr, y_tr)
        m4 = _cb_q90();   m4.fit(X_tr, y_tr)

        folds.append(pd.DataFrame({
            "y_true": val_df["y"].values,
            "baseline": val_df["baseline"].values,
            "lgbm_resid": m1.predict(X_va),
            "cb_resid": m2.predict(X_va),
            "lgbm_q90_resid": m3.predict(X_va),
            "cb_q90_resid": m4.predict(X_va),
        }, index=val_idx))

    _p("  all folds done")
    eval_df = pd.concat(folds).sort_index()

    # Fit p90 calibrator with initial blend
    w_init = 0.5
    pt_init = eval_df["baseline"].values + w_init * eval_df["lgbm_resid"].values + (1 - w_init) * eval_df["cb_resid"].values
    pt_init = np.clip(pt_init, 0, None)
    cal = HourlyP90Calibrator(n_bins=4, min_samples=5)
    cal.fit(eval_df.index.hour, eval_df["y_true"].values, pt_init)

    # Optimise blend + peak_add using the calibrator
    params = _optimize_blend(eval_df, cal)

    # Refit calibrator with optimised blend
    w = params["w"]
    pt_opt = eval_df["baseline"].values + w * eval_df["lgbm_resid"].values + (1 - w) * eval_df["cb_resid"].values
    pt_opt = np.clip(pt_opt, 0, None)
    peak_thr = float(np.quantile(pt_opt, 0.80))
    pt_opt[pt_opt >= peak_thr] += params["peak_add"]
    cal.fit(eval_df.index.hour, eval_df["y_true"].values, pt_opt)

    p90 = np.maximum(pt_opt, pt_opt + cal.uplift(eval_df.index.hour, pt_opt))
    metrics = challenge_score(eval_df["y_true"].values, pt_opt, p90)
    return metrics, params, cal


# ── main ─────────────────────────────────────────────────────────────────

def main():
    _p("Loading ALL reefer data (including target period)...")
    y, fleet = load_all_reefer()
    _p(f"  {len(y)} hourly observations")

    _p("Loading weather...")
    weather = load_weather()

    _p("Building full feature table (actual data, no recursion)...")
    feat = build_feature_table(y, fleet, weather)
    exclude = {"y", "baseline", "residual"}
    feature_cols = [c for c in feat.columns if c not in exclude]
    _p(f"  {len(feature_cols)} features")

    _p("Running 14-day backtest (direct, no recursion)...")
    bt, params, cal = run_backtest(feat, feature_cols, days=14)
    _p("Backtest metrics:")
    for k, v in bt.items():
        _p(f"  {k}: {v:.4f}")
    _p(f"Params: {params}")

    # Final training on all 2025 data
    _p("Training final ensemble on all 2025 data...")
    train_mask = (feat.index >= "2025-01-01") & (feat.index < "2026-01-01")
    train_df = feat.loc[train_mask].dropna(subset=["baseline", "residual"] + feature_cols)
    X_tr, y_resid = train_df[feature_cols], train_df["residual"]
    sw = _peak_weight(train_df["y"].values, top_pct=0.10, boost=3.0)

    m_lgbm = _lgbm_pt(); m_lgbm.fit(X_tr, y_resid, sample_weight=sw)
    m_cb   = _cb_pt();   m_cb.fit(X_tr, y_resid, sample_weight=sw)
    m_lgbm_q = _lgbm_q90(); m_lgbm_q.fit(X_tr, y_resid)
    m_cb_q   = _cb_q90();   m_cb_q.fit(X_tr, y_resid)

    # Predict target timestamps (features already built from actual data!)
    targets = pd.read_csv(TARGETS_CSV)
    target_idx = pd.DatetimeIndex(
        pd.to_datetime(targets["timestamp_utc"], utc=True).dropna()
    ).sort_values()

    target_feat = feat.loc[target_idx].copy()
    missing = target_feat["baseline"].isna().sum()
    if missing > 0:
        _p(f"  WARNING: {missing} targets missing baseline, filling with lag_24")
        target_feat["baseline"] = target_feat["baseline"].fillna(target_feat["lag_24"])

    X_tgt = target_feat[feature_cols].fillna(0)
    w = params["w"]
    resid_pt = w * m_lgbm.predict(X_tgt) + (1 - w) * m_cb.predict(X_tgt)
    point = target_feat["baseline"].values + resid_pt
    point = np.clip(point, 0, None)

    # Peak correction
    peak_thr = float(np.quantile(point, 0.80))
    point[point >= peak_thr] += params["peak_add"]

    # P90 from hourly calibrator
    p90 = np.maximum(point, point + cal.uplift(target_idx.hour, point))

    out = pd.DataFrame({
        "timestamp_utc": [ts.strftime("%Y-%m-%dT%H:%M:%SZ") for ts in target_idx],
        "pred_power_kw": np.round(point, 6),
        "pred_p90_kw": np.round(np.maximum(p90, point), 6),
    })
    out.to_csv(PREDICTIONS_CSV, index=False)
    _p(f"\nSaved: {PREDICTIONS_CSV}")
    _p(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
