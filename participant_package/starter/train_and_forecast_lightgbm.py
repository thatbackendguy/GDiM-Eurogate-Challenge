from __future__ import annotations

from pathlib import Path
from typing import Iterable

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numeric(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_reefer_raw() -> pd.DataFrame:
    df = pd.read_csv(REEFER_CSV, sep=";", usecols=_REEFER_COLS, low_memory=False)
    df["EventTime"] = pd.to_datetime(df["EventTime"], utc=True, errors="coerce")
    for col in ["AvPowerCons", "TemperatureSetPoint", "TemperatureAmbient",
                 "TemperatureReturn", "RemperatureSupply"]:
        df[col] = _to_numeric(df[col])
    df = df.dropna(subset=["EventTime", "AvPowerCons"])
    df["hour"] = df["EventTime"].dt.floor("h")
    return df


def load_hourly_series_and_mix() -> tuple[pd.Series, pd.DataFrame]:
    df = _load_reefer_raw()

    y = df.groupby("hour", as_index=True)["AvPowerCons"].sum() / 1000.0
    y = y.sort_index()

    agg = df.groupby("hour").agg(
        n_containers=("container_visit_uuid", "nunique"),
        power_mean=("AvPowerCons", "mean"),
        power_std=("AvPowerCons", "std"),
        setpoint_mean=("TemperatureSetPoint", "mean"),
        setpoint_std=("TemperatureSetPoint", "std"),
        ambient_mean=("TemperatureAmbient", "mean"),
        return_mean=("TemperatureReturn", "mean"),
        supply_mean=("RemperatureSupply", "mean"),
        size_40_share=("ContainerSize", lambda x: (x == 40).mean()),
    )
    agg["temp_spread"] = agg["return_mean"] - agg["supply_mean"]

    for hw in _TOP_HW:
        hw_share = df.groupby("hour").apply(
            lambda g, h=hw: (g["HardwareType"] == h).mean(),
            include_groups=False,
        )
        agg[f"hw_{hw}_share"] = hw_share

    return y, agg.sort_index()


def _read_weather_hourly(file_path: Path, out_col: str) -> pd.Series:
    chunks = []
    for chunk in pd.read_csv(
        file_path, sep=";", usecols=["UtcTimestamp", "Value"],
        chunksize=500_000, low_memory=False,
    ):
        chunk["UtcTimestamp"] = pd.to_datetime(chunk["UtcTimestamp"], utc=True, errors="coerce")
        chunk["Value"] = _to_numeric(chunk["Value"])
        chunk = chunk.dropna(subset=["UtcTimestamp", "Value"])
        chunk["hour"] = chunk["UtcTimestamp"].dt.floor("h")
        chunks.append(chunk.groupby("hour", as_index=True)["Value"].mean())
    if not chunks:
        return pd.Series(dtype=float, name=out_col)
    s = pd.concat(chunks).groupby(level=0).mean()
    s.name = out_col
    return s


def load_weather_features() -> pd.DataFrame:
    files = sorted(WEATHER_DIR.rglob("*.csv"))
    temp_files = [f for f in files if "temperatur" in f.name.lower()]
    wind_files = [f for f in files if "wind_" in f.name.lower()]

    temp_series = [_read_weather_hourly(f, f"temp_{i}") for i, f in enumerate(temp_files, start=1)]
    wind_series = [_read_weather_hourly(f, f"wind_{i}") for i, f in enumerate(wind_files, start=1)]

    weather = pd.DataFrame()
    if temp_series:
        weather["temp_mean"] = pd.concat(temp_series, axis=1).mean(axis=1)
    if wind_series:
        weather["wind_mean"] = pd.concat(wind_series, axis=1).mean(axis=1)

    weather = weather.sort_index()
    return weather[~weather.index.duplicated(keep="last")]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def make_features(
    y: pd.Series,
    weather_hourly: pd.DataFrame | None = None,
    reefer_mix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(index=y.index)
    frame["y"] = y.values

    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        frame[f"lag_{lag}"] = frame["y"].shift(lag)

    shifted = frame["y"].shift(1)
    for window in [3, 6, 12, 24, 48]:
        frame[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        frame[f"roll_std_{window}"] = shifted.rolling(window).std()
        frame[f"roll_max_{window}"] = shifted.rolling(window).max()
        frame[f"roll_min_{window}"] = shifted.rolling(window).min()

    frame["diff_24"] = shifted - frame["y"].shift(24)
    lag24 = frame["y"].shift(24)
    lag168 = frame["y"].shift(168)
    frame["ratio_24_168"] = lag24 / lag168.replace(0, np.nan)
    frame["diff_168"] = shifted - lag168

    if weather_hourly is not None and not weather_hourly.empty:
        aligned = weather_hourly.reindex(frame.index).ffill().bfill()
        for col in aligned.columns:
            frame[f"{col}_lag24"] = aligned[col].shift(24)
            frame[f"{col}_lag48"] = aligned[col].shift(48)
            frame[f"{col}_lag168"] = aligned[col].shift(168)
            frame[f"{col}_diff24"] = aligned[col].shift(24) - aligned[col].shift(48)

    if reefer_mix is not None and not reefer_mix.empty:
        mix = reefer_mix.reindex(frame.index).ffill().bfill()
        for col in mix.columns:
            frame[f"rmix_{col}_lag1"] = mix[col].shift(1)
            frame[f"rmix_{col}_lag24"] = mix[col].shift(24)
        if "n_containers" in mix.columns:
            frame["rmix_n_containers_diff24"] = mix["n_containers"].shift(1) - mix["n_containers"].shift(24)

    idx = frame.index
    frame["hour"] = idx.hour
    frame["dayofweek"] = idx.dayofweek
    frame["month"] = idx.month
    frame["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    frame["hour_x_weekend"] = frame["hour"] * frame["is_weekend"]
    return frame


# ---------------------------------------------------------------------------
# Metrics (organizer formula)
# ---------------------------------------------------------------------------

def pinball_loss(y_true: np.ndarray, y_pred_q: np.ndarray, q: float = 0.9) -> float:
    err = y_true - y_pred_q
    return float(np.mean(np.maximum(q * err, (q - 1.0) * err)))


def challenge_score(
    y_true: np.ndarray, y_pred: np.ndarray, y_pred_p90: np.ndarray,
) -> dict[str, float]:
    mae_all = float(np.mean(np.abs(y_true - y_pred)))
    peak_thr = float(np.quantile(y_true, 0.8))
    peak_mask = y_true >= peak_thr
    mae_peak = float(np.mean(np.abs(y_true[peak_mask] - y_pred[peak_mask])))
    pb = pinball_loss(y_true, y_pred_p90, q=0.9)
    return {
        "mae_all": mae_all,
        "mae_peak": mae_peak,
        "pinball_p90": pb,
        "score_proxy": 0.5 * mae_all + 0.3 * mae_peak + 0.2 * pb,
    }


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _lgbm_point() -> LGBMRegressor:
    return LGBMRegressor(
        objective="mae", n_estimators=800, learning_rate=0.025,
        num_leaves=63, max_depth=8, min_child_samples=20,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1,
    )


def _lgbm_q90() -> LGBMRegressor:
    return LGBMRegressor(
        objective="quantile", alpha=0.9,
        n_estimators=800, learning_rate=0.025,
        num_leaves=63, max_depth=8, min_child_samples=30,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1,
    )


def _cb_point() -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MAE", iterations=800, learning_rate=0.03,
        depth=8, l2_leaf_reg=3.0, random_seed=42, verbose=0,
    )


def _cb_q90() -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="Quantile:alpha=0.9", iterations=800, learning_rate=0.03,
        depth=8, l2_leaf_reg=3.0, random_seed=42, verbose=0,
    )


def _peak_weight(y: pd.Series, boost: float = 2.0) -> np.ndarray:
    thr = float(np.quantile(y.values, 0.8))
    w = np.ones(len(y), dtype=float)
    w[y.values >= thr] += boost
    return w


# ---------------------------------------------------------------------------
# Recursive forecasting
# ---------------------------------------------------------------------------

def _forecast_single(
    model, known_y: pd.Series, future_index: Iterable[pd.Timestamp],
    feature_cols: list[str],
    weather: pd.DataFrame | None = None,
    rmix: pd.DataFrame | None = None,
) -> pd.Series:
    y_work = known_y.copy()
    preds: dict[pd.Timestamp, float] = {}
    for ts in future_index:
        y_tmp = pd.concat([y_work, pd.Series([np.nan], index=[ts])])
        feat = make_features(y_tmp, weather, rmix).loc[[ts], feature_cols]
        pred = max(0.0, float(model.predict(feat)[0]))
        preds[ts] = pred
        y_work.loc[ts] = pred
    return pd.Series(preds).sort_index()


def _forecast_ensemble(
    models_weights: list[tuple[object, float]],
    known_y: pd.Series, future_index: Iterable[pd.Timestamp],
    feature_cols: list[str],
    weather: pd.DataFrame | None = None,
    rmix: pd.DataFrame | None = None,
) -> pd.Series:
    """Blend multiple models inside the recursive loop so the blended
    prediction feeds back into subsequent steps."""
    y_work = known_y.copy()
    preds: dict[pd.Timestamp, float] = {}
    for ts in future_index:
        y_tmp = pd.concat([y_work, pd.Series([np.nan], index=[ts])])
        feat = make_features(y_tmp, weather, rmix).loc[[ts], feature_cols]
        blend = sum(w * float(m.predict(feat)[0]) for m, w in models_weights)
        pred = max(0.0, blend)
        preds[ts] = pred
        y_work.loc[ts] = pred
    return pd.Series(preds).sort_index()


# ---------------------------------------------------------------------------
# Joint parameter optimisation over backtest results
# ---------------------------------------------------------------------------

def _optimize_params(eval_df: pd.DataFrame) -> dict:
    y_true = eval_df["y_true"].values
    lgbm_pt = eval_df["lgbm_point"].values
    cb_pt = eval_df["cb_point"].values
    lgbm_q = eval_df["lgbm_q90"].values
    cb_q = eval_df["cb_q90"].values

    best: dict = {"w": 0.5, "peak_add": 0, "cal_a": 1.0, "cal_b": 0.0}
    best_score = float("inf")

    for w in np.arange(0.30, 0.75, 0.05):
        point = w * lgbm_pt + (1 - w) * cb_pt
        q90_raw = np.maximum(w * lgbm_q + (1 - w) * cb_q, point)
        peak_thr = float(np.quantile(point, 0.80))
        peak_mask = point >= peak_thr

        for pa in [0, 5, 10, 15, 20, 30, 40, 50]:
            corrected = point.copy()
            corrected[peak_mask] += pa
            for ca in [0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]:
                for cb in [0, 5, 10, 20, 30, 45, 60, 80, 100]:
                    p90 = np.maximum(corrected, ca * q90_raw + cb)
                    s = challenge_score(y_true, corrected, p90)["score_proxy"]
                    if s < best_score:
                        best_score = s
                        best = {"w": float(w), "peak_add": pa,
                                "cal_a": ca, "cal_b": cb}
    return best


# ---------------------------------------------------------------------------
# Rolling backtest  (14 x 24-hour ahead, 4 models per fold)
# ---------------------------------------------------------------------------

def run_backtest(
    y_hist: pd.Series, weather: pd.DataFrame, rmix: pd.DataFrame,
    days: int = 14, peak_boost: float = 2.0,
) -> tuple[dict[str, float], dict]:
    folds: list[pd.DataFrame] = []

    for i in range(days, 0, -1):
        split = y_hist.index.max() - pd.Timedelta(days=i)
        train_y = y_hist.loc[: split - pd.Timedelta(hours=1)]
        val_idx = pd.date_range(split, periods=24, freq="h", tz="UTC")
        val_idx = val_idx.intersection(y_hist.index)
        if len(val_idx) < 24:
            continue

        print(f"  fold {days - i + 1}/{days}  split={split.date()}", end="\r")

        tr = make_features(train_y, weather, rmix).dropna()
        fc = [c for c in tr.columns if c != "y"]
        sw = _peak_weight(tr["y"], boost=peak_boost)

        m_lgbm_pt = _lgbm_point(); m_lgbm_pt.fit(tr[fc], tr["y"], sample_weight=sw)
        m_lgbm_q  = _lgbm_q90();   m_lgbm_q.fit(tr[fc], tr["y"])
        m_cb_pt   = _cb_point();    m_cb_pt.fit(tr[fc], tr["y"], sample_weight=sw)
        m_cb_q    = _cb_q90();      m_cb_q.fit(tr[fc], tr["y"])

        kw = dict(feature_cols=fc, weather=weather, rmix=rmix)
        p_lgbm = _forecast_single(m_lgbm_pt, train_y, val_idx, **kw)
        p_cb   = _forecast_single(m_cb_pt,   train_y, val_idx, **kw)
        q_lgbm = _forecast_single(m_lgbm_q,  train_y, val_idx, **kw)
        q_cb   = _forecast_single(m_cb_q,    train_y, val_idx, **kw)

        folds.append(pd.DataFrame({
            "y_true": y_hist.loc[val_idx].values,
            "lgbm_point": p_lgbm.values,
            "cb_point": p_cb.values,
            "lgbm_q90": q_lgbm.values,
            "cb_q90": q_cb.values,
        }, index=val_idx))

    print()
    eval_df = pd.concat(folds).sort_index()
    params = _optimize_params(eval_df)

    w = params["w"]
    point = w * eval_df["lgbm_point"].values + (1 - w) * eval_df["cb_point"].values
    q90_raw = np.maximum(w * eval_df["lgbm_q90"].values + (1 - w) * eval_df["cb_q90"].values, point)
    peak_thr = float(np.quantile(point, 0.80))
    corrected = point.copy()
    corrected[point >= peak_thr] += params["peak_add"]
    p90 = np.maximum(corrected, params["cal_a"] * q90_raw + params["cal_b"])

    metrics = challenge_score(eval_df["y_true"].values, corrected, p90)
    return metrics, params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading reefer data...")
    y, rmix = load_hourly_series_and_mix()
    print(f"  {len(y)} hourly observations, {len(rmix.columns)} mix features")

    print("Loading weather data...")
    weather = load_weather_features()

    targets = pd.read_csv(TARGETS_CSV)
    target_idx = pd.DatetimeIndex(
        pd.to_datetime(targets["timestamp_utc"], utc=True, errors="coerce").dropna()
    ).sort_values()

    train_y = y.loc[: target_idx.min() - pd.Timedelta(hours=1)]
    if len(train_y) < 24 * 21:
        raise SystemExit("Not enough history.")

    print("Running 14-day ensemble backtest...")
    bt, params = run_backtest(train_y, weather, rmix, days=14, peak_boost=2.0)
    print("Backtest metrics:")
    for k, v in bt.items():
        print(f"  {k}: {v:.4f}")
    print(f"Optimised params: {params}")

    # --- final training on all pre-target history ---
    print("Training final ensemble...")
    tr = make_features(train_y, weather, rmix).dropna()
    fc = [c for c in tr.columns if c != "y"]
    sw = _peak_weight(tr["y"], boost=2.0)
    print(f"  {len(fc)} features")

    m_lgbm_pt = _lgbm_point(); m_lgbm_pt.fit(tr[fc], tr["y"], sample_weight=sw)
    m_lgbm_q  = _lgbm_q90();   m_lgbm_q.fit(tr[fc], tr["y"])
    m_cb_pt   = _cb_point();    m_cb_pt.fit(tr[fc], tr["y"], sample_weight=sw)
    m_cb_q    = _cb_q90();      m_cb_q.fit(tr[fc], tr["y"])

    w = params["w"]

    # Use ensemble-recursive for point predictions (blended feedback)
    print("Generating point predictions (ensemble-recursive)...")
    pred_power = _forecast_ensemble(
        [(m_lgbm_pt, w), (m_cb_pt, 1 - w)],
        train_y, target_idx, fc, weather, rmix,
    )

    # Use ensemble-recursive for p90
    print("Generating p90 predictions (ensemble-recursive)...")
    pred_p90_raw = _forecast_ensemble(
        [(m_lgbm_q, w), (m_cb_q, 1 - w)],
        train_y, target_idx, fc, weather, rmix,
    )

    point = pred_power.values.copy()
    q90_raw = np.maximum(pred_p90_raw.values, point)

    # Peak correction
    peak_thr = float(np.quantile(point, 0.80))
    point[pred_power.values >= peak_thr] += params["peak_add"]

    # P90 calibration
    p90 = np.maximum(point, params["cal_a"] * q90_raw + params["cal_b"])

    out = pd.DataFrame({
        "timestamp_utc": [ts.strftime("%Y-%m-%dT%H:%M:%SZ") for ts in target_idx],
        "pred_power_kw": np.clip(point, 0, None).round(6),
        "pred_p90_kw": np.maximum(p90, point).round(6),
    })
    out.to_csv(PREDICTIONS_CSV, index=False)
    print(f"\nSaved: {PREDICTIONS_CSV}")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
