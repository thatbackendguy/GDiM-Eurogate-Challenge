from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
TRAIN_END = pd.Timestamp("2025-12-31 23:00:00", tz="UTC")
VAL_START = pd.Timestamp("2025-10-15 00:00:00", tz="UTC")
TRAIN_WINDOW_DAYS = 45
PUBLIC_START = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")

sys.path.insert(0, str(ROOT / "scripts"))
import eurogate_forecasting_pipeline as pipeline  # noqa: E402


def official_score(actual: np.ndarray, pred_power: np.ndarray, pred_p90: np.ndarray) -> dict[str, float]:
    peak_threshold = float(np.quantile(actual, 0.9))
    peak_mask = actual >= peak_threshold
    mae_all = float(np.mean(np.abs(actual - pred_power)))
    mae_peak = float(np.mean(np.abs(actual[peak_mask] - pred_power[peak_mask])))
    delta = actual - pred_p90
    pinball_p90 = float(np.mean(np.maximum(0.9 * delta, -0.1 * delta)))
    combined = 0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball_p90
    return {
        "public_composite_score": combined,
        "public_mae_all": mae_all,
        "public_mae_peak": mae_peak,
        "public_pinball_p90": pinball_p90,
        "peak_threshold_kw": peak_threshold,
    }


def build_feature_table() -> tuple[pd.DataFrame, list[str], list[str]]:
    reefer_hourly, _ = pipeline.build_reefer_hourly_frame()
    weather_hourly = pipeline.build_weather_hourly_frame(reefer_hourly.index)
    base = pipeline.add_temporal_features(reefer_hourly.join(weather_hourly)).copy()

    features = pd.DataFrame(index=base.index)
    features["day_of_year"] = base.index.dayofyear
    features["week_of_year"] = base.index.isocalendar().week.astype(int)
    features["day_sin"] = np.sin(2 * np.pi * features["day_of_year"] / 366.0)
    features["day_cos"] = np.cos(2 * np.pi * features["day_of_year"] / 366.0)
    features["week_sin"] = np.sin(2 * np.pi * features["week_of_year"] / 53.0)
    features["week_cos"] = np.cos(2 * np.pi * features["week_of_year"] / 53.0)
    features["is_weekend"] = (base["day_of_week"] >= 5).astype(int)

    lag_columns = [
        "target_power_kw",
        "connected_container_count",
        "temperature_setpoint_mean",
        "temperature_ambient_mean",
        "temperature_return_mean",
        "temperature_supply_mean",
        "temperatur_vc_halle3",
        "temperatur_zentralgate",
        "wind_vc_halle3",
        "wind_zentralgate",
    ]
    shared_lags = [24, 48, 72, 168]
    power_count_lags = [24, 25, 26, 27, 28, 48, 72, 96, 120, 144, 168, 192, 216, 240, 336, 504, 672, 720]

    for column in lag_columns:
        if column not in base.columns:
            continue
        lags = power_count_lags if column in {"target_power_kw", "connected_container_count"} else shared_lags
        for lag in lags:
            features[f"{column}_lag_{lag}"] = base[column].shift(lag)

    shifted_target = base["target_power_kw"].shift(24)
    for window in [24, 72, 168, 336, 720]:
        features[f"power_roll_mean_{window}"] = shifted_target.rolling(window, min_periods=window).mean()
        features[f"power_roll_std_{window}"] = shifted_target.rolling(window, min_periods=window).std()

    features = features.loc[:, ~features.columns.duplicated()].copy()
    enriched = pd.concat([base, features], axis=1)
    enriched["baseline_24"] = enriched["target_power_kw"].shift(24)
    enriched["resid_target"] = enriched["target_power_kw"] - enriched["baseline_24"]
    enriched["timestamp_utc"] = enriched.index

    feature_cols = [
        "hour_of_day",
        "day_of_week",
        "month",
        "day_of_year",
        "week_of_year",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "day_sin",
        "day_cos",
        "week_sin",
        "week_cos",
        "is_weekend",
        "baseline_24",
        *features.columns.tolist(),
    ]
    feature_cols = list(dict.fromkeys(feature_cols))

    weather_cols = [column for column in weather_hourly.columns if column in enriched.columns]
    enriched = enriched.dropna(subset=feature_cols + ["resid_target"]).copy()
    return enriched, feature_cols, weather_cols


def fit_recent_window_model(frame: pd.DataFrame, feature_cols: list[str]) -> tuple[ExtraTreesRegressor, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_window_start = TRAIN_END - pd.Timedelta(days=TRAIN_WINDOW_DAYS) + pd.Timedelta(hours=1)
    train = frame.loc[train_window_start:TRAIN_END].copy()
    validation = frame.loc[VAL_START:TRAIN_END].copy()
    january = frame.loc[PUBLIC_START:].copy()

    model = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    )
    model.fit(train[feature_cols], train["resid_target"])
    return model, train, validation, january


def calibrate_p90_by_prediction_bucket(
    validation_actual: np.ndarray,
    validation_pred: np.ndarray,
    january_pred: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    validation_residual = validation_actual - validation_pred
    default_uplift = max(0.0, float(np.quantile(validation_residual, 0.9)))

    validation_bins = pd.qcut(pd.Series(validation_pred), 10, duplicates="drop")
    bucket_uplift = pd.Series(validation_residual).groupby(validation_bins).quantile(0.9).clip(lower=0.0)
    right_edges = validation_bins.cat.categories.right.to_numpy(dtype=float)
    january_bins = pd.cut(
        pd.Series(january_pred),
        bins=np.unique(np.r_[-np.inf, right_edges]),
        include_lowest=True,
    )
    january_uplift = january_bins.map(bucket_uplift).fillna(default_uplift).to_numpy(dtype=float)
    pred_p90 = np.maximum(january_pred, january_pred + january_uplift)

    summary = {
        "default_uplift_kw": round(default_uplift, 6),
        "bucket_count": int(len(bucket_uplift)),
        "bucket_uplift_kw_mean": round(float(bucket_uplift.mean()), 6),
        "bucket_uplift_kw_max": round(float(bucket_uplift.max()), 6),
    }
    return pred_p90, summary


def build_score_table(team_name: str, metrics: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": team_name,
                "submission_scope": "public_only",
                "status": "ok",
                "public_composite_score": round(metrics["public_composite_score"], 6),
                "public_mae_all": round(metrics["public_mae_all"], 6),
                "public_mae_peak": round(metrics["public_mae_peak"], 6),
                "public_pinball_p90": round(metrics["public_pinball_p90"], 6),
                "error": "",
            }
        ]
    )


def run_pipeline() -> dict[str, object]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    frame, feature_cols, weather_cols = build_feature_table()
    model, train, validation, january = fit_recent_window_model(frame, feature_cols)

    january_pred = np.maximum(january["baseline_24"].to_numpy(dtype=float) + model.predict(january[feature_cols]), 0.0)
    validation_pred = np.maximum(
        validation["baseline_24"].to_numpy(dtype=float) + model.predict(validation[feature_cols]),
        0.0,
    )
    pred_p90, p90_summary = calibrate_p90_by_prediction_bucket(
        validation["target_power_kw"].to_numpy(dtype=float),
        validation_pred,
        january_pred,
    )

    metrics = official_score(
        january["target_power_kw"].to_numpy(dtype=float),
        january_pred,
        pred_p90,
    )

    predictions = pd.DataFrame(
        {
            "timestamp_utc": january["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pred_power_kw": january_pred,
            "pred_p90_kw": pred_p90,
        }
    )
    predictions.to_csv(OUTPUT_DIR / "january_optimized_predictions.csv", index=False)

    leakage = {
        "training_cutoff_utc": TRAIN_END.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "training_window_days": TRAIN_WINDOW_DAYS,
        "max_training_timestamp_utc": train["timestamp_utc"].max().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "min_training_timestamp_utc": train["timestamp_utc"].min().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "january_rows_used_for_model_fit": 0,
        "uses_future_relative_to_target": False,
        "rule_interpretation": "Uses only observations available at least 24 hours before each target hour. January 2026 actuals are never used for model fitting.",
        "allowed_lag_floor_hours": 24,
    }

    payload = {
        "method": "recent_window_extratrees_residual_no_leakage",
        "official_formula_from_markdown": "0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball_p90",
        "note": "Point model is trained on the last 45 days of 2025 only, using legal 24h-ahead lag features. January 2026 actuals are used only for evaluation after prediction generation.",
        "feature_count": len(feature_cols),
        "weather_feature_count": len(weather_cols),
        "selected_model": {
            "family": "ExtraTreesRegressor",
            "n_estimators": 500,
            "min_samples_leaf": 2,
            "residual_target": "target_power_kw - lag_24",
            "training_window_days": TRAIN_WINDOW_DAYS,
        },
        "p90_calibration": {
            "method": "validation_prediction_bucket_quantiles",
            **p90_summary,
        },
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "january_eval_rows": int(len(january)),
        "metrics": {key: round(value, 6) for key, value in metrics.items()},
        "leakage_check": leakage,
        "output_predictions_csv": str(OUTPUT_DIR / "january_optimized_predictions.csv"),
    }
    (OUTPUT_DIR / "final_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "leakage_check.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")

    score_table = build_score_table("reefer_recent_etr_45d_clean", metrics)
    score_table.to_csv(OUTPUT_DIR / "score_table.csv", index=False)
    return payload


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result, indent=2))
