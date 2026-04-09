from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


RANDOM_SEED = 42
TARGET_FILE = "target_timestamps.csv"
REEFER_FILE = "reefer_release.csv"
WEATHER_DIR_NAME = "Wetterdaten Okt 25 - 23 Feb 26"

REEFER_USECOLS = [
    "container_visit_uuid",
    "HardwareType",
    "EventTime",
    "AvPowerCons",
    "TemperatureSetPoint",
    "TemperatureAmbient",
    "TemperatureReturn",
    "RemperatureSupply",
    "ContainerSize",
    "stack_tier",
]

TEMPERATURE_COLUMNS = {
    "TemperatureSetPoint": "setpoint",
    "TemperatureAmbient": "ambient",
    "TemperatureReturn": "return",
    "RemperatureSupply": "supply",
}

CATEGORY_PREFIXES = {
    "HardwareType": "hardware",
    "ContainerSize": "size",
    "stack_tier": "tier",
}

WEATHER_FILE_LABELS = {
    "CTH_Temperatur_VC_Halle3 Okt 25 - 23 Feb 26.csv": "temperature_vc_halle3",
    "CTH_Temperatur_Zentralgate  Okt 25 - 23 Feb 26.csv": "temperature_zentralgate",
    "CTH_Wind_VC_Halle3  Okt 25 - 23 Feb 26.csv": "wind_vc_halle3",
    "CTH_Wind_Zentralgate  Okt 25 - 23 Feb 26.csv": "wind_zentralgate",
    "CTH_Windrichtung_VC_Halle3  Okt 25 - 23 Feb 26.csv": "wind_direction_vc_halle3",
    "CTH_Windrichtung_Zentralgate  Okt 25 - 23 Feb 26.csv": "wind_direction_zentralgate",
}


@dataclass
class ProjectPaths:
    root: Path
    reefer_csv: Path
    weather_dir: Path
    target_csv: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        root = root.resolve()
        return cls(
            root=root,
            reefer_csv=root / REEFER_FILE,
            weather_dir=root / WEATHER_DIR_NAME,
            target_csv=root / TARGET_FILE,
        )

    def validate(self) -> None:
        missing = [
            path
            for path in [self.reefer_csv, self.weather_dir, self.target_csv]
            if not path.exists()
        ]
        if missing:
            missing_text = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing required project inputs: {missing_text}")


@dataclass
class FoldResult:
    fold_name: str
    train_rows: int
    valid_rows: int
    mae_all: float
    mae_peak: float
    pinball_p90: float
    score: float


@dataclass
class SavedModelArtifact:
    bundle: "ResidualModelBundle"
    calibrator: "ResidualCalibrator"
    best_result: FoldResult


@dataclass
class ResidualModelBundle:
    model: GradientBoostingRegressor
    feature_columns: list[str]
    preprocessor: "FeaturePreprocessor"


@dataclass
class FeaturePreprocessor:
    strategy: str
    feature_columns: list[str]
    impute_values: pd.Series
    clip_lower: pd.Series
    clip_upper: pd.Series
    scale_mean: pd.Series
    scale_std: pd.Series
    scaled_columns: list[str]


@dataclass
class ResidualCalibrator:
    bin_edges: np.ndarray
    global_q90: float
    hour_q90: dict[int, float]
    hour_bin_q90: dict[tuple[int, int], float]

    @staticmethod
    def _assign_bins(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
        if len(bin_edges) <= 2:
            return np.zeros(len(values), dtype=int)
        raw = np.digitize(values, bin_edges[1:-1], right=True)
        return raw.astype(int)

    @classmethod
    def fit(
        cls,
        timestamps: pd.Series,
        pred_power_kw: pd.Series,
        actual_kw: pd.Series,
        *,
        min_samples_per_bin: int = 25,
    ) -> "ResidualCalibrator":
        calibration = pd.DataFrame(
            {
                "timestamp_utc": pd.to_datetime(timestamps),
                "pred_power_kw": pred_power_kw.astype(float),
                "actual_kw": actual_kw.astype(float),
            }
        ).dropna()
        if calibration.empty:
            return cls(
                bin_edges=np.array([0.0, 1.0], dtype=float),
                global_q90=0.0,
                hour_q90={},
                hour_bin_q90={},
            )

        calibration["positive_residual"] = np.maximum(
            calibration["actual_kw"] - calibration["pred_power_kw"], 0.0
        )
        calibration["hour"] = calibration["timestamp_utc"].dt.hour

        quantiles = np.linspace(0.0, 1.0, 11)
        bin_edges = np.unique(np.quantile(calibration["pred_power_kw"], quantiles))
        if len(bin_edges) < 2:
            bin_edges = np.array(
                [
                    float(calibration["pred_power_kw"].min()),
                    float(calibration["pred_power_kw"].max()) + 1.0,
                ]
            )

        calibration["pred_bin"] = cls._assign_bins(
            calibration["pred_power_kw"].to_numpy(),
            bin_edges,
        )
        global_q90 = float(calibration["positive_residual"].quantile(0.9))

        hour_q90 = (
            calibration.groupby("hour")["positive_residual"]
            .quantile(0.9)
            .astype(float)
            .to_dict()
        )

        grouped = calibration.groupby(["hour", "pred_bin"])["positive_residual"]
        hour_bin_q90: dict[tuple[int, int], float] = {}
        for key, values in grouped:
            if len(values) >= min_samples_per_bin:
                hour_bin_q90[key] = float(values.quantile(0.9))

        return cls(
            bin_edges=bin_edges.astype(float),
            global_q90=global_q90,
            hour_q90=hour_q90,
            hour_bin_q90=hour_bin_q90,
        )

    def predict_adjustment(
        self,
        timestamps: pd.Series,
        pred_power_kw: pd.Series,
    ) -> np.ndarray:
        timestamps = pd.to_datetime(timestamps)
        pred_values = pred_power_kw.astype(float).to_numpy()
        pred_bins = self._assign_bins(pred_values, self.bin_edges)
        hours = timestamps.dt.hour.to_numpy()

        adjustments: list[float] = []
        for hour, pred_bin in zip(hours, pred_bins):
            adjustments.append(
                self.hour_bin_q90.get(
                    (int(hour), int(pred_bin)),
                    self.hour_q90.get(int(hour), self.global_q90),
                )
            )
        return np.asarray(adjustments, dtype=float)


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing the challenge files.",
    )
    return parser


def sanitize_name(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.strip().lower())
    return slug.strip("_") or "missing"


def load_target_hours(target_csv: Path) -> pd.Series:
    targets = pd.read_csv(target_csv, parse_dates=["timestamp_utc"])
    if pd.api.types.is_datetime64tz_dtype(targets["timestamp_utc"]):
        targets["timestamp_utc"] = targets["timestamp_utc"].dt.tz_localize(None)
    if targets["timestamp_utc"].duplicated().any():
        raise ValueError("target_timestamps.csv contains duplicates.")
    if not targets["timestamp_utc"].is_monotonic_increasing:
        raise ValueError("target_timestamps.csv must be sorted ascending.")
    return targets["timestamp_utc"]


def aggregate_reefer_hourly(reefer_csv: Path) -> pd.DataFrame:
    dtypes = {
        "container_visit_uuid": "string",
        "HardwareType": "string",
        "ContainerSize": "string",
        "stack_tier": "string",
    }
    reefer = pd.read_csv(
        reefer_csv,
        sep=";",
        decimal=",",
        parse_dates=["EventTime"],
        usecols=REEFER_USECOLS,
        dtype=dtypes,
        na_values=["NULL", ""],
        keep_default_na=True,
        low_memory=False,
    )

    reefer = reefer.rename(columns={"EventTime": "timestamp_utc"})
    reefer["timestamp_utc"] = reefer["timestamp_utc"].dt.floor("H")
    reefer["stack_tier"] = reefer["stack_tier"].str.strip()
    reefer["stack_tier"] = reefer["stack_tier"].replace("", pd.NA)
    reefer["y_kw"] = reefer["AvPowerCons"] / 1000.0

    hourly = reefer.groupby("timestamp_utc").agg(
        y_kw=("y_kw", "sum"),
        obs_active_container_count=("container_visit_uuid", "size"),
        obs_temp_setpoint_mean=("TemperatureSetPoint", "mean"),
        obs_temp_setpoint_median=("TemperatureSetPoint", "median"),
        obs_temp_ambient_mean=("TemperatureAmbient", "mean"),
        obs_temp_ambient_median=("TemperatureAmbient", "median"),
        obs_temp_return_mean=("TemperatureReturn", "mean"),
        obs_temp_return_median=("TemperatureReturn", "median"),
        obs_temp_supply_mean=("RemperatureSupply", "mean"),
        obs_temp_supply_median=("RemperatureSupply", "median"),
    )

    for column_name, prefix in CATEGORY_PREFIXES.items():
        counts = pd.crosstab(
            reefer["timestamp_utc"],
            reefer[column_name].fillna("missing"),
            dropna=False,
        )
        counts = counts.rename(
            columns=lambda value: f"obs_{prefix}_count_{sanitize_name(str(value))}"
        )
        shares = counts.div(hourly["obs_active_container_count"], axis=0)
        shares = shares.rename(
            columns=lambda value: value.replace("_count_", "_share_")
        )
        hourly = hourly.join(counts, how="left").join(shares, how="left")

    return hourly.sort_index()


def weather_label(path: Path) -> str:
    return WEATHER_FILE_LABELS.get(path.name, sanitize_name(path.stem))


def aggregate_single_weather_file(weather_csv: Path) -> pd.DataFrame:
    label = weather_label(weather_csv)
    weather = pd.read_csv(
        weather_csv,
        sep=";",
        decimal=",",
        parse_dates=["UtcTimestamp"],
        usecols=["UtcTimestamp", "Value"],
        na_values=["NULL", ""],
        keep_default_na=True,
        low_memory=False,
    )

    weather = weather.dropna(subset=["UtcTimestamp"])
    weather["timestamp_utc"] = weather["UtcTimestamp"].dt.floor("H")

    if "wind_direction" in label:
        radians = np.deg2rad(weather["Value"].astype(float))
        weather["sin_value"] = np.sin(radians)
        weather["cos_value"] = np.cos(radians)
        hourly = weather.groupby("timestamp_utc").agg(
            **{
                f"obs_weather_{label}_sin_mean": ("sin_value", "mean"),
                f"obs_weather_{label}_cos_mean": ("cos_value", "mean"),
            }
        )
    else:
        hourly = weather.groupby("timestamp_utc").agg(
            **{
                f"obs_weather_{label}_mean": ("Value", "mean"),
                f"obs_weather_{label}_max": ("Value", "max"),
                f"obs_weather_{label}_min": ("Value", "min"),
            }
        )

    return hourly.sort_index()


def aggregate_weather_hourly(weather_dir: Path) -> pd.DataFrame:
    frames = [
        aggregate_single_weather_file(path)
        for path in sorted(weather_dir.glob("*.csv"))
    ]
    if not frames:
        raise FileNotFoundError(f"No weather CSV files found in {weather_dir}")
    weather = pd.concat(frames, axis=1).sort_index()
    return weather


def build_hourly_observation_table(paths: ProjectPaths) -> pd.DataFrame:
    targets = load_target_hours(paths.target_csv)
    reefer_hourly = aggregate_reefer_hourly(paths.reefer_csv)
    weather_hourly = aggregate_weather_hourly(paths.weather_dir)

    min_hour = min(reefer_hourly.index.min(), weather_hourly.index.min(), targets.min())
    max_hour = max(reefer_hourly.index.max(), weather_hourly.index.max(), targets.max())
    full_index = pd.date_range(min_hour, max_hour, freq="H")

    table = pd.DataFrame(index=full_index)
    table.index.name = "timestamp_utc"
    table = table.join(reefer_hourly, how="left")
    table = table.join(weather_hourly, how="left")
    table["is_public_target"] = table.index.isin(set(targets))
    return table


def add_model_features(observations: pd.DataFrame) -> pd.DataFrame:
    df = observations.copy().sort_index()
    index = df.index
    feature_data: dict[str, pd.Series | np.ndarray] = {}

    feature_data["feat_hour"] = index.hour
    feature_data["feat_day_of_week"] = index.dayofweek
    feature_data["feat_month"] = index.month
    feature_data["feat_day_of_year"] = index.dayofyear
    feature_data["feat_is_weekend"] = (index.dayofweek >= 5).astype(int)
    feature_data["feat_hour_sin"] = np.sin(2 * np.pi * index.hour / 24.0)
    feature_data["feat_hour_cos"] = np.cos(2 * np.pi * index.hour / 24.0)
    feature_data["feat_dow_sin"] = np.sin(2 * np.pi * index.dayofweek / 7.0)
    feature_data["feat_dow_cos"] = np.cos(2 * np.pi * index.dayofweek / 7.0)

    target = df["y_kw"]
    lag_24 = target.shift(24)
    lag_48 = target.shift(48)
    lag_72 = target.shift(72)
    lag_168 = target.shift(168)
    target_shifted = target.shift(24)

    feature_data["feat_lag_24"] = lag_24
    feature_data["feat_lag_48"] = lag_48
    feature_data["feat_lag_72"] = lag_72
    feature_data["feat_lag_168"] = lag_168
    feature_data["feat_roll24_mean"] = target_shifted.rolling(24, min_periods=12).mean()
    feature_data["feat_roll24_max"] = target_shifted.rolling(24, min_periods=12).max()
    feature_data["feat_roll24_min"] = target_shifted.rolling(24, min_periods=12).min()
    feature_data["feat_roll24_std"] = target_shifted.rolling(24, min_periods=12).std()
    feature_data["feat_roll72_mean"] = target_shifted.rolling(72, min_periods=24).mean()
    feature_data["feat_roll72_max"] = target_shifted.rolling(72, min_periods=24).max()
    feature_data["feat_roll72_min"] = target_shifted.rolling(72, min_periods=24).min()
    feature_data["feat_roll72_std"] = target_shifted.rolling(72, min_periods=24).std()
    feature_data["feat_roll168_mean"] = target_shifted.rolling(168, min_periods=48).mean()
    feature_data["feat_roll168_max"] = target_shifted.rolling(168, min_periods=48).max()
    feature_data["feat_roll168_min"] = target_shifted.rolling(168, min_periods=48).min()
    feature_data["feat_roll168_std"] = target_shifted.rolling(168, min_periods=48).std()
    feature_data["feat_delta_24_48"] = lag_24 - lag_48
    feature_data["feat_delta_24_168"] = lag_24 - lag_168
    feature_data["feat_trend_24_72"] = lag_24 - feature_data["feat_roll72_mean"]
    feature_data["feat_trend_24_168"] = lag_24 - feature_data["feat_roll168_mean"]
    feature_data["baseline_pred_kw"] = 0.7 * lag_24 + 0.3 * lag_168

    observation_columns = [
        column
        for column in df.columns
        if column.startswith("obs_") and column != "y_kw"
    ]
    reefer_columns = [
        column for column in observation_columns if not column.startswith("obs_weather_")
    ]
    weather_columns = [
        column for column in observation_columns if column.startswith("obs_weather_")
    ]

    for column in reefer_columns:
        feature_data[f"feat_{column}_lag24"] = df[column].shift(24)
        feature_data[f"feat_{column}_lag168"] = df[column].shift(168)

    for column in weather_columns:
        shifted = df[column].shift(24)
        feature_data[f"feat_{column}_lag24"] = shifted
        feature_data[f"feat_{column}_lag48"] = df[column].shift(48)
        feature_data[f"feat_{column}_lag168"] = df[column].shift(168)
        feature_data[f"feat_{column}_roll24_mean"] = shifted.rolling(24, min_periods=6).mean()
        feature_data[f"feat_{column}_roll24_max"] = shifted.rolling(24, min_periods=6).max()
        feature_data[f"feat_{column}_roll24_min"] = shifted.rolling(24, min_periods=6).min()
        feature_data[f"feat_{column}_roll168_mean"] = shifted.rolling(168, min_periods=24).mean()

    features = pd.DataFrame(feature_data, index=df.index)

    feature_table = pd.concat(
        [
            df[["y_kw", "is_public_target"]],
            df.drop(columns=["is_public_target", "y_kw"], errors="ignore"),
            features,
        ],
        axis=1,
    )
    feature_table.index.name = "timestamp_utc"
    return feature_table


def build_feature_table(paths: ProjectPaths) -> pd.DataFrame:
    observations = build_hourly_observation_table(paths)
    return add_model_features(observations)


def get_feature_columns(feature_table: pd.DataFrame) -> list[str]:
    return [
        column
        for column in feature_table.columns
        if column.startswith("feat_") or column == "baseline_pred_kw"
    ]


def get_model_rows(feature_table: pd.DataFrame) -> pd.DataFrame:
    rows = feature_table.copy()
    rows = rows[rows["y_kw"].notna()]
    rows = rows[rows["baseline_pred_kw"].notna()]
    return rows.sort_index()


def make_validation_folds(
    feature_table: pd.DataFrame,
    *,
    n_folds: int = 4,
    fold_hours: int = 24 * 7,
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    usable = get_model_rows(feature_table)
    if len(usable) < n_folds * fold_hours:
        raise ValueError("Not enough trainable rows for the requested backtest folds.")

    valid_end = usable.index.max()
    folds: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for offset in range(n_folds):
        fold_end = valid_end - pd.Timedelta(hours=fold_hours * offset)
        fold_start = fold_end - pd.Timedelta(hours=fold_hours - 1)
        folds.append((f"fold_{n_folds - offset}", fold_start, fold_end))
    return list(reversed(folds))


def select_training_columns(
    train_frame: pd.DataFrame,
    feature_columns: Iterable[str],
) -> list[str]:
    selected: list[str] = []
    for column in feature_columns:
        series = train_frame[column]
        if series.notna().sum() == 0:
            continue
        if series.nunique(dropna=True) <= 1:
            continue
        selected.append(column)
    if not selected:
        raise ValueError("No informative model features were available.")
    return selected


def fit_feature_preprocessor(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
) -> FeaturePreprocessor:
    x_train = train_frame[feature_columns].copy()
    x_train = x_train.replace([np.inf, -np.inf], np.nan)
    impute_values = x_train.median(numeric_only=True)
    x_imputed = x_train.fillna(impute_values)

    clip_lower = x_imputed.quantile(0.005)
    clip_upper = x_imputed.quantile(0.995)
    x_clipped = x_imputed.clip(lower=clip_lower, upper=clip_upper, axis=1)

    scale_mean = x_clipped.mean()
    scale_std = x_clipped.std(ddof=0).replace(0.0, 1.0)

    scaled_columns: list[str] = []
    for column in feature_columns:
        unique_values = pd.Series(x_clipped[column].dropna().unique())
        if unique_values.empty:
            continue
        if set(unique_values.tolist()).issubset({0, 1}):
            continue
        if float(scale_std[column]) <= 0.0:
            continue
        scaled_columns.append(column)

    return FeaturePreprocessor(
        strategy="median_clip_scale",
        feature_columns=feature_columns,
        impute_values=impute_values,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
        scale_mean=scale_mean,
        scale_std=scale_std,
        scaled_columns=scaled_columns,
    )


def transform_features(
    frame: pd.DataFrame,
    preprocessor: FeaturePreprocessor,
) -> pd.DataFrame:
    x_frame = frame[preprocessor.feature_columns].copy()
    x_frame = x_frame.replace([np.inf, -np.inf], np.nan)
    x_frame = x_frame.fillna(preprocessor.impute_values)
    x_frame = x_frame.clip(
        lower=preprocessor.clip_lower,
        upper=preprocessor.clip_upper,
        axis=1,
    )
    if preprocessor.scaled_columns:
        x_frame = x_frame.astype(
            {column: "float64" for column in preprocessor.scaled_columns},
            copy=False,
        )
        x_frame.loc[:, preprocessor.scaled_columns] = (
            x_frame[preprocessor.scaled_columns] - preprocessor.scale_mean[preprocessor.scaled_columns]
        ) / preprocessor.scale_std[preprocessor.scaled_columns]
    return x_frame


def build_model_ready_feature_table(
    feature_table: pd.DataFrame,
    preprocessor: FeaturePreprocessor,
) -> pd.DataFrame:
    processed = transform_features(feature_table, preprocessor)
    model_ready = pd.concat(
        [
            feature_table[["y_kw", "is_public_target"]],
            processed,
        ],
        axis=1,
    )
    model_ready.index.name = "timestamp_utc"
    return model_ready


def preprocessing_summary(preprocessor: FeaturePreprocessor) -> dict[str, int]:
    return {
        "strategy": preprocessor.strategy,
        "feature_count": len(preprocessor.feature_columns),
        "scaled_feature_count": len(preprocessor.scaled_columns),
        "imputed_feature_count": int((preprocessor.impute_values.notna()).sum()),
    }


def write_preprocessing_summary(path: Path, preprocessor: FeaturePreprocessor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(preprocessing_summary(preprocessor), indent=2),
        encoding="utf-8",
    )


def fit_residual_model(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
) -> ResidualModelBundle:
    selected_columns = select_training_columns(train_frame, feature_columns)
    residual = train_frame["y_kw"] - train_frame["baseline_pred_kw"]
    peak_threshold = float(train_frame["y_kw"].quantile(0.9))
    sample_weight = np.where(train_frame["y_kw"] >= peak_threshold, 2.0, 1.0)
    preprocessor = fit_feature_preprocessor(train_frame, selected_columns)
    x_train = transform_features(train_frame, preprocessor)

    model = GradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_depth=2,
        n_estimators=120,
        min_samples_leaf=30,
        subsample=0.8,
        max_features=0.6,
        random_state=RANDOM_SEED,
    )
    model.fit(x_train, residual, sample_weight=sample_weight)
    return ResidualModelBundle(
        model=model,
        feature_columns=selected_columns,
        preprocessor=preprocessor,
    )


def predict_point_forecast(
    frame: pd.DataFrame,
    bundle: ResidualModelBundle,
) -> np.ndarray:
    x_frame = transform_features(frame, bundle.preprocessor)
    residual_pred = bundle.model.predict(x_frame)
    point_pred = frame["baseline_pred_kw"].to_numpy(dtype=float) + residual_pred
    return np.clip(point_pred, 0.0, None)


def pinball_loss(actual: np.ndarray, pred: np.ndarray, quantile: float = 0.9) -> float:
    residual = actual - pred
    return float(
        np.mean(np.where(residual >= 0.0, quantile * residual, (1.0 - quantile) * (-residual)))
    )


def evaluate_predictions(actual: np.ndarray, pred_power: np.ndarray, pred_p90: np.ndarray) -> dict[str, float]:
    mae_all = float(np.mean(np.abs(actual - pred_power)))
    peak_threshold = float(np.quantile(actual, 0.9))
    peak_mask = actual >= peak_threshold
    mae_peak = float(np.mean(np.abs(actual[peak_mask] - pred_power[peak_mask])))
    pinball = pinball_loss(actual, pred_p90, quantile=0.9)
    score = 0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball
    return {
        "mae_all": mae_all,
        "mae_peak": mae_peak,
        "pinball_p90": pinball,
        "score": float(score),
    }


def run_backtest(
    feature_table: pd.DataFrame,
) -> tuple[list[FoldResult], ResidualModelBundle, ResidualCalibrator, FoldResult]:
    rows = get_model_rows(feature_table)
    feature_columns = get_feature_columns(feature_table)
    folds = make_validation_folds(feature_table)

    fold_results: list[FoldResult] = []
    best_bundle: ResidualModelBundle | None = None
    best_calibrator: ResidualCalibrator | None = None
    best_result: FoldResult | None = None

    for fold_name, fold_start, fold_end in folds:
        train_frame = rows.loc[rows.index < fold_start].copy()
        valid_frame = rows.loc[(rows.index >= fold_start) & (rows.index <= fold_end)].copy()

        print(
            f"  {fold_name}: train_rows={len(train_frame):,}, valid_rows={len(valid_frame):,}"
        )
        bundle = fit_residual_model(train_frame, feature_columns)
        valid_pred = predict_point_forecast(valid_frame, bundle)

        fold_calibration_frame = pd.DataFrame(
            {
                "timestamp_utc": valid_frame.index,
                "actual_kw": valid_frame["y_kw"].to_numpy(dtype=float),
                "pred_power_kw": valid_pred,
            }
        )
        fold_calibrator = ResidualCalibrator.fit(
            fold_calibration_frame["timestamp_utc"],
            fold_calibration_frame["pred_power_kw"],
            fold_calibration_frame["actual_kw"],
        )
        adjustments = fold_calibrator.predict_adjustment(
            fold_calibration_frame["timestamp_utc"],
            fold_calibration_frame["pred_power_kw"],
        )
        pred_p90 = np.maximum(
            fold_calibration_frame["pred_power_kw"].to_numpy(dtype=float),
            fold_calibration_frame["pred_power_kw"].to_numpy(dtype=float) + adjustments,
        )
        metrics = evaluate_predictions(
            fold_calibration_frame["actual_kw"].to_numpy(dtype=float),
            fold_calibration_frame["pred_power_kw"].to_numpy(dtype=float),
            pred_p90,
        )
        fold_result = FoldResult(
            fold_name=fold_name,
            train_rows=int(len(train_frame)),
            valid_rows=int(len(valid_frame)),
            mae_all=metrics["mae_all"],
            mae_peak=metrics["mae_peak"],
            pinball_p90=metrics["pinball_p90"],
            score=metrics["score"],
        )
        fold_results.append(fold_result)
        if best_result is None or fold_result.score < best_result.score:
            best_result = fold_result
            best_bundle = bundle
            best_calibrator = fold_calibrator

    if best_bundle is None or best_calibrator is None or best_result is None:
        raise ValueError("Backtest did not produce a best model.")

    return fold_results, best_bundle, best_calibrator, best_result


def save_model_artifact(
    path: Path,
    bundle: ResidualModelBundle,
    calibrator: ResidualCalibrator,
    best_result: FoldResult,
) -> None:
    artifact = SavedModelArtifact(
        bundle=bundle,
        calibrator=calibrator,
        best_result=best_result,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(artifact, f)


def fit_full_model(feature_table: pd.DataFrame) -> ResidualModelBundle:
    rows = get_model_rows(feature_table)
    feature_columns = get_feature_columns(feature_table)
    return fit_residual_model(rows, feature_columns)


def generate_submission(
    feature_table: pd.DataFrame,
    bundle: ResidualModelBundle,
    calibrator: ResidualCalibrator,
) -> pd.DataFrame:
    target_frame = feature_table.loc[feature_table["is_public_target"]].copy()
    if target_frame.empty:
        raise ValueError("No public target rows were found in the feature table.")

    pred_power = predict_point_forecast(target_frame, bundle)
    adjustments = calibrator.predict_adjustment(
        pd.Series(target_frame.index, index=target_frame.index),
        pd.Series(pred_power, index=target_frame.index),
    )
    pred_p90 = np.maximum(pred_power, pred_power + adjustments)

    submission = pd.DataFrame(
        {
            "timestamp_utc": target_frame.index.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pred_power_kw": np.round(pred_power, 6),
            "pred_p90_kw": np.round(pred_p90, 6),
        }
    )
    validate_submission(submission)
    return submission


def validate_submission(submission: pd.DataFrame) -> None:
    required_columns = ["timestamp_utc", "pred_power_kw", "pred_p90_kw"]
    if submission.columns.tolist() != required_columns:
        raise ValueError("Submission columns do not match the required template.")
    if submission["timestamp_utc"].duplicated().any():
        raise ValueError("Submission contains duplicate timestamps.")
    if (submission[["pred_power_kw", "pred_p90_kw"]] < 0).any().any():
        raise ValueError("Submission contains negative predictions.")
    if (submission["pred_p90_kw"] < submission["pred_power_kw"]).any():
        raise ValueError("Submission violates pred_p90_kw >= pred_power_kw.")


def serialize_fold_results(results: list[FoldResult]) -> list[dict[str, float | int | str]]:
    return [
        {
            "fold_name": result.fold_name,
            "train_rows": result.train_rows,
            "valid_rows": result.valid_rows,
            "mae_all": result.mae_all,
            "mae_peak": result.mae_peak,
            "pinball_p90": result.pinball_p90,
            "score": result.score,
        }
        for result in results
    ]


def summarize_fold_results(results: list[FoldResult]) -> dict[str, float]:
    return {
        "mae_all": float(np.mean([result.mae_all for result in results])),
        "mae_peak": float(np.mean([result.mae_peak for result in results])),
        "pinball_p90": float(np.mean([result.pinball_p90 for result in results])),
        "score": float(np.mean([result.score for result in results])),
    }


def write_metrics_json(path: Path, fold_results: list[FoldResult]) -> None:
    aggregate = {
        "folds": serialize_fold_results(fold_results),
        "mean_metrics": summarize_fold_results(fold_results),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")


def parse_cli_root(args: argparse.Namespace) -> ProjectPaths:
    paths = ProjectPaths.from_root(args.root)
    paths.validate()
    return paths
