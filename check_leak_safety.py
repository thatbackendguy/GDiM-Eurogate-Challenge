from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.reefer_pipeline import (
    TRAIN_END,
    TRAIN_START,
    get_feature_columns,
    get_model_rows,
    make_validation_folds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run basic leak-safety and split-integrity checks for the reefer pipeline."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/hourly_feature_table.csv"),
        help="Feature table CSV to inspect.",
    )
    return parser.parse_args()


def assert_series_close(name: str, left: pd.Series, right: pd.Series, atol: float = 1e-9) -> None:
    left_values = left.to_numpy(dtype=float)
    right_values = right.to_numpy(dtype=float)
    same = np.isclose(left_values, right_values, atol=atol, equal_nan=True)
    if not bool(np.all(same)):
        mismatch_count = int((~same).sum())
        raise AssertionError(f"{name} failed: {mismatch_count} mismatched rows.")


def main() -> None:
    args = parse_args()
    feature_table = pd.read_csv(args.features.resolve(), parse_dates=["timestamp_utc"]).set_index(
        "timestamp_utc"
    ).sort_index()

    public_rows = feature_table.loc[feature_table["is_public_target"]].copy()
    if public_rows.empty:
        raise AssertionError("No public target rows found in the feature table.")
    if public_rows.index.min() <= TRAIN_END:
        raise AssertionError("Public target rows overlap with the training window.")

    assert_series_close(
        "feat_lag_24",
        feature_table["feat_lag_24"],
        feature_table["y_kw"].shift(24),
    )
    assert_series_close(
        "feat_lag_168",
        feature_table["feat_lag_168"],
        feature_table["y_kw"].shift(168),
    )
    assert_series_close(
        "baseline_pred_kw",
        feature_table["baseline_pred_kw"],
        0.7 * feature_table["feat_lag_24"] + 0.3 * feature_table["feat_lag_168"],
    )

    if "obs_active_container_count" in feature_table.columns:
        assert_series_close(
            "feat_obs_active_container_count_lag24",
            feature_table["feat_obs_active_container_count_lag24"],
            feature_table["obs_active_container_count"].shift(24),
        )
        assert_series_close(
            "feat_obs_active_container_count_lag168",
            feature_table["feat_obs_active_container_count_lag168"],
            feature_table["obs_active_container_count"].shift(168),
        )

    weather_mean_columns = [
        column
        for column in feature_table.columns
        if column.startswith("obs_weather_") and column.endswith("_mean")
    ]
    if weather_mean_columns:
        weather_col = weather_mean_columns[0]
        assert_series_close(
            f"feat_{weather_col}_lag24",
            feature_table[f"feat_{weather_col}_lag24"],
            feature_table[weather_col].shift(24),
        )

    rows = get_model_rows(feature_table)
    for fold_name, fold_start, fold_end in make_validation_folds(feature_table):
        train_frame = rows.loc[
            (rows.index < fold_start)
            & (rows.index >= TRAIN_START)
            & (rows.index <= TRAIN_END)
        ].copy()
        valid_frame = rows.loc[(rows.index >= fold_start) & (rows.index <= fold_end)].copy()
        if train_frame.empty or valid_frame.empty:
            raise AssertionError(f"{fold_name} produced an empty split.")
        if train_frame.index.max() >= valid_frame.index.min():
            raise AssertionError(f"{fold_name} train/valid overlap detected.")
        if train_frame.index.max() > TRAIN_END:
            raise AssertionError(f"{fold_name} training rows extend past TRAIN_END.")

    for model_mode in ["residual", "direct"]:
        feature_columns = get_feature_columns(feature_table, model_mode=model_mode)
        invalid = [
            column
            for column in feature_columns
            if not column.startswith("feat_") and column != "baseline_pred_kw"
        ]
        if invalid:
            raise AssertionError(f"{model_mode} uses unexpected non-lagged columns: {invalid}")

    print("Leak-safety checks passed.")
    print(f"Public target window starts at {public_rows.index.min()} and training ends at {TRAIN_END}.")
    print("Verified lag-24/168 target features, representative reefer and weather lags, and fold split integrity.")


if __name__ == "__main__":
    main()
