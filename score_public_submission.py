from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score one or more submission CSVs against the released public window."
    )
    parser.add_argument(
        "submissions",
        nargs="+",
        type=Path,
        help="Submission CSV files to score.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/hourly_feature_table.csv"),
        help="Feature table containing the known public actuals.",
    )
    return parser.parse_args()


def score_submission(submission: pd.DataFrame, actual: pd.DataFrame) -> dict[str, float | int]:
    merged = submission.merge(actual, on="timestamp_utc", how="inner")
    actual_kw = merged["y_kw"].to_numpy(dtype=float)
    pred_power = merged["pred_power_kw"].to_numpy(dtype=float)
    pred_p90 = merged["pred_p90_kw"].to_numpy(dtype=float)

    mae_all = float(np.mean(np.abs(actual_kw - pred_power)))
    peak_threshold = float(np.quantile(actual_kw, 0.9))
    peak_mask = actual_kw >= peak_threshold
    mae_peak = float(np.mean(np.abs(actual_kw[peak_mask] - pred_power[peak_mask])))

    residual = actual_kw - pred_p90
    pinball = float(
        np.mean(np.where(residual >= 0.0, 0.9 * residual, 0.1 * (-residual)))
    )
    score = 0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball
    return {
        "rows": int(len(merged)),
        "mae_all": mae_all,
        "mae_peak": mae_peak,
        "pinball_p90": pinball,
        "score": float(score),
    }


def main() -> None:
    args = parse_args()
    feature_path = args.features.resolve()
    feature_table = pd.read_csv(feature_path, parse_dates=["timestamp_utc"])
    actual = feature_table.loc[feature_table["y_kw"].notna(), ["timestamp_utc", "y_kw"]].copy()
    actual["timestamp_utc"] = pd.to_datetime(actual["timestamp_utc"])

    rows: list[dict[str, float | int | str]] = []
    for path in args.submissions:
        submission = pd.read_csv(path.resolve())
        submission["timestamp_utc"] = pd.to_datetime(
            submission["timestamp_utc"].str.replace("Z", "+00:00")
        ).dt.tz_localize(None)
        metrics = score_submission(submission, actual)
        rows.append({"submission": str(path), **metrics})

    print(pd.DataFrame(rows).sort_values("score").to_string(index=False))


if __name__ == "__main__":
    main()
