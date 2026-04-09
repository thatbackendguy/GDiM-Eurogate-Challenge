from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.reefer_pipeline import (
    build_model_ready_feature_table,
    build_arg_parser,
    build_feature_table,
    generate_submission,
    parse_cli_root,
    run_backtest,
    save_model_artifact,
    write_preprocessing_summary,
    write_metrics_json,
)


def main() -> None:
    parser = build_arg_parser("Train the reefer model and export predictions.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/hourly_feature_table.csv"),
        help="Existing feature-table CSV to reuse. If missing, features are rebuilt.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("predictions.csv"),
        help="Submission CSV output path.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("outputs/backtest_metrics.json"),
        help="Backtest metrics JSON output path.",
    )
    parser.add_argument(
        "--processed-features",
        type=Path,
        default=Path("outputs/model_ready_feature_table.csv"),
        help="Model-ready preprocessed feature table output path.",
    )
    parser.add_argument(
        "--preprocessing-summary",
        type=Path,
        default=Path("outputs/preprocessing_summary.json"),
        help="JSON summary of preprocessing decisions.",
    )
    parser.add_argument(
        "--best-model",
        type=Path,
        default=Path("outputs/best_model.pkl"),
        help="Pickle path for the lowest-score fold model artifact.",
    )
    args = parser.parse_args()

    paths = parse_cli_root(args)
    feature_path = args.features if args.features.is_absolute() else paths.root / args.features
    prediction_path = (
        args.predictions if args.predictions.is_absolute() else paths.root / args.predictions
    )
    metrics_path = args.metrics if args.metrics.is_absolute() else paths.root / args.metrics
    processed_feature_path = (
        args.processed_features
        if args.processed_features.is_absolute()
        else paths.root / args.processed_features
    )
    preprocessing_summary_path = (
        args.preprocessing_summary
        if args.preprocessing_summary.is_absolute()
        else paths.root / args.preprocessing_summary
    )
    best_model_path = (
        args.best_model if args.best_model.is_absolute() else paths.root / args.best_model
    )

    if feature_path.exists():
        print(f"Loading existing feature table from {feature_path}")
        feature_table = pd.read_csv(feature_path, parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
    else:
        print("Feature table not found. Rebuilding it from raw inputs...")
        feature_table = build_feature_table(paths)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        feature_table.reset_index().to_csv(feature_path, index=False)

    print("Running rolling backtests and calibrating the p90 forecast...")
    fold_results, best_bundle, best_calibrator, best_result = run_backtest(feature_table)
    write_metrics_json(metrics_path, fold_results)
    save_model_artifact(best_model_path, best_bundle, best_calibrator, best_result)

    print(f"Using best fold model: {best_result.fold_name}")
    model_ready = build_model_ready_feature_table(feature_table, best_bundle.preprocessor)
    processed_feature_path.parent.mkdir(parents=True, exist_ok=True)
    model_ready.reset_index().to_csv(processed_feature_path, index=False)
    write_preprocessing_summary(preprocessing_summary_path, best_bundle.preprocessor)

    print("Generating predictions.csv...")
    submission = generate_submission(feature_table, best_bundle, best_calibrator)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(prediction_path, index=False)

    print(f"Wrote submission to {prediction_path}")
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote model-ready features to {processed_feature_path}")
    print(f"Wrote preprocessing summary to {preprocessing_summary_path}")
    print(f"Wrote best model to {best_model_path}")
    print(
        f"Best model result ({best_result.fold_name}): "
        f"mae_all={best_result.mae_all:.3f}, "
        f"mae_peak={best_result.mae_peak:.3f}, "
        f"pinball_p90={best_result.pinball_p90:.3f}, "
        f"score={best_result.score:.3f}"
    )


if __name__ == "__main__":
    main()
