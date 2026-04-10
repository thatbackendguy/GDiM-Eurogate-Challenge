from __future__ import annotations

from pathlib import Path

from pipeline.reefer_pipeline import build_arg_parser, build_feature_table, parse_cli_root


def main() -> None:
    parser = build_arg_parser("Build the canonical hourly reefer feature table.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/hourly_feature_table.csv"),
        help="CSV file to write the feature table to.",
    )
    args = parser.parse_args()

    paths = parse_cli_root(args)
    output_path = args.output
    if not output_path.is_absolute():
        output_path = paths.root / output_path

    print("Loading raw challenge data and building hourly features...")
    feature_table = build_feature_table(paths)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.reset_index().to_csv(output_path, index=False)

    print(f"Wrote {len(feature_table):,} hourly rows to {output_path}")


if __name__ == "__main__":
    main()

