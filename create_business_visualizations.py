from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
FEATURES_CSV = ROOT / "outputs" / "hourly_feature_table.csv"
PREDICTIONS_CSV = ROOT / "predictions.csv"
OUTPUT_DIR = ROOT / "outputs" / "business_analytics"


def ensure_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing feature table: {FEATURES_CSV}")
    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(f"Missing predictions file: {PREDICTIONS_CSV}")

    features = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
    predictions = pd.read_csv(PREDICTIONS_CSV, parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
    if isinstance(predictions.index.dtype, pd.DatetimeTZDtype):
        predictions.index = predictions.index.tz_localize(None)
    return features, predictions


def apply_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.figsize": (14, 7),
            "axes.titlesize": 18,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def chart_load_history_and_forecast(features: pd.DataFrame, predictions: pd.DataFrame, out_dir: Path) -> None:
    actual = features["y_kw"].dropna()
    recent = actual.loc[actual.index >= actual.index.max() - pd.Timedelta(days=90)]
    forecast_start = predictions.index.min()
    forecast_end = predictions.index.max()

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(recent.index, recent.values, color="#0f766e", linewidth=1.2, alpha=0.55, label="Observed hourly load")
    ax.plot(
        recent.index,
        recent.rolling(24, min_periods=12).mean(),
        color="#0f172a",
        linewidth=2.4,
        label="24h rolling mean",
    )
    ax.plot(
        predictions.index,
        predictions["pred_power_kw"].values,
        color="#c2410c",
        linewidth=2.4,
        label="Forecast",
    )
    ax.fill_between(
        predictions.index,
        predictions["pred_power_kw"].values,
        predictions["pred_p90_kw"].values,
        color="#fdba74",
        alpha=0.35,
        label="Forecast to p90 band",
    )
    if forecast_start in actual.index:
        actual_window = actual.loc[(actual.index >= forecast_start) & (actual.index <= forecast_end)]
        ax.plot(actual_window.index, actual_window.values, color="#7c3aed", linewidth=2.0, label="Observed in target window")

    ax.axvspan(forecast_start, forecast_end, color="#fed7aa", alpha=0.18)
    ax.set_title("Recent Reefer Load History With Forecast Window")
    ax.set_ylabel("Load (kW)")
    ax.set_xlabel("Timestamp (UTC)")
    ax.legend(loc="upper left", ncol=2)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    save_figure(fig, out_dir / "01_load_history_and_forecast.png")


def chart_hour_week_heatmap(features: pd.DataFrame, out_dir: Path) -> None:
    actual = features["y_kw"].dropna().to_frame("y_kw")
    actual["hour"] = actual.index.hour
    actual["weekday"] = actual.index.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat = actual.pivot_table(index="weekday", columns="hour", values="y_kw", aggfunc="mean").reindex(order)

    fig, ax = plt.subplots(figsize=(16, 7))
    sns.heatmap(heat, cmap="YlGnBu", ax=ax, cbar_kws={"label": "Average load (kW)"})
    ax.set_title("Average Reefer Load by Weekday and Hour")
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("")
    save_figure(fig, out_dir / "02_weekday_hour_heatmap.png")


def chart_daily_peak_profile(features: pd.DataFrame, out_dir: Path) -> None:
    actual = features["y_kw"].dropna().to_frame("y_kw")
    daily = actual.resample("D").agg(daily_mean_kw=("y_kw", "mean"), daily_peak_kw=("y_kw", "max"))
    daily = daily.dropna()
    daily["month"] = daily.index.to_period("M").astype(str)

    monthly = daily.groupby("month").agg(
        avg_daily_load_kw=("daily_mean_kw", "mean"),
        avg_daily_peak_kw=("daily_peak_kw", "mean"),
        max_daily_peak_kw=("daily_peak_kw", "max"),
    )

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(monthly.index, monthly["avg_daily_load_kw"], color="#0f766e", linewidth=2.2, label="Average daily load")
    ax.plot(monthly.index, monthly["avg_daily_peak_kw"], color="#c2410c", linewidth=2.2, label="Average daily peak")
    ax.bar(monthly.index, monthly["max_daily_peak_kw"], color="#fbbf24", alpha=0.35, label="Max daily peak")
    ax.set_title("Monthly Load and Peak Demand Trend")
    ax.set_ylabel("Load (kW)")
    ax.set_xlabel("Month")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left")
    save_figure(fig, out_dir / "03_monthly_peak_trend.png")


def chart_temperature_vs_load(features: pd.DataFrame, out_dir: Path) -> None:
    candidates = [
        "obs_weather_temperature_zentralgate_mean",
        "obs_weather_temperature_vc_halle3_mean",
        "obs_temp_ambient_mean",
    ]
    temp_col = next((col for col in candidates if col in features.columns), None)
    if temp_col is None:
        return

    sample = features[[temp_col, "y_kw"]].dropna()
    if len(sample) > 12000:
        sample = sample.sample(12000, random_state=42)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hexbin(sample[temp_col], sample["y_kw"], gridsize=35, cmap="viridis", mincnt=1)
    coeffs = np.polyfit(sample[temp_col], sample["y_kw"], deg=1)
    x = np.linspace(sample[temp_col].min(), sample[temp_col].max(), 100)
    ax.plot(x, coeffs[0] * x + coeffs[1], color="#ef4444", linewidth=2.5, label="Linear trend")
    ax.set_title("Temperature vs Reefer Load")
    ax.set_xlabel("Ambient temperature")
    ax.set_ylabel("Load (kW)")
    ax.legend(loc="upper left")
    save_figure(fig, out_dir / "04_temperature_vs_load.png")


def chart_hardware_mix(features: pd.DataFrame, out_dir: Path) -> None:
    share_cols = [c for c in features.columns if c.startswith("obs_hardware_share_")]
    if not share_cols:
        return
    hardware = features[share_cols].copy()
    hardware.columns = [c.replace("obs_hardware_share_", "") for c in share_cols]
    monthly = hardware.resample("M").mean()
    top_cols = monthly.mean().sort_values(ascending=False).head(6).index.tolist()
    monthly = monthly[top_cols]

    fig, ax = plt.subplots(figsize=(16, 7))
    monthly.plot.area(ax=ax, alpha=0.85, linewidth=0)
    ax.set_title("Monthly Hardware Mix of Connected Reefers")
    ax.set_ylabel("Average share of active containers")
    ax.set_xlabel("Month")
    ax.legend(title="Hardware", loc="upper left", ncol=3)
    save_figure(fig, out_dir / "05_hardware_mix_area.png")


def chart_peak_hour_distribution(features: pd.DataFrame, out_dir: Path) -> None:
    actual = features["y_kw"].dropna()
    peak_threshold = actual.quantile(0.9)
    peak_hours = actual[actual >= peak_threshold]
    counts = peak_hours.groupby(peak_hours.index.hour).size().reindex(range(24), fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(counts.index, counts.values, color="#7c3aed", alpha=0.85)
    ax.set_title("When High-Load Hours Happen Most Often")
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Count of top-10% load hours")
    ax.set_xticks(range(24))
    save_figure(fig, out_dir / "06_peak_hour_distribution.png")


def chart_forecast_risk_band(predictions: pd.DataFrame, features: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(predictions.index, predictions["pred_power_kw"], color="#b45309", linewidth=2.5, label="Forecast")
    ax.fill_between(
        predictions.index,
        predictions["pred_power_kw"],
        predictions["pred_p90_kw"],
        color="#f59e0b",
        alpha=0.35,
        label="Risk buffer to p90",
    )
    actual = features["y_kw"].dropna()
    actual_window = actual.loc[(actual.index >= predictions.index.min()) & (actual.index <= predictions.index.max())]
    if not actual_window.empty:
        ax.plot(actual_window.index, actual_window.values, color="#1d4ed8", linewidth=2.0, label="Observed load")
    ax.set_title("Forecast Horizon With Risk Buffer")
    ax.set_ylabel("Load (kW)")
    ax.set_xlabel("Forecast timestamp (UTC)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    save_figure(fig, out_dir / "07_forecast_risk_band.png")


def write_kpi_summary(features: pd.DataFrame, predictions: pd.DataFrame, out_dir: Path) -> Path:
    actual = features["y_kw"].dropna()
    peak_threshold = float(actual.quantile(0.9))
    summary = {
        "history_start": str(actual.index.min()),
        "history_end": str(actual.index.max()),
        "average_load_kw": round(float(actual.mean()), 3),
        "p90_load_threshold_kw": round(peak_threshold, 3),
        "max_load_kw": round(float(actual.max()), 3),
        "forecast_start": str(predictions.index.min()),
        "forecast_end": str(predictions.index.max()),
        "forecast_avg_kw": round(float(predictions["pred_power_kw"].mean()), 3),
        "forecast_max_kw": round(float(predictions["pred_power_kw"].max()), 3),
        "forecast_p90_avg_kw": round(float(predictions["pred_p90_kw"].mean()), 3),
        "forecast_risk_buffer_avg_kw": round(float((predictions["pred_p90_kw"] - predictions["pred_power_kw"]).mean()), 3),
    }
    out_path = out_dir / "kpi_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path


def write_report(out_dir: Path, kpi_path: Path) -> None:
    report = f"""# Business Analytics Visualization Pack

This folder contains business-facing visual summaries of reefer electricity demand and the current forecast horizon.

## Files
- `01_load_history_and_forecast.png`: recent history, rolling mean, forecast, and p90 band
- `02_weekday_hour_heatmap.png`: average demand by weekday and hour
- `03_monthly_peak_trend.png`: average and max daily peak trend by month
- `04_temperature_vs_load.png`: load sensitivity to ambient temperature
- `05_hardware_mix_area.png`: top hardware mix over time
- `06_peak_hour_distribution.png`: when top-10% load hours happen
- `07_forecast_risk_band.png`: forecast horizon with risk buffer
- `{kpi_path.name}`: business KPI summary

## Suggested Business Uses
- Capacity planning: use the history + peak trend + peak-hour chart to identify likely congestion windows.
- Energy procurement: use the forecast risk band to plan upper-bound power draw.
- Operations planning: use the weekday-hour heatmap to schedule labor or load-smoothing actions.
- Asset strategy: use the hardware mix view to see whether load shifts coincide with fleet composition changes.
"""
    (out_dir / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    features, predictions = ensure_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    chart_load_history_and_forecast(features, predictions, OUTPUT_DIR)
    chart_hour_week_heatmap(features, OUTPUT_DIR)
    chart_daily_peak_profile(features, OUTPUT_DIR)
    chart_temperature_vs_load(features, OUTPUT_DIR)
    chart_hardware_mix(features, OUTPUT_DIR)
    chart_peak_hour_distribution(features, OUTPUT_DIR)
    chart_forecast_risk_band(predictions, features, OUTPUT_DIR)
    kpi_path = write_kpi_summary(features, predictions, OUTPUT_DIR)
    write_report(OUTPUT_DIR, kpi_path)

    print(f"Wrote analytics pack to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
