# Business Analytics Visualization Pack

This folder contains business-facing visual summaries of reefer electricity demand and the current forecast horizon.

## Files
- `01_load_history_and_forecast.png`: recent history, rolling mean, forecast, and p90 band
- `02_weekday_hour_heatmap.png`: average demand by weekday and hour
- `03_monthly_peak_trend.png`: average and max daily peak trend by month
- `04_temperature_vs_load.png`: load sensitivity to ambient temperature
- `05_hardware_mix_area.png`: top hardware mix over time
- `06_peak_hour_distribution.png`: when top-10% load hours happen
- `07_forecast_risk_band.png`: forecast horizon with risk buffer
- `kpi_summary.json`: business KPI summary

## Suggested Business Uses
- Capacity planning: use the history + peak trend + peak-hour chart to identify likely congestion windows.
- Energy procurement: use the forecast risk band to plan upper-bound power draw.
- Operations planning: use the weekday-hour heatmap to schedule labor or load-smoothing actions.
- Asset strategy: use the hardware mix view to see whether load shifts coincide with fleet composition changes.
