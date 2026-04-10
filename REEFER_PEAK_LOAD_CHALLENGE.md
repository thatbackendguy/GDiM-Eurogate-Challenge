# Reefer Load Outlook Challenge

## What is this challenge about?

Your task is to forecast the combined electricity consumption of plugged-in reefer containers for a set of future hourly timestamps.

A reefer container is a refrigerated shipping container. While it is plugged in at the terminal, its cooling unit consumes electricity. The goal is to estimate how much reefer-related power demand the terminal is likely to see tomorrow, especially during high-load hours.

## Your task

For each released public timestamp in `target_timestamps.csv`, submit:

- `pred_power_kw`: your best point forecast
- `pred_p90_kw`: a cautious upper estimate for that hour

Your `pred_p90_kw` should usually be at least as large as `pred_power_kw`.

## What data do you get?

You receive:

1. `reefer_release.zip`
2. `wetterdaten.zip`
3. `target_timestamps.csv` with released public target hours only
4. `templates/submission_template.csv`
5. `starter/reefer_starter_notebook.ipynb`

## Domain primer

If you do not come from logistics, this is enough to get started:

- `reefer container` = refrigerated shipping container
- `container terminal` = place where containers are stored, moved, and in this case plugged in
- this challenge is about reefer electricity demand, not general terminal workload
- total reefer power changes because the connected container mix changes and because cooling demand changes over time

## Key rules

1. Use only the supplied files.
2. Treat this as a 24-hour-ahead forecasting problem.
3. Do not use information from the future relative to a target hour.
4. Submit exactly one prediction for every released public timestamp in `target_timestamps.csv`.
5. Submit at most 5 files per team.
6. Your handed-in code or notebook must be easy for organizers to rerun on the hidden full target timestamp list and the complete reefer release data so they can generate your final full submission for private scoring.

## Suggested starting point

An easy first baseline is:

- use the load from the same hour one day earlier as `pred_power_kw`
- set `pred_p90_kw = 1.10 * pred_power_kw`

That is not expected to win, but it is a valid place to start.

## Useful questions

You do not have to answer these separately, but they are good prompts:

1. Which factors appear to influence aggregate reefer power demand most?
2. Are there daily or weekly patterns?
3. Which hours are hardest to predict?
4. Can you improve peak-hour behavior, not just average error?
5. Can you make `pred_p90_kw` meaningfully more useful than a fixed uplift?

## Submission format

Your submission must contain these columns:

- `timestamp_utc`
- `pred_power_kw`
- `pred_p90_kw`

See `templates/submission_template.csv`. Your file should include only the released public target hours.

## Deliverables

Submit:

1. `predictions.csv`
2. a short `approach.md`
3. your code or notebook

Your code submission should be easy to run end to end. Organizers must be able to execute it with a hidden full set of target timestamps and the complete reefer release data to generate a full submission that includes both public and private timestamps for the final score.

## Beyond the hackathon

One possible follow-up topic after the event is whether smarter storage or stacking rules could reduce reefer-related energy peaks. That is intentionally out of scope for this challenge, but it could become a separate research or thesis topic later.
