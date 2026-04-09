# How Your Submission Will Be Evaluated

## Main idea

Your live leaderboard submission is evaluated on three things over the released public target hours:

1. overall forecast accuracy
2. accuracy during high-load hours
3. quality of your upper-risk estimate `pred_p90_kw`

Lower score is better.

The organizer keeps an additional hidden private set for the final ranking. Participants do not submit those private rows directly.
Instead, organizers will rerun the handed-in code or notebook on a hidden full target list and the complete reefer release data to generate the final full submission used for private scoring.

## Metrics

The final score combines:

1. `mae_all`
   mean absolute error over all target hours
2. `mae_peak`
   mean absolute error during high-load hours
3. `pinball_p90`
   a quantile-style loss for `pred_p90_kw`

Combined score:

- `0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball_p90`

## What this means in practice

- A model that is only good on average is not enough.
- Missing high-load hours hurts more.
- `pred_p90_kw` should be a useful upper estimate, not just random padding.

## Submission checks

Your file must:

1. contain every released timestamp from `target_timestamps.csv`
2. contain no duplicates
3. use numeric, non-negative predictions
4. satisfy `pred_p90_kw >= pred_power_kw`

Your code submission must also be reproducible in an organizer rerun. It should be straightforward to run with the hidden full target list and the complete reefer release data so the organizers can generate the full submission for final private scoring.

## Simple strategy

If you are new to forecasting, start with:

- yesterday’s same hour
- maybe also last week’s same hour
- then improve it with time patterns, weather lags, or reefer-derived signals

## Tie-break intuition

If teams are close, stronger handling of difficult high-load hours is likely to matter.
