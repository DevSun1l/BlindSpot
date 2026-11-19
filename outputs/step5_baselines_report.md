- [PASS] SSA on condition A equals SSA on the raw dataset (A = recorded sales)
# Step 5 report: baselines

Scored on 20,996 clean test days of the 5,000-series sample. Metrics are averaged over the 7 test dates (frn-50k-baseline protocol).

## Overall (all series)

| model | horizon | trained on | WAPE (95% CI) | WPE (bias) | MAE | RMSE |
|---|---|---|---|---|---|---|
| naive | next-day | all | 0.419 (0.405-0.427) | -0.005 | 0.473 | 0.765 |
| seasonal_naive | next-day | all | 0.412 (0.401-0.422) | -0.010 | 0.465 | 0.759 |
| moving_avg_7 | next-day | all | 0.327 (0.318-0.335) | -0.009 | 0.371 | 0.592 |
| croston_sba | next-day | all | 0.335 (0.328-0.344) | -0.096 | 0.382 | 0.638 |
| ssa | week-ahead | A | 0.331 (0.322-0.338) | -0.110 | 0.375 | 0.645 |
| ssa | week-ahead | B | 0.365 (0.357-0.371) | -0.212 | 0.414 | 0.734 |
| ssa | week-ahead | C | 0.343 (0.333-0.351) | -0.106 | 0.388 | 0.666 |

## By sales group (WAPE / WPE)

| model | trained on | low-sales | high-sales |
|---|---|---|---|
| naive | all | 0.512 / +0.029 | 0.340 / -0.034 |
| seasonal_naive | all | 0.494 / +0.044 | 0.343 / -0.057 |
| moving_avg_7 | all | 0.396 / +0.043 | 0.268 / -0.054 |
| croston_sba | all | 0.388 / -0.049 | 0.290 / -0.137 |
| ssa | A | 0.379 / -0.071 | 0.291 / -0.144 |
| ssa | B | 0.400 / -0.179 | 0.336 / -0.241 |
| ssa | C | 0.390 / -0.071 | 0.303 / -0.136 |

## Checks

- [PASS] 7 test days per series
- [PASS] no missing baseline forecasts on test days
- [PASS] SSA trained on censored history (B) is biased lower than on clean history (A)

## Summary

ALL CHECKS PASSED

Done in 11s.