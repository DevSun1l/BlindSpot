# Step 6 report: LightGBM A, B and C

Frozen configuration (tuned on A): **gbdt**, objective regression, 259 rounds, learning rate 0.03, 63 leaves. Same settings and seed for all three models.

## Checks

- [PASS] test inputs are identical for A, B and C
- [PASS] all test forecasts are finite and >= 0

## Test-week results (clean test days, frn-50k-baseline protocol)

| model | WAPE (95% CI) | WPE (bias) | MAE | RMSE |
|---|---|---|---|---|
| LightGBM A (clean) | 0.286 (0.280-0.293) | -0.039 | 0.324 | 0.512 |
| LightGBM B (censored) | 0.314 (0.308-0.320) | -0.152 | 0.356 | 0.607 |
| LightGBM C (noise) | 0.296 (0.289-0.303) | -0.053 | 0.335 | 0.550 |
| Moving average 7 (best baseline) | 0.327 (0.318-0.335) | -0.009 | 0.371 | 0.592 |
| SSA A | 0.331 (0.323-0.339) | -0.110 | 0.375 | 0.645 |
| SSA B | 0.365 (0.358-0.372) | -0.212 | 0.414 | 0.734 |
| SSA C | 0.343 (0.334-0.352) | -0.106 | 0.388 | 0.666 |

## By sales group (WAPE / WPE)

| model | low-sales | high-sales |
|---|---|---|
| LightGBM A (clean) | 0.350 / -0.019 | 0.233 / -0.056 |
| LightGBM B (censored) | 0.363 / -0.127 | 0.273 / -0.172 |
| LightGBM C (noise) | 0.354 / -0.026 | 0.246 / -0.076 |
| Moving average 7 (best baseline) | 0.396 / +0.043 | 0.268 / -0.054 |
| SSA A | 0.379 / -0.071 | 0.291 / -0.144 |
| SSA B | 0.400 / -0.179 | 0.336 / -0.241 |
| SSA C | 0.390 / -0.071 | 0.303 / -0.136 |

## Comparison

- A vs best baseline: WAPE 0.286 vs 0.327 (-12.4%)
- B vs A: WAPE +9.6%, bias -0.113
- C vs A: WAPE +3.2%, bias -0.014
- Mean forecast on clean test days: A 1.082, B 0.954, C 1.066; true 1.126
- B forecasts lower than A on 92.4% of clean test days
- [PASS] LightGBM A beats the best baseline (moving average 7)
- [PASS] B is biased lower than A

## Summary

ALL CHECKS PASSED

Done in 13s.