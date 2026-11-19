# Step 4 report: features

## Feature list

| group | features |
|---|---|
| recent_sales | lag_1, lag_2, lag_7, lag_14, roll_mean_7, roll_mean_14, roll_std_7 |
| calendar | weekday, is_weekend, holiday_flag |
| promotion | discount, activity_flag |
| weather | precpt, avg_temperature, avg_humidity, avg_wind_level |
| identity | store_id, product_id, city_id, first_category_id, second_category_id, third_category_id |

22 features. Next-day forecast; sales history uses days up to t-1 only.

## Rows

- Train targets (clean train days, day >= 14): 228,176 (of which validation, last 14 train days: 41,144)
- Test rows: 35,000; scored (clean test days): 20,996
- Masked days among train targets: 30.4%

## Leakage and consistency checks

- [PASS] no stockout or masking column is a feature
- [PASS] A: lag_1 on train rows = previous day's condition-A sales
- [PASS] A: roll_mean_7 = mean of the 7 previous days (excludes day t)
- [PASS] B: lag_1 on train rows = previous day's condition-B sales
- [PASS] B: roll_mean_7 = mean of the 7 previous days (excludes day t)
- [PASS] C: lag_1 on train rows = previous day's condition-C sales
- [PASS] C: roll_mean_7 = mean of the 7 previous days (excludes day t)
- [PASS] test rows have identical features in A, B and C
- [PASS] test targets are the recorded (true) sales in all conditions
- [PASS] no missing history features on train targets
- [PASS] B's lags never exceed A's (censoring flows into history)
- [INFO] B's lag_1 is lower than A's on 14.7% of train rows
- [PASS] validation days come strictly after the fitting days

## How censoring changes the training features (train targets)

| feature | A mean | B mean | C mean | B vs A |
|---|---|---|---|---|
| target | 0.966 | 0.800 | 0.967 | -17.2% |
| lag_1 | 0.947 | 0.835 | 0.947 | -11.9% |
| lag_7 | 0.952 | 0.849 | 0.952 | -10.8% |
| roll_mean_7 | 0.954 | 0.849 | 0.954 | -11.1% |
| roll_mean_14 | 0.944 | 0.842 | 0.944 | -10.8% |
| roll_std_7 | 0.404 | 0.427 | 0.489 | +5.6% |

Correlation of each sales feature with the target (train targets):

| feature | A | B | C |
|---|---|---|---|
| lag_1 | 0.805 | 0.694 | 0.685 |
| lag_2 | 0.754 | 0.667 | 0.646 |
| lag_7 | 0.796 | 0.703 | 0.679 |
| lag_14 | 0.756 | 0.677 | 0.649 |
| roll_mean_7 | 0.869 | 0.799 | 0.782 |
| roll_mean_14 | 0.865 | 0.803 | 0.783 |
| roll_std_7 | 0.653 | 0.651 | 0.621 |

## Summary

ALL CHECKS PASSED

Done in 28s.