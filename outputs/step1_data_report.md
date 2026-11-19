# Step 1 data report: FreshRetailNet-50K

Loaded in 7s (cached at data/raw/).

## Split: train

- Rows: 4,500,000  | series (store x product): 50,000
- Stores: 898 | products: 865 | cities: 18
- Date range: 2024-03-28 -> 2024-06-25 (90 days)
- [PASS] 19 expected columns present (got 19)
- [PASS] 90 distinct dates
- [PASS] every series has exactly 90 rows (min=90, max=90)
- [PASS] no duplicate (store_id, product_id, dt) keys
- [PASS] no missing values in scalar columns
- [PASS] sale_amount is never negative

**Column summary**

| column | dtype | min | max | mean |
|---|---|---|---|---|
| city_id | int64 | 0 | 17 | 4.974 |
| store_id | int64 | 0 | 897 | 409.1 |
| management_group_id | int64 | 0 | 6 | 4.622 |
| first_category_id | int64 | 0 | 31 | 15.21 |
| second_category_id | int64 | 0 | 83 | 44.74 |
| third_category_id | int64 | 0 | 232 | 116.4 |
| product_id | int64 | 0 | 864 | 390.4 |
| dt | datetime64[ns] | 2024-03-28 00:00:00 | 2024-06-25 00:00:00 | - |
| sale_amount | float64 | 0 | 44.9 | 0.9986 |
| stock_hour6_22_cnt | int32 | 0 | 16 | 3.18 |
| discount | float64 | 0 | 1.088 | 0.9111 |
| holiday_flag | int32 | 0 | 1 | 0.3444 |
| activity_flag | int32 | 0 | 1 | 0.3784 |
| precpt | float64 | 0 | 42.5 | 3.698 |
| avg_temperature | float64 | 12.06 | 30.88 | 22.28 |
| avg_humidity | float64 | 27.35 | 100 | 74.45 |
| avg_wind_level | float64 | 0.97 | 3.85 | 1.725 |
- [PASS] hourly arrays have 24 values (got 24, 24)
- [PASS] hours_stock_status values are 0/1 (found [0, 1])
- [PASS] stockout encoding: value 1 = out of stock (matches stock_hour6_22_cnt on 100.00% of rows; opposite encoding matches 2.48%)
- [INFO] sum(hours_sale) equals sale_amount on 100.00% of rows (median relative gap 0)
- [INFO] share of sales inside business hours 6-22: 95.19%

**Stockouts**

- Rows with >=1 stockout hour: 44.27% (conversation expected ~44%)
- Rows fully out of stock (16 h): 4.04%
- Stockout-free (clean) days per series: median 52, min 0, max 85
- Series with >= 30 clean days: 91.86%; with 0 clean days: 0.18%
- Zero-sale rows: 4.46% (of which on stockout days: 76.62%)
- First stockout hour distribution (hour: share of stockout days): 6:14%, 7:1%, 8:2%, 9:3%, 10:5%, 11:7%, 12:6%, 13:5%, 14:5%, 15:6%, 16:9%, 17:11%, 18:9%, 19:6%, 20:5%, 21:5%

**Promotions**

- discount < 1 on 51.53% of rows; activity_flag=1 on 37.84%; holiday_flag=1 on 34.44%

## Split: eval

- Rows: 350,000  | series (store x product): 50,000
- Stores: 898 | products: 865 | cities: 18
- Date range: 2024-06-26 -> 2024-07-02 (7 days)
- [PASS] 19 expected columns present (got 19)
- [PASS] 7 distinct dates
- [PASS] every series has exactly 7 rows (min=7, max=7)
- [PASS] no duplicate (store_id, product_id, dt) keys
- [PASS] no missing values in scalar columns
- [PASS] sale_amount is never negative

**Column summary**

| column | dtype | min | max | mean |
|---|---|---|---|---|
| city_id | int64 | 0 | 17 | 4.974 |
| store_id | int64 | 0 | 897 | 409.1 |
| management_group_id | int64 | 0 | 6 | 4.622 |
| first_category_id | int64 | 0 | 31 | 15.21 |
| second_category_id | int64 | 0 | 83 | 44.74 |
| third_category_id | int64 | 0 | 232 | 116.4 |
| product_id | int64 | 0 | 864 | 390.4 |
| dt | datetime64[ns] | 2024-06-26 00:00:00 | 2024-07-02 00:00:00 | - |
| sale_amount | float64 | 0 | 49.9 | 1.193 |
| stock_hour6_22_cnt | int32 | 0 | 16 | 2.897 |
| discount | float64 | 0 | 1 | 0.9217 |
| holiday_flag | int32 | 0 | 1 | 0.2857 |
| activity_flag | int32 | 0 | 1 | 0.3297 |
| precpt | float64 | 0.4949 | 14.84 | 5.883 |
| avg_temperature | float64 | 26.4 | 32.29 | 29.01 |
| avg_humidity | float64 | 51.47 | 86.38 | 78.49 |
| avg_wind_level | float64 | 1.05 | 3.68 | 1.665 |
- [PASS] hourly arrays have 24 values (got 24, 24)
- [PASS] hours_stock_status values are 0/1 (found [0, 1])
- [PASS] stockout encoding: value 1 = out of stock (matches stock_hour6_22_cnt on 100.00% of rows; opposite encoding matches 2.19%)
- [INFO] sum(hours_sale) equals sale_amount on 100.00% of rows (median relative gap 0)
- [INFO] share of sales inside business hours 6-22: 95.02%

**Stockouts**

- Rows with >=1 stockout hour: 40.98% (conversation expected ~44%)
- Rows fully out of stock (16 h): 3.58%
- Stockout-free (clean) days per series: median 4, min 1, max 7
- Series with >= 30 clean days: 0.00%; with 0 clean days: 0.00%
- Zero-sale rows: 4.16% (of which on stockout days: 73.96%)
- First stockout hour distribution (hour: share of stockout days): 6:14%, 7:1%, 8:2%, 9:4%, 10:5%, 11:7%, 12:6%, 13:5%, 14:5%, 15:6%, 16:9%, 17:11%, 18:9%, 19:6%, 20:5%, 21:5%

**Promotions**

- discount < 1 on 49.49% of rows; activity_flag=1 on 32.97%; holiday_flag=1 on 28.57%

## Train/eval consistency

- [PASS] same series in train and eval (50,000 vs 50,000)
- [PASS] eval starts after train ends (2024-06-25 < 2024-06-26)

## Summary

ALL CHECKS PASSED