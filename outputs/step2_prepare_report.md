# Step 2 report: cleaning and preparation

## Cleaning

- [PASS] no rows lost (4,850,000 rows = train + eval)
- discount fixed on 17,373 rows (0.358%): <=0 -> 1.0, >1 -> 1.0
- [PASS] discount now within (0, 1]
- [PASS] every day is clean XOR stockout
- [PASS] every series has 97 days (90 train + 7 eval)
- [PASS] days are consecutive with no gaps in every series
- [PASS] raw stockout count renamed to stockout_hours

## Series eligibility

- Rules: >= 30 clean train days, >= 1 clean eval day, mean train sales > 0
- Eligible: 45,932 of 50,000 series (91.9%)
- Sales groups (all series): low 36,448, high 13,552

## Study sample (n=5,000, seed=42, stratified by first category x sales group)

- [PASS] sample has the requested size
- [PASS] sample is unique and all eligible

| measure | eligible pool | sample |
|---|---|---|
| share low-sales series | 74.5% | 74.5% |
| mean train sales | 0.946 | 0.944 |
| median clean train days | 53 | 53 |
| mean stockout rate (train) | 41.3% | 41.4% |
| mean promo rate (train) | 39.1% | 39.4% |
| stores / products / categories | 897 / 837 / 32 | 853 / 511 / 29 |

## Example series (series_id=18, store 0, product 127), last 10 days

| dt | split | sale_amount | stockout_hours | clean | discount | activity |
|---|---|---|---|---|---|---|
| 2024-06-23 | train | 0.80 | 0 | 1 | 0.95 | 0 |
| 2024-06-24 | train | 0.70 | 0 | 1 | 0.93 | 0 |
| 2024-06-25 | train | 0.40 | 0 | 1 | 0.94 | 0 |
| 2024-06-26 | eval | 1.30 | 0 | 1 | 0.93 | 0 |
| 2024-06-27 | eval | 0.40 | 0 | 1 | 0.94 | 0 |
| 2024-06-28 | eval | 0.80 | 0 | 1 | 0.94 | 0 |
| 2024-06-29 | eval | 0.60 | 2 | 0 | 0.94 | 0 |
| 2024-06-30 | eval | 1.10 | 6 | 0 | 0.93 | 0 |
| 2024-07-01 | eval | 0.80 | 0 | 1 | 0.94 | 0 |
| 2024-07-02 | eval | 0.90 | 0 | 1 | 0.94 | 0 |

## Summary

ALL CHECKS PASSED

Done in 66s.