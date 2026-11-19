# Step 6a report: LightGBM tuning (condition A only)

20 hyperparameter sets x 2 boosting types = 40 fits; 187,032 fitting rows, 41,144 validation rows. Took 932s.

## gbdt vs dart

| boosting | best valid WAPE | median valid WAPE | mean seconds per fit |
|---|---|---|---|
| gbdt | 0.2957 | 0.2983 | 4 |
| dart | 0.3003 | 0.3064 | 42 |

Paired comparison (same hyperparameters): dart better in 0 of 20 trials.

**Chosen: gbdt** (valid WAPE 0.2957, 259 rounds). Tie rule: dart only if better than gbdt's best by > 0.002.

## Frozen configuration (used unchanged for A, B and C)

```json
{
  "params":{
    "objective":"regression",
    "learning_rate":0.03,
    "num_leaves":63,
    "min_child_samples":50,
    "feature_fraction":0.6,
    "bagging_fraction":0.7,
    "lambda_l2":1.0,
    "boosting":"gbdt"
  },
  "n_rounds":259,
  "valid_wape":0.2957,
  "tuned_on":"condition A, validation = last 14 train days",
  "seed":42
}
```

## Top 10 trials

```
 trial boosting  objective  learning_rate  num_leaves  min_child_samples  feature_fraction  bagging_fraction  lambda_l2  n_estimators  rounds  valid_wape  seconds
     9     gbdt regression           0.03          63                 50               0.6              0.70        1.0          1000     259      0.2957      5.5
     8     gbdt regression           0.03         127                 50               0.6              0.85       10.0           300     251      0.2962      8.5
    14     gbdt    tweedie           0.05          63                100               0.6              0.70        0.0           300     112      0.2964      4.0
     5     gbdt regression           0.10          63                 50               0.6              1.00        1.0          1000      69      0.2966      1.9
     4     gbdt regression           0.05         127                 20               1.0              1.00        0.0           600      97      0.2970      6.7
    16     gbdt    tweedie           0.05          63                100               0.6              0.85       10.0           300     107      0.2972      4.0
     6     gbdt    tweedie           0.10         127                 20               0.8              0.85        1.0           300      51      0.2975      5.6
    15     gbdt regression           0.10          63                 50               1.0              0.85        0.0          1000      65      0.2975      2.3
    19     gbdt    tweedie           0.03          31                100               0.6              0.85        1.0          1000     234      0.2977      4.2
     2     gbdt    tweedie           0.03         127                 50               0.8              0.85        0.0          1000     154      0.2982      8.5
```