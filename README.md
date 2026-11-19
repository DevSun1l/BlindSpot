# BlindSpot: A Study and Prototype for Auditing Demand-Forecast Explanations under Stockout Censoring

> Stockouts hide demand. Do they hide the truth from explanations too?

BlindSpot is an empirical study, with an interactive prototype, of how stockout censoring affects the
reliability of SHAP and what-if explanations for perishable-demand forecasts, and the inventory cost
of acting on them.

**Status:** Phase 1 complete. Full write-up: [docs/writeup.md](docs/writeup.md).

## Headline result

The same LightGBM model was trained on clean sales (A), on sales censored by simulated stockouts (B), and on
the same amount of random noise (C). Over 4 masking seeds:

| | A clean | B censored | C noise |
|---|---|---|---|
| Forecast WAPE | 0.286 | 0.316 | 0.294 |
| Known promotion effect recovered by the what-if | 90% | 59% | 85% |
| Per-row SHAP agreement with A (noise floor 0.944) | - | 0.857 | 0.888 |
| Everyday ordering cost per unit of demand | 0.205 | 0.233 | 0.212 |

Censoring makes the model learn a promotion effect about 40% too small while its accuracy drops only about 10%.
SHAP still describes the model faithfully and still ranks the promotion feature 2nd, so the explanation looks
trustworthy while its size is wrong, and orders based on it cost more. Matched noise does far less damage.

## Phase 1 design
- Dataset: [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) (CC BY 4.0),
  a stratified sample of 5,000 store-product series
- Model: LightGBM, tuned once on condition A (gbdt vs dart), frozen for B and C
- Explanations: SHAP TreeExplainer + what-if on `discount` / `activity_flag`
- Decisions: single-day newsvendor inventory simulation with stated relative costs
- Evaluation protocol and SSA baseline follow [frn-50k-baseline](https://github.com/Dingdong-Inc/frn-50k-baseline) (Apache 2.0)

## Setup (Windows, PowerShell)
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\check_env.py
```

## Reproduce the study

The dataset (about 60 MB) downloads from Hugging Face on first use and is cached in `data/raw/`.

| Step | Command | Output (in `outputs/`) |
|---|---|---|
| 1 Data check | `scripts\step1_check_data.py` | `step1_data_report.md` |
| 2 Prepare and sample | `scripts\step2_prepare_data.py` | `step2_prepare_report.md` |
| 2.5 EDA | `scripts\step2_5_eda.py` | `eda/` |
| 3 Conditions A/B/C | `scripts\step3_build_conditions.py` | `step3_conditions_report.md` |
| 4 Features | `scripts\step4_build_features.py` | `step4_features_report.md` |
| 5 Baselines | `scripts\step5_baselines.py` | `step5_baselines_report.md` |
| 6a Tuning (slow) | `scripts\step6a_tune.py` | `models/lgb_config.json` |
| 6b Train A/B/C | `scripts\step6b_train.py` | `step6_report.md` |
| 7 SHAP | `scripts\step7_shap.py` | `step7_shap_report.md` |
| 8 Faithfulness | `scripts\step8_faithfulness.py` | `step8_faithfulness_report.md` |
| 9 What-if | `scripts\step9_whatif.py` | `step9_whatif_report.md` |
| 10 Inventory | `scripts\step10_inventory.py` | `step10_inventory_report.md` |
| 11 Runner + seeds | `scripts\step11_experiment.py` | `step11_experiment_report.md` |

Run each with `.\.venv\Scripts\python.exe scripts\<name>.py`. After Steps 1-2 (and 6a once), Step 11 re-runs
Steps 3-10 and the multi-seed check in one command (about 7 minutes):
```powershell
.\.venv\Scripts\python.exe scripts\step11_experiment.py
```
Every script ends with `ALL CHECKS PASSED` or lists the failed checks. Tests:
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Prototype app
```powershell
.\.venv\Scripts\streamlit.exe run app\streamlit_app.py
```
Pick a store-product and a test day to compare models A, B and C: sales history with stockout and censored
days, forecasts, SHAP drivers, a what-if on discount and promotion (sales history cannot be changed), the
recommended order and its cost, and the study's results table. It needs the outputs of Steps 3-11.

## Layout
| Folder | Contents |
|---|---|
| `src/blindspot/` | Pipeline code (data, conditions, features, models, SHAP, what-if, inventory, experiment) |
| `scripts/` | Runnable entry points, one per step |
| `tests/` | pytest tests |
| `app/` | Streamlit prototype |
| `data/`, `outputs/` | Local data and results (not committed) |
| `docs/` | Write-up |

## Scope
Phase 1 measures the problem; it does not fix it. Censoring-aware demand recovery (for example Tobit models)
and an interactive or API layer that warns when an explanation may be distorted are Phase 2. SHAP and what-if
describe what the models do, not what causes sales.

## Citation
Wang et al. (2025). *FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand
Recovery and Forecasting in Fresh Retail.* arXiv:2505.16319.
