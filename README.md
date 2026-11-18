# BlindSpot: A Study and Prototype for Auditing Demand-Forecast Explanations under Stockout Censoring

> Stockouts hide demand. Do they hide the truth from explanations too?

BlindSpot is an empirical study, with an interactive prototype, of how stockout censoring affects the
reliability of SHAP and what-if explanations for perishable-demand forecasts, and the inventory cost
of acting on them.

**Status:** Phase 1 in progress (Step 0: environment setup).

## Phase 1 design
- Dataset: [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) (CC BY 4.0)
- Model: LightGBM, trained under three conditions: clean (A), censored (B), matched noise (C)
- Explanations: SHAP TreeExplainer + what-if on `discount` / `activity_flag`
- Decisions: reorder-point / newsvendor inventory simulation
- Evaluation protocol and SSA baseline follow [frn-50k-baseline](https://github.com/Dingdong-Inc/frn-50k-baseline) (Apache 2.0)

## Setup (Windows, PowerShell)
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\check_env.py
```

## Layout
| Folder | Contents |
|---|---|
| `src/blindspot/` | Pipeline code (data, conditions, features, models, SHAP, inventory) |
| `scripts/` | Runnable entry points |
| `tests/` | pytest tests |
| `app/` | Streamlit prototype |
| `data/`, `outputs/` | Local data and results (not committed) |
| `docs/` | Write-up and notes |

## Citation
Wang et al. (2025). *FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand
Recovery and Forecasting in Fresh Retail.* arXiv:2505.16319.
