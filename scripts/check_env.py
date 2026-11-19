"""Verify the Python runtime, required package imports, and a minimal LightGBM + SHAP smoke test."""
import importlib
import sys

LIBS = ["pandas", "numpy", "pyarrow", "datasets", "lightgbm", "shap", "scipy",
        "sklearn", "statsforecast", "matplotlib", "plotly", "streamlit", "pytest", "blindspot"]

print(f"Python {sys.version.split()[0]}  ({sys.executable})")
failed = []
for name in LIBS:
    try:
        mod = importlib.import_module(name)
        print(f"  OK    {name:<14} {getattr(mod, '__version__', '')}")
    except Exception as exc:  # noqa: BLE001
        failed.append(name)
        print(f"  FAIL  {name:<14} {exc}")

# Verify LightGBM + TreeExplainer recover the dominant feature on synthetic linear data.
import numpy as np
import lightgbm as lgb
import shap

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
y = 2 * X[:, 0] + rng.normal(scale=0.1, size=200)
model = lgb.LGBMRegressor(n_estimators=20, verbose=-1).fit(X, y)
sv = shap.TreeExplainer(model).shap_values(X[:5])
top = int(np.abs(sv).mean(axis=0).argmax())
print(f"  SHAP smoke test: top feature index = {top} (expected 0)")

print("\nRESULT:", "ALL OK" if not failed and top == 0 else f"PROBLEMS: {failed}")
