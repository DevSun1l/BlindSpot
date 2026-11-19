"""Step 11 tests: metric directions and the cross-seed consistency table."""
import numpy as np
import pandas as pd

from blindspot.experiment import METRICS, RELATIVE_TO_A, consistency, worse


def test_worse_directions():
    assert worse("wape", 0.3, 0.2) and not worse("wape", 0.2, 0.3)
    assert worse("fill_rate", 0.8, 0.9)
    assert worse("bias", -0.15, 0.05)                         # larger |bias|
    assert worse("injected_recovery", 0.6, 0.9)                # further from 100%
    assert worse("injected_recovery", 1.5, 1.1)                # over-recovery also counts
    assert worse("promo_uplift_vs_A", 0.6, 0.98)
    assert set(RELATIVE_TO_A) <= set(METRICS)


def test_consistency_counts():
    rows = []
    for s, (a, b, c) in zip([1, 2, 3], [(0.29, 0.31, 0.30), (0.29, 0.32, 0.29), (0.29, 0.28, 0.30)]):
        rows += [(s, "A", "wape", a), (s, "B", "wape", b), (s, "C", "wape", c)]
        rows += [(s, "A", "row_spearman_vs_A", 1.0), (s, "B", "row_spearman_vs_A", 0.85), (s, "C", "row_spearman_vs_A", 0.9)]
    t = consistency(pd.DataFrame(rows, columns=["seed", "condition", "metric", "value"])).set_index("metric")
    assert t.loc["wape", "n_seeds"] == 3
    assert t.loc["wape", "B_worse_than_A"] == 2 and t.loc["wape", "B_worse_than_C"] == 2
    assert np.isclose(t.loc["wape", "B_min"], 0.28) and np.isclose(t.loc["wape", "B_max"], 0.32)
    assert np.isnan(t.loc["row_spearman_vs_A", "B_worse_than_A"])      # trivially true, so not reported
    assert t.loc["row_spearman_vs_A", "B_worse_than_C"] == 3
