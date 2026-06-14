"""Tests for the panel-estimation analysis layer."""

import numpy as np
import pandas as pd

from sera.twin import panel_estimation as pe


def synthetic_panel(n_entities=12, years=range(2001, 2021), seed=0):
    """Panel where B_t = +0.8*A_{t-1} and C_t = -0.6*A_{t-1}, with big entity offsets.

    The offsets are what a pooled (no-fixed-effects) regression confounds and a
    within transform removes; the lagged couplings are the signal to recover.
    """
    rng = np.random.default_rng(seed)
    rows = []
    a_prev = {}
    for e in range(n_entities):
        off_a, off_b, off_c = rng.normal(0, 50, size=3)
        a_prev[e] = 0.0
        for y in years:
            a = off_a + rng.normal(0, 1)
            b = off_b + 0.8 * a_prev[e] + rng.normal(0, 0.1)
            c = off_c - 0.6 * a_prev[e] + rng.normal(0, 0.1)
            rows.append({"entity": f"E{e}", "year": y, "A": a, "B": b, "C": c})
            a_prev[e] = a
    return pd.DataFrame(rows)


def test_two_way_within_removes_entity_offsets():
    panel = synthetic_panel()
    within = pe.two_way_within(panel, ["A"])
    # After demeaning, entity means of A are all (nearly) equal.
    entity_means = within.groupby("entity")["A"].mean()
    assert entity_means.std() < 1e-6


def test_add_lags_shifts_within_entity():
    panel = synthetic_panel(n_entities=2, years=range(2001, 2005))
    lagged = pe.add_lags(panel, ["A"], lag=1)
    first_rows = lagged.groupby("entity").head(1)
    assert first_rows["A_lag1"].isna().all()  # no lag available for first year


def test_learn_couplings_recovers_signs_with_fixed_effects():
    panel = synthetic_panel()
    learned = pe.learn_couplings(panel, fe=True)
    assert learned["B"]["A"] > 0  # B follows A positively
    assert learned["C"]["A"] < 0  # C follows A negatively
    # A has no real driver; its coefficients should be comparatively small.
    assert abs(learned["B"]["A"]) > abs(learned["A"].get("B", 0.0))


def test_coupling_agreement_scores_signs():
    # income (dir +1) -> poverty_rate (dir -1): documented sign is negative.
    learned = {"poverty_rate": {"income": -2.0}, "income": {"poverty_rate": +1.0}}
    graph = {"income": ["poverty_rate"]}
    result = pe.coupling_agreement(learned, graph)
    assert result["edges_scored"] == 1
    assert result["sign_agree"] == 1  # data sign (−) matches documented (−)

    learned_wrong = {"poverty_rate": {"income": +2.0}}
    result_wrong = pe.coupling_agreement(learned_wrong, graph)
    assert result_wrong["sign_disagree"] == 1


def test_expected_sign_uses_indicator_directions():
    # Two "good" indicators move together (+); good vs bad move against (−).
    assert pe.expected_sign("income", "gdp_per_capita") == 1
    assert pe.expected_sign("income", "poverty_rate") == -1


def test_temporal_backtest_runs_and_separates_from_random():
    panel = synthetic_panel(n_entities=15)
    bt = pe.temporal_backtest(panel, fe=False, min_samples=20)
    assert bt["targets"] >= 1
    assert np.isfinite(bt["random_r2_mean"])
    assert "temporal_r2_median" in bt and "direction_accuracy_mean" in bt
