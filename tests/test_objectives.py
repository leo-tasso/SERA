"""Tests for the pluggable ethical objectives."""

import numpy as np
import pandas as pd
import pytest

from sera.twin.objectives import (
    DEFAULT_OBJECTIVE_ID,
    EgalitarianObjective,
    Objective,
    RawlsianObjective,
    UtilitarianGdpObjective,
    WellbeingObjective,
    available_objectives,
    build_objective,
    gini,
)


def make_state(gdp=(100.0, 200.0, 300.0)):
    n = len(gdp)
    return pd.DataFrame(
        {
            "area_code": [f"P{i}" for i in range(n)],
            "year": [2025] * n,
            "gdp_per_capita": list(gdp),
            "life_expectancy": [82.0] * n,
            "unemployment_rate": [8.0] * n,
            "poverty_rate": [15.0] * n,
        }
    )


class TestGini:
    def test_perfect_equality_is_zero(self):
        assert gini(np.array([5.0, 5.0, 5.0, 5.0])) == pytest.approx(0.0)

    def test_total_concentration_approaches_one(self):
        value = gini(np.array([0.0] * 99 + [100.0]))
        assert value == pytest.approx(0.99, abs=1e-9)

    def test_known_two_value_case(self):
        # Two people, one has everything: G = 0.5.
        assert gini(np.array([0.0, 10.0])) == pytest.approx(0.5)

    def test_empty_and_zero_inputs_are_safe(self):
        assert gini(np.array([])) == 0.0
        assert gini(np.array([0.0, 0.0])) == 0.0


class TestUtilitarian:
    def test_score_is_total_gdp(self):
        assert UtilitarianGdpObjective().score_year(make_state()) == pytest.approx(600.0)

    def test_indifferent_to_distribution(self):
        equal = make_state(gdp=(200.0, 200.0, 200.0))
        unequal = make_state(gdp=(0.0, 100.0, 500.0))
        objective = UtilitarianGdpObjective()
        assert objective.score_year(equal) == pytest.approx(objective.score_year(unequal))


class TestRawlsian:
    def test_score_is_worst_province_scaled(self):
        assert RawlsianObjective().score_year(make_state()) == pytest.approx(100.0 * 3)

    def test_growth_elsewhere_does_not_count(self):
        objective = RawlsianObjective()
        before = objective.score_year(make_state(gdp=(100.0, 200.0, 300.0)))
        after = objective.score_year(make_state(gdp=(100.0, 400.0, 900.0)))
        assert after == pytest.approx(before)

    def test_lifting_the_worst_province_counts(self):
        objective = RawlsianObjective()
        before = objective.score_year(make_state(gdp=(100.0, 200.0, 300.0)))
        after = objective.score_year(make_state(gdp=(150.0, 200.0, 300.0)))
        assert after > before


class TestEgalitarian:
    def test_equal_distribution_scores_full_total(self):
        state = make_state(gdp=(200.0, 200.0, 200.0))
        assert EgalitarianObjective().score_year(state) == pytest.approx(600.0)

    def test_inequality_is_penalised_at_same_total(self):
        objective = EgalitarianObjective()
        equal = objective.score_year(make_state(gdp=(200.0, 200.0, 200.0)))
        unequal = objective.score_year(make_state(gdp=(50.0, 150.0, 400.0)))
        assert unequal < equal


class TestWellbeing:
    def test_baseline_state_scores_zero(self):
        objective = WellbeingObjective()
        state = make_state()
        objective.prepare(state, [])
        assert objective.score_year(state) == pytest.approx(0.0)

    def test_gdp_growth_alone_raises_score(self):
        objective = WellbeingObjective()
        base = make_state()
        objective.prepare(base, [])
        richer = base.copy()
        richer["gdp_per_capita"] = richer["gdp_per_capita"] * 1.1
        assert objective.score_year(richer) > 0.0

    def test_rising_unemployment_lowers_score(self):
        objective = WellbeingObjective()
        base = make_state()
        objective.prepare(base, [])
        worse = base.copy()
        worse["unemployment_rate"] = worse["unemployment_rate"] * 1.5
        assert objective.score_year(worse) < 0.0

    def test_gdp_gain_can_be_outweighed_by_social_losses(self):
        objective = WellbeingObjective()
        base = make_state()
        objective.prepare(base, [])
        tradeoff = base.copy()
        tradeoff["gdp_per_capita"] = tradeoff["gdp_per_capita"] * 1.05  # +5% * 0.35
        tradeoff["unemployment_rate"] = tradeoff["unemployment_rate"] * 1.30  # +30% * 0.20
        assert objective.score_year(tradeoff) < 0.0


class TestRegistry:
    def test_available_objectives_metadata(self):
        objectives = {item["id"]: item for item in available_objectives()}
        assert set(objectives) == {
            "utilitarian", "rawlsian", "cvar", "prioritarian",
            "egalitarian", "sufficientarian", "wellbeing",
        }
        for item in objectives.values():
            assert item["label"]
            assert item["description"]
            assert "parameters" in item  # may be empty, but always present

    def test_parameterized_objectives_expose_parameters(self):
        objectives = {item["id"]: item for item in available_objectives()}
        for oid in ("cvar", "prioritarian", "sufficientarian"):
            params = objectives[oid]["parameters"]
            assert params and all({"id", "min", "max", "default"} <= set(p) for p in params)

    def test_build_objective_accepts_and_filters_params(self):
        # Known parameter is applied; unknown parameter is ignored (no crash).
        assert build_objective("cvar", alpha=0.5).alpha == 0.5
        assert build_objective("utilitarian", alpha=0.5).objective_id == "utilitarian"

    def test_build_objective_by_id(self):
        assert isinstance(build_objective("rawlsian"), RawlsianObjective)
        assert isinstance(build_objective("egalitarian"), EgalitarianObjective)
        assert isinstance(build_objective("wellbeing"), WellbeingObjective)

    def test_unknown_id_falls_back_to_default(self):
        objective = build_objective("does-not-exist")
        assert isinstance(objective, UtilitarianGdpObjective)
        assert isinstance(objective, Objective)
        assert objective.objective_id == DEFAULT_OBJECTIVE_ID
