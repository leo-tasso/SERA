"""Tests for the pluggable policy models and the rollout environment."""

import numpy as np
import pandas as pd
import pytest

from sera.twin.objectives import RawlsianObjective
from sera.twin.policy import (
    BaselinePolicy,
    BlendedPolicy,
    NeuralPolicy,
    ParamSpec,
    PolicyModel,
    RolloutEnv,
    UniformLeverPolicy,
    available_models,
    build_policy,
)

PROVINCES = ["AA", "BB", "CC"]
SPECS = [
    ParamSpec("lever_a", baseline=10.0, min=0.0, max=20.0),
    ParamSpec("lever_b", baseline=50.0, min=25.0, max=100.0),
]


class GrowthSimulator:
    """Stub twin: GDP grows by lever_a's province value each year."""

    def simulate_year(self, current_state, parameters_year, apply_rules=True, apply_bounds=True):
        merged = current_state.merge(
            parameters_year[["area_code", "lever_a"]], on="area_code", how="left"
        )
        result = current_state.copy()
        result["year"] = current_state["year"] + 1
        result["gdp_per_capita"] = (
            current_state["gdp_per_capita"].values + merged["lever_a"].values
        )
        return result


def make_state():
    return pd.DataFrame(
        {
            "area_code": PROVINCES,
            "year": [2025] * len(PROVINCES),
            "gdp_per_capita": [100.0, 200.0, 300.0],
            "unemployment_rate": [8.0, 6.0, 10.0],
        }
    )


def make_env(horizon=3, constraint_fn=None, reserve_pool=0.0, objective=None):
    return RolloutEnv(
        simulator=GrowthSimulator(),
        initial_state=make_state(),
        indicator_cols=["gdp_per_capita", "unemployment_rate"],
        param_specs=SPECS,
        provinces=PROVINCES,
        horizon=horizon,
        base_year=2025,
        constraint_fn=constraint_fn,
        reserve_pool=reserve_pool,
        objective=objective,
    )


class TestRolloutEnv:
    def test_rollout_returns_five_tuple_with_expected_shapes(self):
        env = make_env(horizon=3)
        trajectory, gdp_series, welfare_series, allocations_by_year, final_reserve = env.rollout(
            BaselinePolicy(SPECS)
        )
        assert len(gdp_series) == 3
        assert len(welfare_series) == 3
        assert sorted(allocations_by_year) == [2026, 2027, 2028]
        assert len(trajectory) == 3 * len(PROVINCES)
        assert final_reserve == 0.0

    def test_baseline_gdp_accumulates_lever_a(self):
        env = make_env(horizon=2)
        _t, gdp_series, _w, _a, _r = env.rollout(BaselinePolicy(SPECS))
        # Start total 600; +10 per province per year => 630, 660.
        assert gdp_series == [630.0, 660.0]

    def test_default_objective_welfare_equals_gdp(self):
        env = make_env(horizon=2)
        _t, gdp_series, welfare_series, _a, _r = env.rollout(BaselinePolicy(SPECS))
        assert welfare_series == gdp_series

    def test_rawlsian_objective_scores_worst_province(self):
        env = make_env(horizon=2, objective=RawlsianObjective())
        _t, _g, welfare_series, _a, _r = env.rollout(BaselinePolicy(SPECS))
        # Worst province starts at 100, +10/year; min * n_provinces.
        assert welfare_series == [110.0 * 3, 120.0 * 3]
        assert env.score(BaselinePolicy(SPECS)) == pytest.approx(330.0 + 360.0)

    def test_constraint_fn_receives_and_returns_reserve(self):
        seen = []

        def constraint(allocations, state, reserve):
            seen.append(reserve)
            return allocations, reserve + 5.0

        env = make_env(horizon=3, constraint_fn=constraint, reserve_pool=1.0)
        _t, _g, _w, _a, final_reserve = env.rollout(BaselinePolicy(SPECS))
        assert seen == [1.0, 6.0, 11.0]
        assert final_reserve == 16.0

    def test_params_frame_clamps_and_fills_defaults(self):
        env = make_env()
        frame = env._build_params_frame(
            2026,
            {
                "AA": {"lever_a": 999.0, "lever_b": "not-a-number"},
                # BB intentionally missing entirely.
                "CC": {"lever_a": -10.0},
            },
        )
        row_aa = frame[frame["area_code"] == "AA"].iloc[0]
        row_bb = frame[frame["area_code"] == "BB"].iloc[0]
        row_cc = frame[frame["area_code"] == "CC"].iloc[0]
        assert row_aa["lever_a"] == 20.0  # clamped to max
        assert row_aa["lever_b"] == 50.0  # invalid value -> baseline
        assert row_bb["lever_a"] == 10.0  # missing province -> baseline
        assert row_cc["lever_a"] == 0.0  # clamped to min

    def test_score_is_cumulative_gdp(self):
        env = make_env(horizon=2)
        assert env.score(BaselinePolicy(SPECS)) == pytest.approx(630.0 + 660.0)


class TestPolicies:
    def test_baseline_decide_covers_all_provinces(self):
        env = make_env()
        allocations = BaselinePolicy(SPECS).decide(make_state(), 0, env)
        assert set(allocations) == set(PROVINCES)
        assert allocations["AA"] == {"lever_a": 10.0, "lever_b": 50.0}

    def test_neural_decide_respects_bounds(self):
        env = make_env()
        policy = NeuralPolicy(SPECS, seed=1)
        allocations = policy.decide(make_state(), 0, env)
        for province in PROVINCES:
            for spec in SPECS:
                value = allocations[province][spec.key]
                assert spec.min <= value <= spec.max

    def test_neural_is_deterministic_for_a_seed(self):
        env = make_env()
        a = NeuralPolicy(SPECS, seed=7).decide(make_state(), 0, env)
        b = NeuralPolicy(SPECS, seed=7).decide(make_state(), 0, env)
        assert a == b

    def test_neural_fit_improves_or_keeps_score(self):
        env = make_env(horizon=2)
        policy = NeuralPolicy(SPECS, seed=0)
        info = policy.fit(env, iterations=3)
        assert info["best_score"] >= info["start_score"]
        assert env.score(policy) == pytest.approx(info["best_score"])

    def test_neural_fit_optimizes_env_objective(self):
        env = make_env(horizon=2, objective=RawlsianObjective())
        policy = NeuralPolicy(SPECS, seed=0)
        info = policy.fit(env, iterations=3)
        assert env.score(policy) == pytest.approx(info["best_score"])

    def test_fit_progress_callback_called(self):
        env = make_env(horizon=1)
        calls = []
        NeuralPolicy(SPECS, seed=0).fit(
            env, iterations=2, progress=lambda step, total, score: calls.append(step)
        )
        assert calls == [0, 1, 2]

    def test_uniform_decide_is_identical_across_provinces_and_in_bounds(self):
        env = make_env()
        allocations = UniformLeverPolicy(SPECS, seed=1).decide(make_state(), 0, env)
        assert set(allocations) == set(PROVINCES)
        first = allocations[PROVINCES[0]]
        for province in PROVINCES:
            assert allocations[province] == first
        for spec in SPECS:
            assert spec.min <= first[spec.key] <= spec.max

    def test_uniform_fit_improves_or_keeps_score(self):
        env = make_env(horizon=2)
        policy = UniformLeverPolicy(SPECS, seed=0)
        info = policy.fit(env, iterations=4)
        assert info["best_score"] >= info["start_score"]
        assert env.score(policy) == pytest.approx(info["best_score"])

    def test_uniform_fit_works_under_any_objective(self):
        env = make_env(horizon=2, objective=RawlsianObjective())
        policy = UniformLeverPolicy(SPECS, seed=0)
        info = policy.fit(env, iterations=4)
        assert env.score(policy) == pytest.approx(info["best_score"])

    def test_blended_full_blend_matches_inner_policy(self):
        env = make_env()
        inner = NeuralPolicy(SPECS, seed=3)
        blended = BlendedPolicy(inner, blend=1.0).decide(make_state(), 0, env)
        assert blended == inner.decide(make_state(), 0, env)

    def test_blended_zero_blend_is_baseline(self):
        env = make_env()
        inner = NeuralPolicy(SPECS, seed=3)
        blended = BlendedPolicy(inner, blend=0.0).decide(make_state(), 0, env)
        assert blended == BaselinePolicy(SPECS).decide(make_state(), 0, env)

    def test_blended_half_blend_is_halfway(self):
        env = make_env()
        inner = NeuralPolicy(SPECS, seed=3)
        full = inner.decide(make_state(), 0, env)
        half = BlendedPolicy(inner, blend=0.5).decide(make_state(), 0, env)
        for province in PROVINCES:
            for spec in SPECS:
                expected = spec.baseline + 0.5 * (full[province][spec.key] - spec.baseline)
                assert half[province][spec.key] == pytest.approx(expected)

    def test_blended_clamps_blend_into_unit_interval(self):
        inner = BaselinePolicy(SPECS)
        assert BlendedPolicy(inner, blend=2.5).blend == 1.0
        assert BlendedPolicy(inner, blend=-1.0).blend == 0.0


class TestRegistry:
    def test_available_models_metadata(self):
        models = {m["id"]: m for m in available_models()}
        assert models["baseline"]["trainable"] is False
        assert models["neural"]["trainable"] is True
        assert models["uniform_cem"]["trainable"] is True

    def test_build_policy_by_id(self):
        assert isinstance(build_policy("baseline", SPECS), BaselinePolicy)
        assert isinstance(build_policy("neural", SPECS), NeuralPolicy)
        assert isinstance(build_policy("uniform_cem", SPECS), UniformLeverPolicy)

    def test_build_policy_resolves_legacy_gdp_nn_alias(self):
        assert isinstance(build_policy("gdp_nn", SPECS), NeuralPolicy)

    def test_build_policy_unknown_id_falls_back_to_neural(self):
        policy = build_policy("does-not-exist", SPECS)
        assert isinstance(policy, NeuralPolicy)
        assert isinstance(policy, PolicyModel)
