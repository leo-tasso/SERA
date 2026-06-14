"""Tests for the post-hoc explanation tools (permutation importance, distillation)."""

import numpy as np
import pandas as pd

from sera.twin.explain import (
    distill_to_tree,
    explain_policy_posthoc,
    permutation_importance,
)
from sera.twin.policy import NeuralPolicy, ParamSpec, RolloutEnv

PROVINCES = ["AA", "BB", "CC", "DD"]
SPECS = [
    ParamSpec("lever_a", baseline=10.0, min=0.0, max=20.0),
    ParamSpec("lever_b", baseline=50.0, min=25.0, max=100.0),
]


class GrowthSimulator:
    def simulate_year(self, current_state, parameters_year, apply_rules=True, apply_bounds=True):
        merged = current_state.merge(
            parameters_year[["area_code", "lever_a"]], on="area_code", how="left"
        )
        result = current_state.copy()
        result["year"] = current_state["year"] + 1
        result["gdp_per_capita"] = current_state["gdp_per_capita"].values + merged["lever_a"].values
        return result


def make_state():
    return pd.DataFrame(
        {
            "area_code": PROVINCES,
            "year": [2025] * len(PROVINCES),
            "gdp_per_capita": [100.0, 200.0, 300.0, 150.0],
            "unemployment_rate": [8.0, 6.0, 10.0, 12.0],
        }
    )


def make_env(horizon=2):
    return RolloutEnv(
        simulator=GrowthSimulator(),
        initial_state=make_state(),
        indicator_cols=["gdp_per_capita", "unemployment_rate"],
        param_specs=SPECS,
        provinces=PROVINCES,
        horizon=horizon,
        base_year=2025,
    )


def make_prepared_policy(env):
    policy = NeuralPolicy(SPECS, seed=3)
    policy.prepare(env)
    return policy


class TestPermutationImportance:
    def test_returns_one_entry_per_feature_sorted_desc(self):
        env = make_env()
        policy = make_prepared_policy(env)
        importances = permutation_importance(policy, env, seed=0)
        assert {item["feature"] for item in importances} == {
            "gdp_per_capita",
            "unemployment_rate",
        }
        values = [item["importance"] for item in importances]
        assert values == sorted(values, reverse=True)
        assert all(value >= 0.0 for value in values)

    def test_deterministic_for_a_seed(self):
        env = make_env()
        policy = make_prepared_policy(env)
        a = permutation_importance(policy, env, seed=1)
        b = permutation_importance(policy, env, seed=1)
        assert a == b


class TestDistillation:
    def test_surrogate_reports_fidelity_and_segments(self):
        env = make_env()
        policy = make_prepared_policy(env)
        surrogate = distill_to_tree(policy, env, max_depth=2, seed=0)
        assert surrogate is not None
        assert np.isfinite(surrogate["fidelity_r2"])
        assert surrogate["fidelity_r2"] <= 1.0
        assert surrogate["max_depth"] == 2
        assert surrogate["segments"]
        for segment in surrogate["segments"]:
            assert segment["conditions"]
            assert segment["n_samples"] >= 1
            for lever in segment["levers"]:
                spec = next(s for s in SPECS if s.key == lever["lever"])
                assert spec.min <= lever["value"] <= spec.max

    def test_returns_none_without_features(self):
        env = make_env()
        policy = NeuralPolicy(SPECS, feature_keys=[], seed=0)
        # Force-prepare with no features (e.g. no indicator overlap).
        policy.feature_keys = []
        policy.ref_means = {}
        policy._mins = np.array([s.min for s in SPECS])
        policy._spans = np.array([s.max - s.min for s in SPECS])
        policy._init_network(np.random.default_rng(0))
        assert distill_to_tree(policy, env) is None


class TestPosthocBundle:
    def test_bundle_shape(self):
        env = make_env()
        policy = make_prepared_policy(env)
        bundle = explain_policy_posthoc(policy, env)
        assert bundle["type"] == "neural_posthoc"
        assert bundle["importances"]
        assert bundle["surrogate"] is not None

    def test_neural_policy_explain_uses_bundle(self):
        env = make_env()
        policy = NeuralPolicy(SPECS, seed=0)
        explanation = policy.explain(env)
        assert explanation["type"] == "neural_posthoc"

    def test_neural_policy_explain_without_env_is_none(self):
        policy = NeuralPolicy(SPECS, seed=0)
        assert policy.explain() is None
