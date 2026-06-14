"""Tests for the NSGA-II efficiency-equity Pareto frontier."""

import numpy as np
import pandas as pd

from sera.twin.pareto import (
    crowding_distance,
    dominates,
    evaluate_levers,
    non_dominated_sort,
    nsga2_front,
)
from sera.twin.policy import ParamSpec, RolloutEnv

PROVINCES = ["AA", "BB", "CC"]
SPECS = [
    ParamSpec("lever_a", baseline=10.0, min=0.0, max=20.0),
    ParamSpec("lever_b", baseline=50.0, min=25.0, max=100.0),
]

# Per-province sensitivity to lever_a: raising the (uniform) lever grows total
# GDP but concentrates the gains, so efficiency and equality genuinely conflict.
WEIGHTS = {"AA": 0.5, "BB": 1.0, "CC": 3.0}


class TradeoffSimulator:
    def simulate_year(self, current_state, parameters_year, apply_rules=True, apply_bounds=True):
        merged = current_state.merge(
            parameters_year[["area_code", "lever_a"]], on="area_code", how="left"
        )
        result = current_state.copy()
        result["year"] = current_state["year"] + 1
        weights = current_state["area_code"].map(WEIGHTS).to_numpy(dtype=float)
        result["gdp_per_capita"] = (
            current_state["gdp_per_capita"].values + merged["lever_a"].values * weights
        )
        return result


def make_env(horizon=2):
    initial = pd.DataFrame(
        {
            "area_code": PROVINCES,
            "year": [2025] * len(PROVINCES),
            "gdp_per_capita": [100.0, 200.0, 300.0],
        }
    )
    return RolloutEnv(
        simulator=TradeoffSimulator(),
        initial_state=initial,
        indicator_cols=["gdp_per_capita"],
        param_specs=SPECS,
        provinces=PROVINCES,
        horizon=horizon,
        base_year=2025,
    )


class TestDominance:
    def test_dominates_requires_strict_improvement_somewhere(self):
        assert dominates(np.array([2.0, 1.0]), np.array([1.0, 1.0]))
        assert not dominates(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
        assert not dominates(np.array([2.0, 0.5]), np.array([1.0, 1.0]))

    def test_non_dominated_sort_orders_fronts(self):
        F = np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 3.0], [0.1, 0.1]])
        fronts = non_dominated_sort(F)
        assert sorted(fronts[0]) == [1, 2]  # mutually non-dominated
        assert 0 in fronts[1]
        assert 3 in fronts[-1]

    def test_crowding_distance_boundary_points_are_infinite(self):
        F = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
        front = [0, 1, 2, 3]
        distances = crowding_distance(F, front)
        assert np.isinf(distances[0]) and np.isinf(distances[-1])
        assert np.all(distances[1:-1] < np.inf)


class TestEvaluateLevers:
    def test_metrics_are_finite_and_keyed(self):
        env = make_env()
        metrics = evaluate_levers(env, np.array([0.5, 0.5]))
        assert set(metrics) == {"gdp_total", "gini", "worst_gdp", "reserve"}
        assert all(np.isfinite(value) for value in metrics.values())

    def test_high_lever_raises_gdp_and_inequality(self):
        env = make_env()
        low = evaluate_levers(env, np.array([0.0, 0.5]))
        high = evaluate_levers(env, np.array([1.0, 0.5]))
        assert high["gdp_total"] > low["gdp_total"]
        assert high["gini"] > low["gini"]


class TestNsga2Front:
    def test_front_is_mutually_non_dominated(self):
        env = make_env()
        result = nsga2_front(env, popsize=6, generations=3, seed=0)
        points = result["points"]
        assert points
        F = np.array(
            [
                [p["metrics"]["gdp_total"], -p["metrics"]["gini"], p["metrics"]["worst_gdp"]]
                for p in points
            ]
        )
        for i in range(len(points)):
            for j in range(len(points)):
                if i != j:
                    assert not dominates(F[i], F[j])

    def test_archive_size_and_corner_tags(self):
        env = make_env()
        popsize, generations = 6, 3
        result = nsga2_front(env, popsize=popsize, generations=generations, seed=0)
        assert result["evaluations"] == popsize * (generations + 1)
        tags = {tag for point in result["points"] for tag in point.get("tags", [])}
        assert tags == {"utilitarian", "egalitarian", "rawlsian"}

    def test_points_sorted_by_gini_and_in_unit_box(self):
        env = make_env()
        result = nsga2_front(env, popsize=6, generations=2, seed=1)
        ginis = [point["metrics"]["gini"] for point in result["points"]]
        assert ginis == sorted(ginis)
        for point in result["points"]:
            assert np.all(point["x"] >= 0.0) and np.all(point["x"] <= 1.0)

    def test_deterministic_for_a_seed(self):
        env = make_env()
        a = nsga2_front(env, popsize=6, generations=2, seed=7)
        b = nsga2_front(env, popsize=6, generations=2, seed=7)
        assert len(a["points"]) == len(b["points"])
        for pa, pb in zip(a["points"], b["points"]):
            assert pa["metrics"] == pb["metrics"]

    def test_progress_callback_called_per_generation(self):
        env = make_env()
        calls = []
        nsga2_front(
            env,
            popsize=6,
            generations=2,
            seed=0,
            progress=lambda gen, total, evals: calls.append(gen),
        )
        assert calls == [0, 1, 2]


class TestClusteredFront:
    def test_clustered_points_carry_cluster_tables(self):
        env = make_env()
        result = nsga2_front(env, popsize=8, generations=3, seed=0, n_clusters=3)
        assert result["nClusters"] == 3
        for point in result["points"]:
            # x stays a national-average lever vector for display compatibility.
            assert len(point["x"]) == len(SPECS)
            assert "clusters" in point and point["clusters"]
            for cluster in point["clusters"]:
                assert set(cluster["levers"]) == {spec.key for spec in SPECS}

    def test_clustering_can_separate_provinces(self):
        # Per-province targeting should reach a wider spread of inequality than
        # one shared national vector can, on a simulator built to reward it.
        env = make_env()
        uniform = nsga2_front(env, popsize=8, generations=4, seed=0, n_clusters=1)
        clustered = nsga2_front(env, popsize=8, generations=4, seed=0, n_clusters=3)
        uni_span = max(p["metrics"]["gini"] for p in uniform["points"]) - min(
            p["metrics"]["gini"] for p in uniform["points"]
        )
        clu_span = max(p["metrics"]["gini"] for p in clustered["points"]) - min(
            p["metrics"]["gini"] for p in clustered["points"]
        )
        assert clu_span >= uni_span
