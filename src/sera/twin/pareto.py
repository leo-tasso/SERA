"""NSGA-II Pareto front over the efficiency–equity trade-off.

Instead of optimising one ethical objective at a time (`sera.twin.objectives`),
this module evolves a population of uniform national lever vectors against
**three objectives at once** — total final-year GDP (efficiency), inter-
provincial Gini (inequality, minimised), and the worst-off province's GDP (the
floor) — and returns the non-dominated front. The result reframes the ethics
dashboard: the efficiency–equity trade-off is a *frontier*, not a dropdown,
and each named ethical framework is simply a corner of it.

The search space is the shared national lever vector (one point in
``[0, 1]^n_levers``): cheap enough that a population NSGA-II is affordable
against the slow twin, and every front point stays directly inspectable as a
lever table.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from sera.twin.objectives import gini

GDP_KEY = "gdp_per_capita"

# Corner tags: which single-objective framework each extreme point corresponds to.
CORNER_TAGS = {
    "gdp_total": "utilitarian",
    "neg_gini": "egalitarian",
    "worst_gdp": "rawlsian",
}


def evaluate_levers(env, x: np.ndarray) -> dict:
    """Roll out one shared lever vector and score the final simulated year."""
    from sera.twin.policy import UniformLeverPolicy

    policy = UniformLeverPolicy(env.param_specs)
    policy.prepare(env)
    policy.x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    trajectory, gdp_series, _welfare, _allocations, reserve = env.rollout(policy)

    values = np.array([], dtype=float)
    if not trajectory.empty and GDP_KEY in trajectory.columns:
        final_year = int(env.base_year + env.horizon)
        final_rows = trajectory[trajectory["year"] == final_year]
        values = (
            final_rows[GDP_KEY].astype(float).to_numpy()
            if not final_rows.empty
            else np.array([], dtype=float)
        )
    if values.size == 0:
        return {"gdp_total": 0.0, "gini": 0.0, "worst_gdp": 0.0, "reserve": float(reserve)}
    return {
        "gdp_total": float(values.sum()),
        "gini": float(gini(values)),
        "worst_gdp": float(values.min()),
        "reserve": float(reserve),
    }


def _maximization_matrix(metrics: List[dict]) -> np.ndarray:
    """Metrics as a maximisation matrix: [gdp_total, -gini, worst_gdp]."""
    return np.array(
        [[m["gdp_total"], -m["gini"], m["worst_gdp"]] for m in metrics], dtype=float
    )


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Pareto dominance for maximisation: a is at least as good everywhere, better somewhere."""
    return bool(np.all(a >= b) and np.any(a > b))


def non_dominated_sort(F: np.ndarray) -> List[List[int]]:
    """Fast non-dominated sort; returns fronts as lists of row indices."""
    n = F.shape[0]
    dominated_by: List[List[int]] = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(F[i], F[j]):
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif dominates(F[j], F[i]):
                dominated_by[j].append(i)
                domination_count[i] += 1
    for i in range(n):
        if domination_count[i] == 0:
            fronts[0].append(i)

    current = 0
    while fronts[current]:
        next_front: List[int] = []
        for i in fronts[current]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current += 1
        fronts.append(next_front)
    return [front for front in fronts if front]


def crowding_distance(F: np.ndarray, front: List[int]) -> np.ndarray:
    """Crowding distance of each member of one front (higher = more isolated)."""
    size = len(front)
    distance = np.zeros(size, dtype=float)
    if size <= 2:
        return np.full(size, np.inf)
    sub = F[front]
    for col in range(sub.shape[1]):
        order = np.argsort(sub[:, col])
        span = sub[order[-1], col] - sub[order[0], col]
        distance[order[0]] = distance[order[-1]] = np.inf
        if span <= 0:
            continue
        for pos in range(1, size - 1):
            distance[order[pos]] += (
                sub[order[pos + 1], col] - sub[order[pos - 1], col]
            ) / span
    return distance


def _rank_and_crowding(F: np.ndarray) -> tuple:
    ranks = np.zeros(F.shape[0], dtype=int)
    crowding = np.zeros(F.shape[0], dtype=float)
    for rank, front in enumerate(non_dominated_sort(F)):
        distances = crowding_distance(F, front)
        for pos, index in enumerate(front):
            ranks[index] = rank
            crowding[index] = distances[pos]
    return ranks, crowding


def nsga2_front(
    env,
    *,
    popsize: int = 12,
    generations: int = 6,
    seed: int = 0,
    mutation_prob: float = 0.35,
    mutation_sigma: float = 0.15,
    progress=None,
) -> dict:
    """Evolve uniform lever vectors against (GDP, −Gini, worst-off GDP).

    Returns the non-dominated front over *every* evaluated candidate (an
    archive, not just the final population), with each extreme point tagged by
    the single-objective ethical framework it corresponds to.
    """
    rng = np.random.default_rng(seed)
    dim = len(env.param_specs)
    popsize = max(4, int(popsize))

    # Seed the population with the historical baseline and the mid-range point
    # so the front is anchored near "do nothing".
    baseline_x = np.array(
        [
            (spec.baseline - spec.min) / max(spec.max - spec.min, 1e-9)
            for spec in env.param_specs
        ],
        dtype=float,
    ).clip(0.0, 1.0)
    population = np.vstack(
        [baseline_x, np.full(dim, 0.5), rng.uniform(0.0, 1.0, size=(popsize - 2, dim))]
    )
    metrics = [evaluate_levers(env, x) for x in population]
    archive_x = [x.copy() for x in population]
    archive_metrics = list(metrics)
    if progress is not None:
        progress(0, generations, len(archive_x))

    for generation in range(1, generations + 1):
        F = _maximization_matrix(metrics)
        ranks, crowding = _rank_and_crowding(F)

        def tournament() -> np.ndarray:
            i, j = rng.integers(len(population)), rng.integers(len(population))
            if ranks[i] != ranks[j]:
                return population[i] if ranks[i] < ranks[j] else population[j]
            return population[i] if crowding[i] >= crowding[j] else population[j]

        children = []
        while len(children) < popsize:
            parent_a, parent_b = tournament(), tournament()
            mask = rng.random(dim) < 0.5  # uniform crossover
            child = np.where(mask, parent_a, parent_b).astype(float)
            mutate = rng.random(dim) < mutation_prob
            child[mutate] += mutation_sigma * rng.standard_normal(int(mutate.sum()))
            children.append(np.clip(child, 0.0, 1.0))
        children = np.array(children)
        child_metrics = [evaluate_levers(env, x) for x in children]
        archive_x.extend(x.copy() for x in children)
        archive_metrics.extend(child_metrics)

        # Environmental selection over parents + children.
        combined = np.vstack([population, children])
        combined_metrics = metrics + child_metrics
        F_combined = _maximization_matrix(combined_metrics)
        ranks_c, crowding_c = _rank_and_crowding(F_combined)
        order = np.lexsort((-crowding_c, ranks_c))[:popsize]
        population = combined[order]
        metrics = [combined_metrics[index] for index in order]

        if progress is not None:
            progress(generation, generations, len(archive_x))

    # Final front over the whole archive, deduplicated on rounded levers.
    F_archive = _maximization_matrix(archive_metrics)
    first_front = non_dominated_sort(F_archive)[0]
    seen = set()
    points = []
    for index in first_front:
        key = tuple(np.round(archive_x[index], 3))
        if key in seen:
            continue
        seen.add(key)
        points.append({"x": archive_x[index], "metrics": archive_metrics[index]})
    points.sort(key=lambda point: point["metrics"]["gini"])

    # Tag the extreme points with the ethical framework they correspond to.
    if points:
        corner_indices = {
            "gdp_total": max(range(len(points)), key=lambda i: points[i]["metrics"]["gdp_total"]),
            "neg_gini": min(range(len(points)), key=lambda i: points[i]["metrics"]["gini"]),
            "worst_gdp": max(range(len(points)), key=lambda i: points[i]["metrics"]["worst_gdp"]),
        }
        for point in points:
            point["tags"] = []
        for metric_key, point_index in corner_indices.items():
            points[point_index]["tags"].append(CORNER_TAGS[metric_key])

    return {
        "points": points,
        "evaluations": len(archive_x),
        "popsize": int(popsize),
        "generations": int(generations),
    }
