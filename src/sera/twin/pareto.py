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


def _final_year_metrics(env, trajectory, reserve: float) -> dict:
    """The three Pareto objectives, read off the final simulated year."""
    values = np.array([], dtype=float)
    if trajectory is not None and not trajectory.empty and GDP_KEY in trajectory.columns:
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


def _baseline_unit(env) -> np.ndarray:
    """The historical baseline lever vector, normalised into the ``[0, 1]`` box."""
    return np.array(
        [
            (spec.baseline - spec.min) / max(spec.max - spec.min, 1e-9)
            for spec in env.param_specs
        ],
        dtype=float,
    ).clip(0.0, 1.0)


def evaluate_levers(env, x: np.ndarray) -> dict:
    """Roll out one shared lever vector and score the final simulated year."""
    from sera.twin.policy import UniformLeverPolicy

    policy = UniformLeverPolicy(env.param_specs)
    policy.prepare(env)
    policy.x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    trajectory, _gdp_series, _welfare, _allocations, reserve = env.rollout(policy)
    return _final_year_metrics(env, trajectory, reserve)


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


def _cluster_evaluator(env, n_clusters: int, seed: int):
    """Build a province-clustered evaluator for the NSGA-II search.

    Returns ``(evaluate, dim, decode)`` where the genome is one lever vector per
    cluster (``dim = k * n_levers``), ``evaluate(genome)`` rolls the resulting
    per-province policy through the twin, and ``decode(genome)`` yields a
    human-readable per-cluster lever table plus a national-average vector. The
    k-means assignment is fixed from the starting state, so the search only
    varies the per-cluster levers --- which is what gives the front the power to
    treat provinces differently and so expose a real efficiency--equity
    trade-off that uniform national policy cannot.
    """
    from sera.twin.policy import ClusterLeverPolicy

    policy = ClusterLeverPolicy(env.param_specs, n_clusters=int(n_clusters), seed=seed)
    policy.prepare(env)
    k = int(policy._k)
    n_levers = len(env.param_specs)
    mins = np.array([spec.min for spec in env.param_specs], dtype=float)
    spans = np.array([max(spec.max - spec.min, 0.0) for spec in env.param_specs], dtype=float)
    member_counts = [
        sum(1 for cluster in policy.assignments.values() if cluster == c) for c in range(k)
    ]

    def evaluate(genome: np.ndarray) -> dict:
        policy.genome = np.clip(np.asarray(genome, dtype=float), 0.0, 1.0)
        trajectory, _gdp, _welfare, _allocations, reserve = env.rollout(policy)
        return _final_year_metrics(env, trajectory, reserve)

    def decode(genome: np.ndarray) -> dict:
        levers = np.clip(np.asarray(genome, dtype=float), 0.0, 1.0).reshape(k, n_levers)
        values = mins + levers * spans  # (k, n_levers)
        # National average lever vector, population-unweighted across clusters,
        # so the bridge's existing per-lever table still has something to show.
        mean_unit = levers.mean(axis=0)
        clusters = [
            {
                "id": c,
                "provinces": member_counts[c],
                "levers": {
                    spec.key: float(values[c, col])
                    for col, spec in enumerate(env.param_specs)
                },
            }
            for c in range(k)
        ]
        return {"x": mean_unit, "clusters": clusters, "k": k}

    return evaluate, k * n_levers, decode


def nsga2_front(
    env,
    *,
    popsize: int = 12,
    generations: int = 6,
    seed: int = 0,
    n_clusters: int = 1,
    mutation_prob: float = 0.35,
    mutation_sigma: float = 0.15,
    progress=None,
) -> dict:
    """Evolve lever vectors against (GDP, −Gini, worst-off GDP).

    With ``n_clusters == 1`` the genome is a single national lever vector
    applied identically to every province (the original behaviour). With
    ``n_clusters > 1`` the genome is one lever vector per k-means cluster of
    provinces, so the search can *target* regions --- the experiment shows this
    is what turns the degenerate uniform "frontier" (a ray) into a real
    efficiency--equity trade-off.

    Returns the non-dominated front over *every* evaluated candidate (an
    archive, not just the final population), with each extreme point tagged by
    the single-objective ethical framework it corresponds to.
    """
    rng = np.random.default_rng(seed)
    popsize = max(4, int(popsize))
    clustered = int(n_clusters) > 1

    if clustered:
        evaluate, dim, decode = _cluster_evaluator(env, int(n_clusters), seed)
        baseline_x = np.tile(_baseline_unit(env), dim // len(env.param_specs))
    else:
        decode = None
        dim = len(env.param_specs)

        def evaluate(x):
            return evaluate_levers(env, x)

        baseline_x = _baseline_unit(env)

    # Seed the population with the historical baseline and the mid-range point
    # so the front is anchored near "do nothing".
    population = np.vstack(
        [baseline_x, np.full(dim, 0.5), rng.uniform(0.0, 1.0, size=(popsize - 2, dim))]
    )
    metrics = [evaluate(x) for x in population]
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
        child_metrics = [evaluate(x) for x in children]
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
        point = {"x": archive_x[index], "metrics": archive_metrics[index]}
        if clustered:
            decoded = decode(archive_x[index])
            point["genome"] = archive_x[index]
            point["x"] = decoded["x"]  # national-average vector for display
            point["clusters"] = decoded["clusters"]
        points.append(point)
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
        "nClusters": int(n_clusters) if clustered else 1,
    }
