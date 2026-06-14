"""Pluggable policy-model interface that lets different ML models drive the twin.

A :class:`PolicyModel` maps the current national/provincial state to a set of
annual policy parameters. This is the seam that allows *different* machine
learning models (or hand-written rules) to "use" the digital twin: each model
only has to implement :meth:`PolicyModel.decide`. A :class:`RolloutEnv` wraps
the :class:`DigitalTwinSimulator` and rolls a policy forward over a multi-year
horizon, applying the national budget constraint each year.

What the optimizer maximises is *not* hard-wired: the env scores rollouts with
a pluggable ethical :class:`~sera.twin.objectives.Objective` (utilitarian GDP,
Rawlsian maximin, egalitarian Sen welfare, multi-indicator wellbeing), so the
ethical framework and the search model are independent choices.

All trainable models are objective-agnostic and gradient-free (the twin is a
non-differentiable black box of sklearn models plus rules). They span a
deliberate **explainability spectrum** — every model carries an
``explainability`` tag and an :meth:`PolicyModel.explain` artifact, so the UI
can show *why* a policy chose its levers, not just what it chose:

- :class:`NeuralPolicy` — a tiny pure-NumPy MLP applied per province, trained
  with mirrored-sampling Evolution Strategies. Black box; explained post hoc
  via permutation importance and surrogate-tree distillation
  (:mod:`sera.twin.explain`).
- :class:`LinearPolicy` — the same per-province setup with no hidden layer:
  each lever is a sigmoid-squashed linear function of the province's
  indicators. White box; the signed weight matrix *is* the explanation.
- :class:`DecisionListPolicy` — one human-readable IF/THEN rule per lever,
  thresholds and levels found with the Cross-Entropy Method. White box.
- :class:`ClusterLeverPolicy` — provinces grouped into a few indicator
  clusters, one shared lever vector per cluster (regional policy packages),
  found with CEM. White box.
- :class:`UniformLeverPolicy` — one shared national lever vector applied to
  every province, found with the Cross-Entropy Method. White box.
- :class:`BayesianUniformPolicy` — the same shared lever vector found with
  Gaussian-process Bayesian optimisation. Gray box: far fewer twin rollouts,
  and the fitted surrogate yields per-lever partial-dependence curves with
  uncertainty.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from sera.twin.objectives import Objective, UtilitarianGdpObjective
from sera.twin.simulator import DigitalTwinSimulator

logger = logging.getLogger(__name__)

GDP_KEY = "gdp_per_capita"

# Indicators the neural policy reads (intersected with what's actually available).
PREFERRED_FEATURE_KEYS = [
    "gdp_per_capita",
    "unemployment_rate",
    "income",
    "life_expectancy",
    "business_density",
    "poverty_rate",
]


@dataclass
class ParamSpec:
    """Bounds for one policy lever, used to clamp/decode model outputs."""

    key: str
    baseline: float
    min: float
    max: float


def _fit_report(best_score: float, start_score: float, iterations: int, **extra) -> dict:
    """The training summary every ``fit`` returns to the UI."""
    report = {
        "best_score": float(best_score),
        "start_score": float(start_score),
        "improvement_pct": (
            float(100.0 * (best_score - start_score) / abs(start_score)) if start_score else 0.0
        ),
        "iterations": int(iterations),
    }
    report.update(extra)
    return report


def cem_optimize(
    evaluate: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    iterations: int,
    popsize: int,
    elite_frac: float,
    smoothing: float,
    rng: np.random.Generator,
    progress=None,
    sigma_init: float = 0.25,
    sigma_floor: float = 0.02,
) -> dict:
    """Cross-Entropy Method over a vector in ``[0, 1]^dim``.

    Keeps a diagonal Gaussian over the search space, samples ``popsize``
    candidates per iteration, and refits the Gaussian to the elite fraction.
    Shared by every CEM-trained policy (uniform, rules, clusters).
    """
    dim = int(x0.size)
    n_elite = max(1, int(round(popsize * elite_frac)))
    mu = x0.copy()
    sigma = np.full(dim, sigma_init)
    best_x = x0.copy()
    best_score = float(evaluate(best_x))
    start_score = best_score
    history = [best_score]  # best-so-far per iteration, index 0 = untrained
    if progress is not None:
        progress(0, iterations, best_score)

    for iteration in range(1, iterations + 1):
        candidates = np.clip(mu + sigma * rng.standard_normal((popsize, dim)), 0.0, 1.0)
        scores = np.array([evaluate(candidate) for candidate in candidates])

        elite_idx = np.argsort(scores)[-n_elite:]
        elites = candidates[elite_idx]
        mu = smoothing * elites.mean(axis=0) + (1.0 - smoothing) * mu
        sigma = smoothing * elites.std(axis=0) + (1.0 - smoothing) * sigma
        sigma = np.maximum(sigma, sigma_floor)  # keep exploring

        top = int(elite_idx[-1])
        if scores[top] > best_score:
            best_score = float(scores[top])
            best_x = candidates[top].copy()

        history.append(best_score)
        if progress is not None:
            progress(iteration, iterations, best_score)

    return {
        "best_x": best_x,
        "best_score": best_score,
        "start_score": start_score,
        "history": history,
    }


def _kmeans(
    matrix: np.ndarray, n_clusters: int, rng: np.random.Generator, iterations: int = 30
) -> np.ndarray:
    """Plain-NumPy Lloyd's k-means; returns the cluster index per row.

    Deterministic for a given matrix: the first centre is the row closest to
    the overall mean, the rest are chosen farthest-point; ``rng`` only breaks
    exact ties via reassignment of empty clusters.
    """
    n_rows = matrix.shape[0]
    k = max(1, min(int(n_clusters), n_rows))
    if k == 1:
        return np.zeros(n_rows, dtype=int)

    first = int(np.argmin(((matrix - matrix.mean(axis=0)) ** 2).sum(axis=1)))
    centers = [matrix[first]]
    for _ in range(1, k):
        dist2 = np.min([((matrix - center) ** 2).sum(axis=1) for center in centers], axis=0)
        centers.append(matrix[int(np.argmax(dist2))])
    centers = np.array(centers, dtype=float)

    assignment = np.zeros(n_rows, dtype=int)
    for _ in range(iterations):
        dist2 = ((matrix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assignment = dist2.argmin(axis=1)
        for cluster in range(k):
            members = matrix[assignment == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
            else:
                centers[cluster] = matrix[int(rng.integers(n_rows))]
    return assignment


# --------------------------------------------------------------------------- #
# Rollout environment
# --------------------------------------------------------------------------- #
# Constraint callable: (allocations, state_df, reserve) -> (scaled_allocations, new_reserve)
ConstraintFn = Callable[
    [Dict[str, Dict[str, float]], pd.DataFrame, float],
    "tuple[Dict[str, Dict[str, float]], float]",
]


@dataclass
class RolloutEnv:
    """Run a policy forward through the twin for ``horizon`` years."""

    simulator: DigitalTwinSimulator
    initial_state: pd.DataFrame
    indicator_cols: List[str]
    param_specs: List[ParamSpec]
    provinces: List[str]
    horizon: int
    base_year: int
    constraint_fn: Optional[ConstraintFn] = None
    reserve_pool: float = 0.0
    objective: Optional[Objective] = None

    def __post_init__(self) -> None:
        self.param_keys = [spec.key for spec in self.param_specs]
        self._spec_by_key = {spec.key: spec for spec in self.param_specs}
        if self.objective is None:
            self.objective = UtilitarianGdpObjective()
        self.objective.prepare(self.initial_state, self.indicator_cols)

    def _build_params_frame(
        self, year: int, allocations: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        rows = []
        for province in self.provinces:
            province_alloc = allocations.get(province, {})
            row = {"area_code": province, "year": year}
            for spec in self.param_specs:
                raw = province_alloc.get(spec.key, spec.baseline)
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    value = float(spec.baseline)
                row[spec.key] = min(max(value, spec.min), spec.max)
            rows.append(row)
        return pd.DataFrame(rows)

    def national_gdp(self, state: pd.DataFrame) -> float:
        """Total national GDP proxy = sum of provincial GDP per capita."""
        if GDP_KEY not in state.columns:
            return 0.0
        series = pd.to_numeric(state[GDP_KEY], errors="coerce").fillna(0.0)
        return float(series.sum())

    def rollout(self, policy: "PolicyModel"):
        """Simulate the full horizon under ``policy``.

        Returns ``(trajectory_df, gdp_series, welfare_series, allocations_by_year,
        final_reserve)`` where ``gdp_series`` is the national GDP for each
        simulated year, ``welfare_series`` is the ethical objective's score for
        each year, and ``final_reserve`` is the unspent budget left after the
        last year.
        """
        state = self.initial_state.copy()
        reserve = float(self.reserve_pool)
        frames: List[pd.DataFrame] = []
        gdp_series: List[float] = []
        welfare_series: List[float] = []
        allocations_by_year: Dict[int, Dict[str, Dict[str, float]]] = {}

        for step in range(self.horizon):
            year = self.base_year + step + 1
            allocations = policy.decide(state, step, self)
            if self.constraint_fn is not None:
                allocations, reserve = self.constraint_fn(allocations, state, reserve)
            params_frame = self._build_params_frame(year, allocations)
            state = self.simulator.simulate_year(
                state, params_frame, apply_rules=True, apply_bounds=True
            )
            state = state.sort_values("area_code").reset_index(drop=True)
            frames.append(state.copy())
            gdp_series.append(self.national_gdp(state))
            welfare_series.append(float(self.objective.score_year(state)))
            allocations_by_year[year] = allocations

        trajectory = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return trajectory, gdp_series, welfare_series, allocations_by_year, reserve

    def score(self, policy: "PolicyModel") -> float:
        """Cumulative objective welfare over the horizon (the optimisation target)."""
        _trajectory, _gdp, welfare_series, _allocations, _reserve = self.rollout(policy)
        return float(np.sum(welfare_series))


# --------------------------------------------------------------------------- #
# Policy interface
# --------------------------------------------------------------------------- #
class PolicyModel(ABC):
    """Interface every model that drives the twin must implement."""

    model_id: str = "base"
    label: str = "Base policy"
    description: str = ""
    trainable: bool = False
    # How auditable the *resulting policy* is: "white-box" (directly readable),
    # "gray-box" (a transparent surrogate with uncertainty), or "black-box"
    # (only post-hoc explanations). Surfaced in the UI as a badge.
    explainability: Optional[str] = None

    def __init__(self, param_specs: List[ParamSpec]):
        self.param_specs = list(param_specs)

    @abstractmethod
    def decide(
        self, state: pd.DataFrame, step: int, env: "RolloutEnv"
    ) -> Dict[str, Dict[str, float]]:
        """Return per-province allocations ``{province: {lever: value}}`` for the year."""

    def fit(self, env: "RolloutEnv", *, iterations: int = 6, progress=None) -> dict:
        """Optional training hook. Default policies need no fitting."""
        return {}

    def explain(self, env: Optional["RolloutEnv"] = None) -> Optional[dict]:
        """JSON-serialisable explanation of the (trained) policy, or ``None``.

        The shape depends on the model class (``type`` discriminates); the UI
        renders each type differently. White-box models return their own
        parameters; black-box models return post-hoc artifacts.
        """
        return None


class BaselinePolicy(PolicyModel):
    """Holds every lever at its historical baseline (reference scenario)."""

    model_id = "baseline"
    label = "Baseline (historical levers)"
    description = "Keeps every policy lever at its historical baseline value."
    trainable = False
    explainability = "white-box"

    def decide(self, state, step, env):
        base = {spec.key: spec.baseline for spec in self.param_specs}
        return {code: dict(base) for code in env.provinces}


class NeuralPolicy(PolicyModel):
    """A tiny NumPy MLP policy trained to maximise the env's ethical objective.

    The same shared network is applied to *each province* using that province's
    own indicators (plus the year position), emitting a value in ``[0, 1]`` per
    lever decoded into the lever's bounds. Because every province feeds different
    state, each receives its own tailored lever vector while the weight count
    stays small. It is optimised with mirrored-sampling Evolution Strategies
    (gradient-free, since the twin is a non-differentiable black box of sklearn
    models + rules); the optimisation target is whatever ethical objective the
    rollout environment is configured with.
    """

    model_id = "neural"
    label = "Neural network (per-province levers)"
    description = (
        "A small neural network trained with evolution strategies. Each "
        "province gets levers tailored to its own indicators, chosen to "
        "maximise the selected ethical objective over the horizon. Not "
        "directly interpretable; audited post hoc with permutation "
        "importance and a distilled decision tree."
    )
    trainable = True
    explainability = "black-box"

    def __init__(
        self,
        param_specs: List[ParamSpec],
        feature_keys: Optional[List[str]] = None,
        n_hidden: int = 16,
        seed: int = 0,
    ):
        super().__init__(param_specs)
        self.feature_keys = list(feature_keys) if feature_keys is not None else []
        self.n_hidden = int(n_hidden)
        self.seed = int(seed)
        self.ref_means: Dict[str, float] = {}
        self._n_in = 0
        self._n_out = len(self.param_specs)
        self._shapes: List[tuple] = []
        self.theta: Optional[np.ndarray] = None
        self._mins: Optional[np.ndarray] = None
        self._spans: Optional[np.ndarray] = None

    # -- network plumbing ---------------------------------------------------- #
    def _init_network(self, rng: np.random.Generator) -> None:
        self._n_in = 2 + len(self.feature_keys)  # bias + year-position + features
        hidden, out = self.n_hidden, self._n_out
        self._shapes = [
            (self._n_in, hidden),  # W1
            (hidden,),  # b1
            (hidden, out),  # W2
            (out,),  # b2
        ]
        sizes = [int(np.prod(shape)) for shape in self._shapes]
        # Small random init keeps the first guess near "all levers mid-range".
        self.theta = rng.standard_normal(sum(sizes)) * 0.1

    def _unpack(self, theta: np.ndarray):
        params = []
        offset = 0
        for shape in self._shapes:
            size = int(np.prod(shape))
            params.append(theta[offset : offset + size].reshape(shape))
            offset += size
        return params  # [W1, b1, W2, b2]

    def set_theta(self, theta: np.ndarray) -> None:
        self.theta = np.asarray(theta, dtype=float)

    # -- features ------------------------------------------------------------ #
    def _national_means(self, state: pd.DataFrame) -> Dict[str, float]:
        means: Dict[str, float] = {}
        for key in self.feature_keys:
            if key in state.columns:
                series = pd.to_numeric(state[key], errors="coerce").dropna()
                means[key] = float(series.mean()) if not series.empty else 0.0
        return means

    def _feature_matrix(self, state: pd.DataFrame, step: int, env: "RolloutEnv") -> np.ndarray:
        """Per-province feature matrix, shape ``(n_provinces, n_in)``.

        Each province's own indicator values are normalised against the common
        national reference means, so provinces in different states map to
        different feature vectors (and therefore different lever decisions).
        """
        records: Dict[str, dict] = {}
        for record in state.to_dict("records"):
            code = str(record.get("area_code", "")).strip().upper()
            records[code] = record

        horizon = max(env.horizon, 1)
        matrix = np.empty((len(env.provinces), self._n_in), dtype=float)
        for index, code in enumerate(env.provinces):
            record = records.get(code, {})
            row = [1.0, step / horizon]
            for key in self.feature_keys:
                ref = self.ref_means.get(key, 0.0) or 1e-9
                try:
                    cur = float(record.get(key))
                except (TypeError, ValueError):
                    cur = ref
                if not np.isfinite(cur):
                    cur = ref
                row.append(float(np.clip(cur / ref - 1.0, -3.0, 3.0)))
            matrix[index] = row
        return matrix

    def prepare(self, env: "RolloutEnv") -> None:
        """Lock in feature reference means and initialise weights from the env."""
        if not self.feature_keys:
            self.feature_keys = [key for key in PREFERRED_FEATURE_KEYS if key in env.indicator_cols]
        self.ref_means = self._national_means(env.initial_state)
        self._mins = np.array([spec.min for spec in self.param_specs], dtype=float)
        self._spans = np.array(
            [max(spec.max - spec.min, 0.0) for spec in self.param_specs], dtype=float
        )
        rng = np.random.default_rng(self.seed)
        self._init_network(rng)

    def _allocations_from_values(
        self, values: np.ndarray, env: "RolloutEnv"
    ) -> Dict[str, Dict[str, float]]:
        """Turn a ``(n_provinces, n_levers)`` value matrix into allocations."""
        allocations: Dict[str, Dict[str, float]] = {}
        for index, code in enumerate(env.provinces):
            allocations[code] = {
                spec.key: float(values[index, col]) for col, spec in enumerate(self.param_specs)
            }
        return allocations

    # -- decision ------------------------------------------------------------ #
    def decide(self, state, step, env):
        if self.theta is None:
            self.prepare(env)
        features = self._feature_matrix(state, step, env)  # (P, n_in)
        W1, b1, W2, b2 = self._unpack(self.theta)
        hidden = np.tanh(features @ W1 + b1)  # (P, hidden)
        logits = hidden @ W2 + b2  # (P, n_out)
        activations = 1.0 / (1.0 + np.exp(-logits))  # (P, n_out) in [0, 1]
        values = self._mins + activations * self._spans  # (P, n_out)
        return self._allocations_from_values(values, env)

    # -- training ------------------------------------------------------------ #
    def fit(
        self,
        env: "RolloutEnv",
        *,
        iterations: int = 6,
        popsize: int = 6,
        sigma: float = 0.25,
        lr: float = 0.2,
        progress=None,
    ) -> dict:
        """Optimise the network weights with mirrored-sampling Evolution Strategies."""
        self.prepare(env)
        rng = np.random.default_rng(self.seed)
        dim = self.theta.size

        best_theta = self.theta.copy()
        best_score = self._evaluate(env, best_theta)
        start_score = best_score
        history = [best_score]  # best-so-far per iteration, index 0 = untrained
        if progress is not None:
            progress(0, iterations, best_score)

        for iteration in range(1, iterations + 1):
            # Snapshot the search centre: _evaluate mutates self.theta, so every
            # candidate must be perturbed from this fixed point, not the last one.
            center = self.theta.copy()
            eps = rng.standard_normal((popsize, dim))
            perturbations = np.vstack([eps, -eps])  # mirrored sampling
            scores = np.empty(perturbations.shape[0], dtype=float)
            for i, delta in enumerate(perturbations):
                scores[i] = self._evaluate(env, center + sigma * delta)

            top = int(np.argmax(scores))
            if scores[top] > best_score:
                best_score = float(scores[top])
                best_theta = (center + sigma * perturbations[top]).copy()

            std = scores.std()
            if std > 1e-8:
                advantages = (scores - scores.mean()) / std
                gradient = (perturbations.T @ advantages) / (perturbations.shape[0] * sigma)
                self.theta = center + lr * gradient
            else:
                self.theta = center

            history.append(best_score)
            if progress is not None:
                progress(iteration, iterations, best_score)

        self.theta = best_theta
        return _fit_report(
            best_score,
            start_score,
            iterations,
            feature_keys=list(self.feature_keys),
            history=history,
        )

    def _evaluate(self, env: "RolloutEnv", theta: np.ndarray) -> float:
        self.set_theta(theta)
        return env.score(self)

    # -- explanation ----------------------------------------------------------- #
    def explain(self, env: Optional["RolloutEnv"] = None) -> Optional[dict]:
        """Post-hoc audit: permutation importance + surrogate-tree distillation."""
        if env is None:
            return None
        from sera.twin.explain import explain_policy_posthoc

        if self.theta is None:
            self.prepare(env)
        return explain_policy_posthoc(self, env, seed=self.seed)


class LinearPolicy(NeuralPolicy):
    """A transparent linear policy: every lever is one row of signed weights.

    Same per-province setup as :class:`NeuralPolicy` but with no hidden layer:
    ``lever = sigmoid(features @ W)`` decoded into the lever bounds, where the
    features are the province's indicators normalised against the national
    reference means. Trained with the identical mirrored-sampling Evolution
    Strategies loop (inherited), so any performance gap versus the MLP is a
    measured *price of transparency*, not an artifact of a different
    optimiser. The weight matrix itself is the explanation: a positive weight
    means the lever rises when that indicator is above the national reference.
    """

    model_id = "linear"
    label = "Linear policy (auditable weights)"
    description = (
        "Each lever is a linear function of the province's indicators, "
        "trained with the same evolution strategy as the neural network. "
        "Fully auditable: every lever's response to every indicator is a "
        "single signed weight you can read and contest."
    )
    trainable = True
    explainability = "white-box"

    def __init__(
        self,
        param_specs: List[ParamSpec],
        feature_keys: Optional[List[str]] = None,
        seed: int = 0,
    ):
        super().__init__(param_specs, feature_keys=feature_keys, n_hidden=0, seed=seed)

    def _init_network(self, rng: np.random.Generator) -> None:
        self._n_in = 2 + len(self.feature_keys)  # bias + year-position + features
        self._shapes = [(self._n_in, self._n_out)]  # a single weight matrix
        self.theta = rng.standard_normal(self._n_in * self._n_out) * 0.1

    def decide(self, state, step, env):
        if self.theta is None:
            self.prepare(env)
        features = self._feature_matrix(state, step, env)  # (P, n_in)
        (weights,) = self._unpack(self.theta)
        activations = 1.0 / (1.0 + np.exp(-(features @ weights)))  # (P, n_out)
        values = self._mins + activations * self._spans
        return self._allocations_from_values(values, env)

    def explain(self, env: Optional["RolloutEnv"] = None) -> Optional[dict]:
        if self.theta is None:
            if env is None:
                return None
            self.prepare(env)
        (weights,) = self._unpack(self.theta)
        feature_names = ["bias", "year_position", *self.feature_keys]
        levers = []
        for col, spec in enumerate(self.param_specs):
            levers.append(
                {
                    "lever": spec.key,
                    "baseline": float(spec.baseline),
                    "min": float(spec.min),
                    "max": float(spec.max),
                    "weights": {
                        feature_names[row]: float(weights[row, col])
                        for row in range(len(feature_names))
                    },
                }
            )
        return {
            "type": "linear_weights",
            "features": feature_names,
            "levers": levers,
            "note": (
                "A positive weight raises the lever when the indicator is above "
                "the national reference; a negative weight lowers it. The bias "
                "column is the lever's base inclination."
            ),
        }


class DecisionListPolicy(NeuralPolicy):
    """One human-readable IF/THEN rule per lever, optimised with CEM.

    Each lever gets a single threshold rule on one indicator's deviation from
    the national reference: ``IF indicator > reference × (1 + τ) THEN lever =
    A ELSE lever = B``. The rule's indicator (via selection logits), threshold
    τ, and the two levels are all part of one genome searched with the
    Cross-Entropy Method. The trained policy can be printed as a short list of
    sentences a non-technical reader can audit — the most transparent
    per-province model in the registry.
    """

    model_id = "rules"
    label = "Rule list (IF/THEN per lever)"
    description = (
        "One IF/THEN rule per lever — thresholds on province indicators and "
        "two lever levels, found by cross-entropy search. The entire policy "
        "prints as a short list of sentences anyone can audit."
    )
    trainable = True
    explainability = "white-box"

    # Thresholds τ live in [-THRESHOLD_SPAN/2, +THRESHOLD_SPAN/2] as a relative
    # deviation from the national reference mean (±25% by default).
    THRESHOLD_SPAN = 0.5

    def __init__(
        self,
        param_specs: List[ParamSpec],
        feature_keys: Optional[List[str]] = None,
        seed: int = 0,
    ):
        super().__init__(param_specs, feature_keys=feature_keys, n_hidden=0, seed=seed)
        self.genome: Optional[np.ndarray] = None

    # Genome layout per lever: [feature-selection logits (n_features), threshold,
    # level-if-above, level-if-below], every entry in [0, 1].
    def _block_size(self) -> int:
        return len(self.feature_keys) + 3

    def prepare(self, env: "RolloutEnv") -> None:
        super().prepare(env)
        size = len(self.param_specs) * self._block_size()
        if self.genome is None or self.genome.size != size:
            self.genome = np.full(size, 0.5)

    def _rule_parts(self, col: int) -> tuple:
        """Decode lever ``col``'s genome block into (feature_idx, τ, hi, lo)."""
        n_features = len(self.feature_keys)
        block = self.genome[col * self._block_size() : (col + 1) * self._block_size()]
        threshold_u, level_above, level_below = block[n_features:]
        feature_idx = int(np.argmax(block[:n_features])) if n_features else -1
        tau = (float(threshold_u) - 0.5) * self.THRESHOLD_SPAN
        return feature_idx, tau, float(level_above), float(level_below)

    def decide(self, state, step, env):
        if self.genome is None:
            self.prepare(env)
        deviations = self._feature_matrix(state, step, env)[:, 2:]  # (P, n_features)
        values = np.empty((len(env.provinces), self._n_out), dtype=float)
        for col in range(self._n_out):
            feature_idx, tau, level_above, level_below = self._rule_parts(col)
            if feature_idx >= 0:
                fired = deviations[:, feature_idx] > tau
                normalized = np.where(fired, level_above, level_below)
            else:
                normalized = np.full(len(env.provinces), level_below)
            values[:, col] = self._mins[col] + np.clip(normalized, 0.0, 1.0) * self._spans[col]
        return self._allocations_from_values(values, env)

    def fit(
        self,
        env: "RolloutEnv",
        *,
        iterations: int = 6,
        popsize: int = 16,
        elite_frac: float = 0.25,
        smoothing: float = 0.5,
        progress=None,
    ) -> dict:
        """Cross-Entropy Method over the rule genome."""
        self.prepare(env)
        rng = np.random.default_rng(self.seed)
        result = cem_optimize(
            lambda genome: self._evaluate_genome(env, genome),
            self.genome.copy(),
            iterations=iterations,
            popsize=popsize,
            elite_frac=elite_frac,
            smoothing=smoothing,
            rng=rng,
            progress=progress,
        )
        self.genome = result["best_x"]
        return _fit_report(
            result["best_score"],
            result["start_score"],
            iterations,
            history=result.get("history"),
            rules=len(self.param_specs),
        )

    def _evaluate_genome(self, env: "RolloutEnv", genome: np.ndarray) -> float:
        self.genome = np.asarray(genome, dtype=float)
        return env.score(self)

    def explain(self, env: Optional["RolloutEnv"] = None) -> Optional[dict]:
        if self.genome is None:
            if env is None:
                return None
            self.prepare(env)
        rules = []
        for col, spec in enumerate(self.param_specs):
            feature_idx, tau, level_above, level_below = self._rule_parts(col)
            span = max(spec.max - spec.min, 0.0)
            decoded_above = spec.min + np.clip(level_above, 0.0, 1.0) * span
            decoded_below = spec.min + np.clip(level_below, 0.0, 1.0) * span
            feature = self.feature_keys[feature_idx] if feature_idx >= 0 else None
            reference = float(self.ref_means.get(feature, 0.0)) if feature else 0.0
            rules.append(
                {
                    "lever": spec.key,
                    "baseline": float(spec.baseline),
                    "feature": feature,
                    "threshold_pct": float(tau * 100.0),
                    "threshold_value": float(reference * (1.0 + tau)) if reference else None,
                    "value_if_above": float(decoded_above),
                    "value_if_below": float(decoded_below),
                }
            )
        return {
            "type": "decision_rules",
            "rules": rules,
            "note": (
                "Each lever follows one rule: provinces with the indicator above "
                "the threshold get the first level, all others the second. "
                "Thresholds are relative to the national reference mean at the "
                "start of the run."
            ),
        }


class UniformLeverPolicy(PolicyModel):
    """One shared national lever vector, found with the Cross-Entropy Method.

    The structural opposite of :class:`NeuralPolicy`: every province receives
    the *same* lever values, so the search space is just one point in
    ``[0, 1]^n_levers`` decoded into the lever bounds. CEM keeps a Gaussian
    over that space, samples a population per iteration, and refits the
    Gaussian to the elite candidates under the env's ethical objective. Like
    the neural policy it is objective-agnostic — the framework being maximised
    comes entirely from the rollout environment.
    """

    model_id = "uniform_cem"
    label = "Uniform national policy (CEM)"
    description = (
        "Cross-entropy search for a single lever vector applied identically "
        "to every province — no regional tailoring, optimised for the "
        "selected ethical objective. The result is one readable lever table."
    )
    trainable = True
    explainability = "white-box"

    def __init__(self, param_specs: List[ParamSpec], seed: int = 0):
        super().__init__(param_specs)
        self.seed = int(seed)
        self.x: Optional[np.ndarray] = None  # shared levers in [0, 1]^n
        self._mins: Optional[np.ndarray] = None
        self._spans: Optional[np.ndarray] = None

    def prepare(self, env: "RolloutEnv") -> None:
        self._mins = np.array([spec.min for spec in self.param_specs], dtype=float)
        self._spans = np.array(
            [max(spec.max - spec.min, 0.0) for spec in self.param_specs], dtype=float
        )
        if self.x is None:
            self.x = np.full(len(self.param_specs), 0.5)

    def decide(self, state, step, env):
        if self.x is None:
            self.prepare(env)
        values = self._mins + np.clip(self.x, 0.0, 1.0) * self._spans
        shared = {spec.key: float(values[col]) for col, spec in enumerate(self.param_specs)}
        return {code: dict(shared) for code in env.provinces}

    def fit(
        self,
        env: "RolloutEnv",
        *,
        iterations: int = 6,
        popsize: int = 12,
        elite_frac: float = 0.3,
        smoothing: float = 0.5,
        progress=None,
    ) -> dict:
        """Cross-Entropy Method over the shared lever vector."""
        self.prepare(env)
        rng = np.random.default_rng(self.seed)
        result = cem_optimize(
            lambda x: self._evaluate(env, x),
            self.x.copy(),
            iterations=iterations,
            popsize=popsize,
            elite_frac=elite_frac,
            smoothing=smoothing,
            rng=rng,
            progress=progress,
        )
        self.x = result["best_x"]
        return _fit_report(
            result["best_score"],
            result["start_score"],
            iterations,
            history=result.get("history"),
        )

    def _evaluate(self, env: "RolloutEnv", x: np.ndarray) -> float:
        self.x = np.asarray(x, dtype=float)
        return env.score(self)

    def explain(self, env: Optional["RolloutEnv"] = None) -> Optional[dict]:
        if self.x is None:
            if env is None:
                return None
            self.prepare(env)
        values = self._mins + np.clip(self.x, 0.0, 1.0) * self._spans
        return {
            "type": "lever_table",
            "levers": [
                {
                    "lever": spec.key,
                    "value": float(values[col]),
                    "baseline": float(spec.baseline),
                    "min": float(spec.min),
                    "max": float(spec.max),
                }
                for col, spec in enumerate(self.param_specs)
            ],
            "note": "One national lever vector, applied identically to every province.",
        }


class BayesianUniformPolicy(UniformLeverPolicy):
    """The shared national lever vector, found with GP Bayesian optimisation.

    Every twin rollout is expensive (110 provinces × horizon years), so instead
    of CEM's population sampling this model fits a Gaussian-process surrogate
    of the objective over the lever box and picks each next rollout by Expected
    Improvement — typically an order of magnitude fewer rollouts. The fitted
    surrogate doubles as the explanation: per-lever partial-dependence curves
    with uncertainty, i.e. "how sure is the optimizer that raising this lever
    helps, under this ethical objective".
    """

    model_id = "uniform_bayes"
    label = "Uniform national policy (Bayesian opt.)"
    description = (
        "Gaussian-process Bayesian optimisation of a single national lever "
        "vector. Far fewer twin rollouts than evolutionary search, and the "
        "surrogate shows how the objective responds to each lever — with "
        "uncertainty, complementing the causal-rule sensitivity band."
    )
    trainable = True
    explainability = "gray-box"

    def __init__(self, param_specs: List[ParamSpec], seed: int = 0):
        super().__init__(param_specs, seed=seed)
        self._gp = None
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    def fit(
        self,
        env: "RolloutEnv",
        *,
        iterations: int = 6,
        n_init: int = 6,
        candidate_pool: int = 256,
        progress=None,
    ) -> dict:
        """Sequential Bayesian optimisation: GP surrogate + Expected Improvement."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern

        self.prepare(env)
        rng = np.random.default_rng(self.seed)
        dim = len(self.param_specs)

        # Initial design: the mid-range starting point plus random probes.
        X = [np.full(dim, 0.5)]
        X.extend(rng.uniform(0.0, 1.0, size=(max(1, n_init - 1), dim)))
        y = [self._evaluate(env, x) for x in X]
        start_score = float(y[0])
        best_idx = int(np.argmax(y))
        best_x, best_score = X[best_idx].copy(), float(y[best_idx])
        if progress is not None:
            progress(0, iterations, best_score)

        def make_gp():
            return GaussianProcessRegressor(
                kernel=Matern(length_scale=0.25, length_scale_bounds=(0.05, 2.0), nu=2.5),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=1,
                random_state=self.seed,
            )

        for iteration in range(1, iterations + 1):
            gp = make_gp()
            gp.fit(np.asarray(X), np.asarray(y))
            candidates = rng.uniform(0.0, 1.0, size=(candidate_pool, dim))
            local = np.clip(
                best_x + 0.1 * rng.standard_normal((candidate_pool // 4, dim)), 0.0, 1.0
            )
            candidates = np.vstack([candidates, local])
            mean, std = gp.predict(candidates, return_std=True)
            improvement = mean - best_score - 0.01 * max(float(np.std(y)), 1e-9)
            z = improvement / np.maximum(std, 1e-12)
            expected = improvement * _norm_cdf(z) + std * _norm_pdf(z)
            expected[std <= 1e-12] = 0.0

            next_x = candidates[int(np.argmax(expected))]
            score = self._evaluate(env, next_x)
            X.append(next_x.copy())
            y.append(score)
            if score > best_score:
                best_score = float(score)
                best_x = next_x.copy()
            if progress is not None:
                progress(iteration, iterations, best_score)

        self._X, self._y = np.asarray(X), np.asarray(y)
        self._gp = make_gp()
        self._gp.fit(self._X, self._y)
        self.x = best_x
        return _fit_report(best_score, start_score, iterations, evaluations=len(y))

    def explain(self, env: Optional["RolloutEnv"] = None) -> Optional[dict]:
        if self._gp is None or self.x is None:
            return super().explain(env)
        grid_points = 9
        grid_u = np.linspace(0.0, 1.0, grid_points)
        levers = []
        for col, spec in enumerate(self.param_specs):
            span = max(spec.max - spec.min, 0.0)
            probe = np.tile(self.x, (grid_points, 1))
            probe[:, col] = grid_u
            mean, std = self._gp.predict(probe, return_std=True)
            levers.append(
                {
                    "lever": spec.key,
                    "best_value": float(spec.min + np.clip(self.x[col], 0, 1) * span),
                    "baseline": float(spec.baseline),
                    "effect_range": float(mean.max() - mean.min()),
                    "mean_std": float(std.mean()),
                    "grid": [
                        {
                            "value": float(spec.min + u * span),
                            "mean": float(m),
                            "std": float(s),
                        }
                        for u, m, s in zip(grid_u, mean, std)
                    ],
                }
            )
        levers.sort(key=lambda item: -item["effect_range"])
        return {
            "type": "partial_dependence",
            "levers": levers,
            "evaluations": int(self._y.size),
            "note": (
                "GP-estimated objective response per lever (others held at the "
                "optimum). Effect range is the surrogate's estimate, not a "
                "measurement; the uncertainty column is the GP's own doubt."
            ),
        }


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    # erf-based standard normal CDF (avoids importing scipy at module level).
    from math import erf

    return np.array([0.5 * (1.0 + erf(float(v) / np.sqrt(2.0))) for v in np.atleast_1d(z)])


class ClusterLeverPolicy(PolicyModel):
    """Regional policy packages: k-means provinces, one lever vector per cluster.

    Sits exactly between :class:`UniformLeverPolicy` (one national vector) and
    :class:`NeuralPolicy` (110 implicit vectors): the provinces are grouped by
    their starting indicators into a few clusters — the way real governments
    write "plans for the industrial north" or "plans for the Mezzogiorno" —
    and CEM searches one shared lever vector per cluster. The explanation is
    the cluster membership plus one small lever table per cluster.
    """

    model_id = "cluster_cem"
    label = "Regional cluster policy (CEM)"
    description = (
        "Groups provinces into a few clusters by their indicators and finds "
        "one lever vector per cluster — regional policy packages, fully "
        "inspectable as a handful of lever tables."
    )
    trainable = True
    explainability = "white-box"

    def __init__(self, param_specs: List[ParamSpec], n_clusters: int = 4, seed: int = 0):
        super().__init__(param_specs)
        self.n_clusters = max(1, int(n_clusters))
        self.seed = int(seed)
        self.feature_keys: List[str] = []
        self.assignments: Dict[str, int] = {}
        self.cluster_profiles: List[Dict[str, float]] = []
        self.genome: Optional[np.ndarray] = None
        self._k = 1
        self._mins: Optional[np.ndarray] = None
        self._spans: Optional[np.ndarray] = None

    def prepare(self, env: "RolloutEnv") -> None:
        self._mins = np.array([spec.min for spec in self.param_specs], dtype=float)
        self._spans = np.array(
            [max(spec.max - spec.min, 0.0) for spec in self.param_specs], dtype=float
        )
        self.feature_keys = [
            key
            for key in PREFERRED_FEATURE_KEYS
            if key in env.indicator_cols and key in env.initial_state.columns
        ]
        records: Dict[str, dict] = {}
        for record in env.initial_state.to_dict("records"):
            code = str(record.get("area_code", "")).strip().upper()
            records[code] = record

        if self.feature_keys:
            matrix = np.zeros((len(env.provinces), len(self.feature_keys)), dtype=float)
            for col, key in enumerate(self.feature_keys):
                series = pd.to_numeric(env.initial_state[key], errors="coerce")
                fallback = float(series.mean()) if series.notna().any() else 0.0
                for row, code in enumerate(env.provinces):
                    try:
                        value = float(records.get(code, {}).get(key))
                    except (TypeError, ValueError):
                        value = fallback
                    matrix[row, col] = value if np.isfinite(value) else fallback
                # z-score per feature so no indicator dominates the distance.
                std = matrix[:, col].std()
                matrix[:, col] = (matrix[:, col] - matrix[:, col].mean()) / (std or 1.0)
            rng = np.random.default_rng(self.seed)
            labels = _kmeans(matrix, self.n_clusters, rng)
        else:
            labels = np.zeros(len(env.provinces), dtype=int)

        self._k = int(labels.max()) + 1 if labels.size else 1
        self.assignments = {code: int(labels[row]) for row, code in enumerate(env.provinces)}

        self.cluster_profiles = []
        for cluster in range(self._k):
            members = [code for code in env.provinces if self.assignments[code] == cluster]
            profile: Dict[str, float] = {}
            for key in self.feature_keys:
                values = []
                for code in members:
                    try:
                        value = float(records.get(code, {}).get(key))
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value):
                        values.append(value)
                if values:
                    profile[key] = float(np.mean(values))
            self.cluster_profiles.append(profile)

        size = self._k * len(self.param_specs)
        if self.genome is None or self.genome.size != size:
            self.genome = np.full(size, 0.5)

    def decide(self, state, step, env):
        if self.genome is None:
            self.prepare(env)
        levers = self.genome.reshape(self._k, len(self.param_specs))
        values = self._mins + np.clip(levers, 0.0, 1.0) * self._spans  # (k, n_levers)
        allocations: Dict[str, Dict[str, float]] = {}
        for code in env.provinces:
            cluster = self.assignments.get(code, 0)
            allocations[code] = {
                spec.key: float(values[cluster, col]) for col, spec in enumerate(self.param_specs)
            }
        return allocations

    def fit(
        self,
        env: "RolloutEnv",
        *,
        iterations: int = 6,
        popsize: int = 14,
        elite_frac: float = 0.3,
        smoothing: float = 0.5,
        progress=None,
    ) -> dict:
        """Cross-Entropy Method over the per-cluster lever vectors."""
        self.prepare(env)
        rng = np.random.default_rng(self.seed)
        result = cem_optimize(
            lambda genome: self._evaluate_genome(env, genome),
            self.genome.copy(),
            iterations=iterations,
            popsize=popsize,
            elite_frac=elite_frac,
            smoothing=smoothing,
            rng=rng,
            progress=progress,
        )
        self.genome = result["best_x"]
        return _fit_report(
            result["best_score"],
            result["start_score"],
            iterations,
            history=result.get("history"),
            clusters=self._k,
        )

    def _evaluate_genome(self, env: "RolloutEnv", genome: np.ndarray) -> float:
        self.genome = np.asarray(genome, dtype=float)
        return env.score(self)

    def explain(self, env: Optional["RolloutEnv"] = None) -> Optional[dict]:
        if self.genome is None:
            if env is None:
                return None
            self.prepare(env)
        levers = self.genome.reshape(self._k, len(self.param_specs))
        values = self._mins + np.clip(levers, 0.0, 1.0) * self._spans
        provinces_by_cluster: Dict[int, List[str]] = {}
        for code, cluster in self.assignments.items():
            provinces_by_cluster.setdefault(cluster, []).append(code)
        clusters = []
        for cluster in range(self._k):
            clusters.append(
                {
                    "id": cluster,
                    "provinces": sorted(provinces_by_cluster.get(cluster, [])),
                    "profile": {
                        key: float(value)
                        for key, value in (
                            self.cluster_profiles[cluster]
                            if cluster < len(self.cluster_profiles)
                            else {}
                        ).items()
                    },
                    "levers": [
                        {
                            "lever": spec.key,
                            "value": float(values[cluster, col]),
                            "baseline": float(spec.baseline),
                            "min": float(spec.min),
                            "max": float(spec.max),
                        }
                        for col, spec in enumerate(self.param_specs)
                    ],
                }
            )
        return {
            "type": "cluster_policy",
            "features": list(self.feature_keys),
            "clusters": clusters,
            "note": (
                "Provinces are clustered on their starting indicators; every "
                "province in a cluster receives the same lever package."
            ),
        }


class BlendedPolicy(PolicyModel):
    """Scale another policy's intervention toward the historical baseline.

    ``blend=1.0`` reproduces the inner policy exactly; ``blend=0.5`` moves every
    lever halfway back toward its baseline value; ``blend=0.0`` is the baseline
    itself. Used to present graded intervention candidates (full / moderate /
    none) to a human decision-maker instead of a single "optimal" answer.
    """

    model_id = "blended"
    label = "Blended intervention"
    description = "An existing policy's levers, scaled toward the historical baseline."
    trainable = False

    def __init__(self, inner: PolicyModel, blend: float):
        super().__init__(inner.param_specs)
        self.inner = inner
        self.blend = float(min(max(blend, 0.0), 1.0))

    def decide(self, state, step, env):
        inner_allocations = self.inner.decide(state, step, env)
        baselines = {spec.key: spec.baseline for spec in self.param_specs}
        blended: Dict[str, Dict[str, float]] = {}
        for code, levers in inner_allocations.items():
            blended[code] = {
                key: baselines.get(key, value) + self.blend * (value - baselines.get(key, value))
                for key, value in levers.items()
            }
        return blended


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_POLICY_CLASSES = {
    BaselinePolicy.model_id: BaselinePolicy,
    NeuralPolicy.model_id: NeuralPolicy,
    LinearPolicy.model_id: LinearPolicy,
    DecisionListPolicy.model_id: DecisionListPolicy,
    ClusterLeverPolicy.model_id: ClusterLeverPolicy,
    UniformLeverPolicy.model_id: UniformLeverPolicy,
    BayesianUniformPolicy.model_id: BayesianUniformPolicy,
}

# Old ids kept working so saved UI state / scripts don't break.
_MODEL_ALIASES = {"gdp_nn": NeuralPolicy.model_id}

# Backward-compatible name for the pre-objectives neural policy.
GdpMaximizerPolicy = NeuralPolicy


def available_models() -> List[dict]:
    """Metadata for every selectable policy model (for the UI dropdown)."""
    return [
        {
            "id": cls.model_id,
            "label": cls.label,
            "description": cls.description,
            "trainable": cls.trainable,
            "explainability": cls.explainability,
        }
        for cls in _POLICY_CLASSES.values()
    ]


def build_policy(model_id: str, param_specs: List[ParamSpec], seed: int = 0) -> PolicyModel:
    """Instantiate a policy by id, defaulting to the neural policy.

    ``seed`` controls the gradient-free optimizer's RNG; passing different
    seeds gives the multi-seed replication the report uses to separate genuine
    framework effects from search luck. Policies that take no seed (the
    baseline) ignore it.
    """
    resolved = _MODEL_ALIASES.get(model_id, model_id)
    cls = _POLICY_CLASSES.get(resolved, NeuralPolicy)
    try:
        return cls(param_specs, seed=int(seed))
    except TypeError:
        return cls(param_specs)
