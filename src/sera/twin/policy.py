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

Two trainable models ship with the twin, both objective-agnostic and
gradient-free (the twin is a non-differentiable black box of sklearn models
plus rules):

- :class:`NeuralPolicy` — a tiny pure-NumPy MLP applied per province (each
  province gets levers tailored to its own state), trained with
  mirrored-sampling Evolution Strategies.
- :class:`UniformLeverPolicy` — one shared national lever vector applied to
  every province, found with the Cross-Entropy Method.
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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


class BaselinePolicy(PolicyModel):
    """Holds every lever at its historical baseline (reference scenario)."""

    model_id = "baseline"
    label = "Baseline (historical levers)"
    description = "Keeps every policy lever at its historical baseline value."
    trainable = False

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
        "maximise the selected ethical objective over the horizon."
    )
    trainable = True

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
            self.feature_keys = [
                key for key in PREFERRED_FEATURE_KEYS if key in env.indicator_cols
            ]
        self.ref_means = self._national_means(env.initial_state)
        self._mins = np.array([spec.min for spec in self.param_specs], dtype=float)
        self._spans = np.array(
            [max(spec.max - spec.min, 0.0) for spec in self.param_specs], dtype=float
        )
        rng = np.random.default_rng(self.seed)
        self._init_network(rng)

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

        allocations: Dict[str, Dict[str, float]] = {}
        for index, code in enumerate(env.provinces):
            allocations[code] = {
                spec.key: float(values[index, col])
                for col, spec in enumerate(self.param_specs)
            }
        return allocations

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

            if progress is not None:
                progress(iteration, iterations, best_score)

        self.theta = best_theta
        return {
            "best_score": float(best_score),
            "start_score": float(start_score),
            "improvement_pct": float(
                100.0 * (best_score - start_score) / abs(start_score)
            )
            if start_score
            else 0.0,
            "iterations": int(iterations),
            "feature_keys": list(self.feature_keys),
        }

    def _evaluate(self, env: "RolloutEnv", theta: np.ndarray) -> float:
        self.set_theta(theta)
        return env.score(self)


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
        "selected ethical objective."
    )
    trainable = True

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
        shared = {
            spec.key: float(values[col]) for col, spec in enumerate(self.param_specs)
        }
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
        dim = len(self.param_specs)
        n_elite = max(1, int(round(popsize * elite_frac)))

        mu = self.x.copy()
        sigma = np.full(dim, 0.25)
        best_x = self.x.copy()
        best_score = self._evaluate(env, best_x)
        start_score = best_score
        if progress is not None:
            progress(0, iterations, best_score)

        for iteration in range(1, iterations + 1):
            candidates = np.clip(
                mu + sigma * rng.standard_normal((popsize, dim)), 0.0, 1.0
            )
            scores = np.array([self._evaluate(env, c) for c in candidates])

            elite_idx = np.argsort(scores)[-n_elite:]
            elites = candidates[elite_idx]
            mu = smoothing * elites.mean(axis=0) + (1.0 - smoothing) * mu
            sigma = smoothing * elites.std(axis=0) + (1.0 - smoothing) * sigma
            sigma = np.maximum(sigma, 0.02)  # keep exploring

            top = int(elite_idx[-1])
            if scores[top] > best_score:
                best_score = float(scores[top])
                best_x = candidates[top].copy()

            if progress is not None:
                progress(iteration, iterations, best_score)

        self.x = best_x
        return {
            "best_score": float(best_score),
            "start_score": float(start_score),
            "improvement_pct": float(
                100.0 * (best_score - start_score) / abs(start_score)
            )
            if start_score
            else 0.0,
            "iterations": int(iterations),
        }

    def _evaluate(self, env: "RolloutEnv", x: np.ndarray) -> float:
        self.x = np.asarray(x, dtype=float)
        return env.score(self)


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
                key: baselines.get(key, value)
                + self.blend * (value - baselines.get(key, value))
                for key, value in levers.items()
            }
        return blended


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_POLICY_CLASSES = {
    BaselinePolicy.model_id: BaselinePolicy,
    NeuralPolicy.model_id: NeuralPolicy,
    UniformLeverPolicy.model_id: UniformLeverPolicy,
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
        }
        for cls in _POLICY_CLASSES.values()
    ]


def build_policy(model_id: str, param_specs: List[ParamSpec]) -> PolicyModel:
    """Instantiate a policy by id, defaulting to the neural policy."""
    resolved = _MODEL_ALIASES.get(model_id, model_id)
    cls = _POLICY_CLASSES.get(resolved, NeuralPolicy)
    return cls(param_specs)
