"""Pluggable ethical objectives for the policy optimizer.

Each :class:`Objective` turns one simulated year's provincial state into a
single welfare number; the rollout environment sums it over the horizon and
the policy models maximise that sum. Swapping the objective therefore swaps
the *ethical framework* the optimizer embodies, while the policy model (the
"how do we search" part) stays the same:

- ``utilitarian``  — total national GDP, the classic sum-of-outcomes view.
- ``rawlsian``     — maximin: only the worst-off province counts.
- ``egalitarian``  — Sen welfare: total GDP discounted by inter-provincial
  inequality (Gini).
- ``wellbeing``    — a multi-indicator composite (GDP, life expectancy,
  unemployment, poverty) measured as relative change from the starting state.

All objectives are "more is better"; scales differ between objectives (the
optimizers only compare candidates under one objective at a time).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd

GDP_KEY = "gdp_per_capita"


def gini(values: np.ndarray) -> float:
    """Gini coefficient in [0, 1] via the relative mean absolute difference."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    array = np.clip(array, 0.0, None)
    if array.size == 0:
        return 0.0
    total = array.sum()
    if total <= 0:
        return 0.0
    array = np.sort(array)
    n = array.size
    ranks = np.arange(1, n + 1)
    return float((2.0 * np.sum(ranks * array)) / (n * total) - (n + 1.0) / n)


def _gdp_values(state: pd.DataFrame) -> np.ndarray:
    if GDP_KEY not in state.columns:
        return np.array([], dtype=float)
    series = pd.to_numeric(state[GDP_KEY], errors="coerce").dropna()
    return series.to_numpy(dtype=float)


class Objective(ABC):
    """One ethical framework, expressed as a per-year welfare function."""

    objective_id: str = "base"
    label: str = "Base objective"
    description: str = ""

    def prepare(self, initial_state: pd.DataFrame, indicator_cols: List[str]) -> None:
        """Optional hook to lock in reference values from the starting state."""

    @abstractmethod
    def score_year(self, state: pd.DataFrame) -> float:
        """Welfare of one simulated year's provincial state (higher is better)."""


class UtilitarianGdpObjective(Objective):
    """Maximise the sum of provincial GDP per capita (the historical default)."""

    objective_id = "utilitarian"
    label = "Utilitarian (total GDP)"
    description = (
        "Maximise total national GDP per capita summed across provinces. "
        "Indifferent to how gains are distributed: a euro in Milan counts "
        "the same as a euro in Crotone."
    )

    def score_year(self, state: pd.DataFrame) -> float:
        values = _gdp_values(state)
        return float(values.sum()) if values.size else 0.0


class RawlsianObjective(Objective):
    """Maximin: judge each year by its worst-off province alone."""

    objective_id = "rawlsian"
    label = "Rawlsian (worst-off province)"
    description = (
        "Maximise the GDP per capita of the worst-off province (Rawls' "
        "difference principle). Scored as the national total Italy would "
        "have if every province lived like its poorest one, so growth "
        "anywhere else counts for nothing."
    )

    def score_year(self, state: pd.DataFrame) -> float:
        values = _gdp_values(state)
        if values.size == 0:
            return 0.0
        return float(values.min()) * values.size


class EgalitarianObjective(Objective):
    """Sen welfare: total GDP discounted by inter-provincial inequality."""

    objective_id = "egalitarian"
    label = "Egalitarian (GDP × equality)"
    description = (
        "Maximise Sen's welfare function: total national GDP multiplied by "
        "(1 − Gini) across provinces. Growth still counts, but growth that "
        "widens the North–South divide is penalised."
    )

    def score_year(self, state: pd.DataFrame) -> float:
        values = _gdp_values(state)
        if values.size == 0:
            return 0.0
        return float(values.sum()) * (1.0 - gini(values))


class CVaRObjective(Objective):
    """Smoothed maximin: the mean of the worst ``alpha`` fraction of provinces.

    Hard maximin (:class:`RawlsianObjective`) scores a single province out of
    110, which is the sparsest training signal in the registry and the likely
    reason it under-optimizes its own floor under a small search budget. This
    Conditional-Value-at-Risk objective keeps the Rawlsian spirit --- it cares
    only about the bottom of the distribution --- but averages the worst
    ``alpha`` fraction (default 20%, i.e. ~22 provinces), giving the optimizer
    a far denser gradient. As ``alpha`` -> 0 it recovers pure maximin; at
    ``alpha`` = 1 it is the (rescaled) utilitarian mean. Comparing it against
    hard maximin disentangles "the theory is destructive" from "the objective
    was hard to optimize".
    """

    objective_id = "cvar"
    label = "Rawlsian (smoothed, worst fraction)"
    description = (
        "Maximise the average GDP per capita of the worst-off fraction of "
        "provinces (Conditional Value at Risk). A denser-signal cousin of "
        "strict maximin: it still protects the bottom of the distribution but "
        "averages the worst ~20% instead of the single poorest province."
    )

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = float(min(max(alpha, 0.01), 1.0))

    def score_year(self, state: pd.DataFrame) -> float:
        values = _gdp_values(state)
        if values.size == 0:
            return 0.0
        k = max(1, int(math.ceil(self.alpha * values.size)))
        worst_k = np.sort(values)[:k]
        # Rescale by n (as Rawlsian does) so the score is comparable in
        # magnitude to the national total in the UI's charts.
        return float(worst_k.mean()) * values.size


class PrioritarianObjective(Objective):
    """Prioritarianism: sum of a strictly concave transform of provincial GDP.

    Benefits matter more the worse off their recipient is (Parfit). The
    concavity parameter ``rho`` interpolates the continuum between the
    utilitarian and Rawlsian endpoints: ``rho`` = 0 is exactly utilitarian
    (linear), and as ``rho`` -> 1 the transform approaches the logarithm and
    the optimizer's behaviour slides toward maximin --- without Rawls'
    exclusive focus on the minimum and without egalitarianism's concern for
    relative gaps (it weighs levels, not gaps, so it evades the leveling-down
    objection).
    """

    objective_id = "prioritarian"
    label = "Prioritarian (priority to the worse-off)"
    description = (
        "Maximise the sum of a concave transform of provincial GDP per capita: "
        "a euro is worth more to a poorer province than to a richer one. The "
        "concavity slider tunes how much priority the worst-off get, sweeping "
        "from utilitarian (no priority) toward Rawlsian (maximal priority)."
    )

    def __init__(self, rho: float = 0.5) -> None:
        self.rho = float(min(max(rho, 0.0), 0.99))

    def score_year(self, state: pd.DataFrame) -> float:
        values = _gdp_values(state)
        if values.size == 0:
            return 0.0
        values = np.clip(values, 0.0, None)
        if self.rho <= 0.0:
            return float(values.sum())
        # CRRA-style increasing concave utility u(y) = y^(1 - rho).
        return float(np.power(values, 1.0 - self.rho).sum())


class SufficientarianObjective(Objective):
    """Sufficientarianism: minimise the total shortfall below a threshold.

    What matters morally is that each province has *enough*: inequalities above
    the sufficiency threshold carry no weight (Frankfurt). The score is the
    negative total shortfall below a threshold ``theta``, so it is maximised
    (toward zero) by pulling provinces up to the line and is indifferent to
    everything above it. Because "the threshold is the entire theory" and there
    is no non-arbitrary euro value for "a sufficient province", ``theta`` is set
    *relative to the starting distribution* (a fraction of the initial median
    provincial GDP) and exposed as a user-facing parameter rather than a
    hard-coded constant.
    """

    objective_id = "sufficientarian"
    label = "Sufficientarian (everyone above a floor)"
    description = (
        "Minimise the total shortfall of provinces below a sufficiency "
        "threshold (a fraction of the starting median province). Inequality "
        "above the threshold is morally irrelevant; only the gap of the "
        "below-threshold provinces counts. The threshold is yours to set."
    )

    def __init__(self, threshold_ratio: float = 0.8) -> None:
        self.threshold_ratio = float(min(max(threshold_ratio, 0.0), 2.0))
        self.threshold: float = 0.0

    def prepare(self, initial_state: pd.DataFrame, indicator_cols: List[str]) -> None:
        values = _gdp_values(initial_state)
        median = float(np.median(values)) if values.size else 0.0
        self.threshold = self.threshold_ratio * median

    def score_year(self, state: pd.DataFrame) -> float:
        values = _gdp_values(state)
        if values.size == 0:
            return 0.0
        if self.threshold <= 0.0:
            self.prepare(state, [])
        shortfall = np.clip(self.threshold - values, 0.0, None).sum()
        return -float(shortfall)


class WellbeingObjective(Objective):
    """Multi-indicator composite: GDP, health, work, and poverty together."""

    objective_id = "wellbeing"
    label = "Multi-objective wellbeing"
    description = (
        "Maximise a composite of GDP per capita, life expectancy, "
        "unemployment, and poverty, each measured as percent change from "
        "the starting year. GDP is only part of a good life here."
    )

    # indicator -> (weight, direction); direction -1 means lower is better.
    COMPONENTS: Dict[str, tuple] = {
        "gdp_per_capita": (0.35, 1),
        "life_expectancy": (0.25, 1),
        "unemployment_rate": (0.20, -1),
        "poverty_rate": (0.20, -1),
    }

    def __init__(self) -> None:
        self.ref_means: Dict[str, float] = {}

    def prepare(self, initial_state: pd.DataFrame, indicator_cols: List[str]) -> None:
        self.ref_means = {}
        for key in self.COMPONENTS:
            if key in initial_state.columns:
                series = pd.to_numeric(initial_state[key], errors="coerce").dropna()
                mean = float(series.mean()) if not series.empty else 0.0
                if mean != 0.0:
                    self.ref_means[key] = mean

    def score_year(self, state: pd.DataFrame) -> float:
        if not self.ref_means:
            self.prepare(state, [])
        total_weight = sum(
            self.COMPONENTS[key][0] for key in self.ref_means if key in state.columns
        )
        if total_weight <= 0:
            return 0.0
        score = 0.0
        for key, ref in self.ref_means.items():
            if key not in state.columns:
                continue
            series = pd.to_numeric(state[key], errors="coerce").dropna()
            if series.empty:
                continue
            weight, direction = self.COMPONENTS[key]
            relative_change = float(series.mean()) / ref - 1.0
            score += (weight / total_weight) * direction * relative_change
        return score * 100.0  # composite improvement in percent points


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_OBJECTIVE_CLASSES = {
    UtilitarianGdpObjective.objective_id: UtilitarianGdpObjective,
    RawlsianObjective.objective_id: RawlsianObjective,
    CVaRObjective.objective_id: CVaRObjective,
    PrioritarianObjective.objective_id: PrioritarianObjective,
    EgalitarianObjective.objective_id: EgalitarianObjective,
    SufficientarianObjective.objective_id: SufficientarianObjective,
    WellbeingObjective.objective_id: WellbeingObjective,
}

# Tunable parameters per objective, so the UI can render a slider next to the
# dropdown and pass the chosen value to ``build_objective``. Each entry is the
# constructor keyword plus its range; absent ids have no parameters.
_OBJECTIVE_PARAMETERS: Dict[str, List[dict]] = {
    CVaRObjective.objective_id: [
        {
            "id": "alpha",
            "label": "Worst-off fraction",
            "min": 0.05,
            "max": 1.0,
            "default": 0.2,
            "step": 0.05,
        }
    ],
    PrioritarianObjective.objective_id: [
        {
            "id": "rho",
            "label": "Concavity (priority to the worse-off)",
            "min": 0.0,
            "max": 0.99,
            "default": 0.5,
            "step": 0.05,
        }
    ],
    SufficientarianObjective.objective_id: [
        {
            "id": "threshold_ratio",
            "label": "Threshold (× starting median province)",
            "min": 0.3,
            "max": 1.5,
            "default": 0.8,
            "step": 0.05,
        }
    ],
}

DEFAULT_OBJECTIVE_ID = UtilitarianGdpObjective.objective_id


def objective_parameters(objective_id: str) -> List[dict]:
    """Tunable-parameter metadata for one objective (empty if it has none)."""
    return list(_OBJECTIVE_PARAMETERS.get(objective_id, []))


def available_objectives() -> List[dict]:
    """Metadata for every selectable ethical objective (for the UI dropdown)."""
    return [
        {
            "id": cls.objective_id,
            "label": cls.label,
            "description": cls.description,
            "parameters": objective_parameters(cls.objective_id),
        }
        for cls in _OBJECTIVE_CLASSES.values()
    ]


def build_objective(objective_id: str, **params) -> Objective:
    """Instantiate an objective by id, defaulting to the utilitarian one.

    Extra keyword arguments configure tunable objectives (e.g. ``alpha`` for
    CVaR, ``rho`` for prioritarian, ``threshold_ratio`` for sufficientarian);
    unknown keys for a given objective are ignored, so the UI can pass whatever
    slider value it has without knowing which objective consumes it.
    """
    cls = _OBJECTIVE_CLASSES.get(objective_id, UtilitarianGdpObjective)
    allowed = {p["id"] for p in _OBJECTIVE_PARAMETERS.get(objective_id, [])}
    kwargs = {key: value for key, value in params.items() if key in allowed}
    return cls(**kwargs)
