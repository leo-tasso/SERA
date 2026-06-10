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
    EgalitarianObjective.objective_id: EgalitarianObjective,
    WellbeingObjective.objective_id: WellbeingObjective,
}

DEFAULT_OBJECTIVE_ID = UtilitarianGdpObjective.objective_id


def available_objectives() -> List[dict]:
    """Metadata for every selectable ethical objective (for the UI dropdown)."""
    return [
        {
            "id": cls.objective_id,
            "label": cls.label,
            "description": cls.description,
        }
        for cls in _OBJECTIVE_CLASSES.values()
    ]


def build_objective(objective_id: str) -> Objective:
    """Instantiate an objective by id, defaulting to the utilitarian one."""
    cls = _OBJECTIVE_CLASSES.get(objective_id, UtilitarianGdpObjective)
    return cls()
