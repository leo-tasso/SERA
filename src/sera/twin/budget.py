"""National budget constraint for the policy rollout.

The budget is the ethically load-bearing part of the simulation: distributive
justice is only a live question under scarcity, and this is what makes the
simulated scarcity binding. It used to live in ``ui/backend_bridge.py``, which
meant the headless experiments had to import the UI layer to run; it now lives
in the twin package so it is unit-testable and reusable without Electron.

Model (mirrored by the frontend ``ResourceMeter`` in ``ui/renderer.js``):

* Each province has a spending *cost ratio* (average of its spending levers
  relative to baseline) and a *revenue ratio* (average of its tax levers
  relative to baseline). Both scale a fixed share of that province's GDP.
* The national ``base_pool`` is the sum of provincial revenues; the national
  cost is the sum of provincial spending. At historical baseline levers the two
  are equal (100% utilisation), so cutting taxes below baseline shrinks the pool
  and makes "stimulate with tax cuts *and* max spending everywhere" infeasible.
* Unspent budget carries over as a reserve that extends the pool next year.

The pure functions here take an explicit ``param_baselines`` map so they have no
dependency on how levers are configured; thin wrappers in the bridge supply the
baselines from ``parameter_limits``.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Tuple

import pandas as pd

GDP_KEY = "gdp_per_capita"

# Spending levers consume the national pool. Public-sector wages are
# compensation spending and belong here, not with the regulatory levers.
SPENDING_PARAMS = frozenset(
    {
        "healthcare_spending_allocation",
        "education_spending_allocation",
        "infrastructure_investment_allocation",
        "social_welfare_spending_allocation",
        "rd_innovation_incentives",
        "green_energy_environment_investment",
        "pension_retirement_spending",
        "agriculture_support_level",
        "manufacturing_incentives",
        "tourism_support_level",
        "small_business_support",
        "public_sector_wage_levels",
        "housing_urban_development_support",
    }
)

# Tax levers fund the national pool. Cutting them below baseline shrinks the
# budget available for the spending levers; raising them grows it (at the cost
# of their own causal drag on the economy).
TAX_PARAMS = frozenset(
    {
        "income_tax_rate",
        "corporate_tax_rate",
        "property_wealth_tax_rate",
        "vat_consumption_tax_rate",
    }
)

Allocations = Dict[str, Dict[str, float]]
ConstraintFn = Callable[[Allocations, pd.DataFrame, float], "Tuple[Allocations, float]"]


def _gdp_by_province(current_state: pd.DataFrame) -> Dict[str, float]:
    gdp: Dict[str, float] = {}
    for _, row in current_state.iterrows():
        code = str(row.get("area_code", "")).strip().upper()
        value = float(row.get(GDP_KEY, 0) or 0)
        if value > 0:
            gdp[code] = value
    return gdp


def budget_usage(
    allocations: Allocations,
    current_state: pd.DataFrame,
    spending_intensity_pct: float,
    param_baselines: Mapping[str, float],
) -> Tuple[float, float]:
    """Return ``(total_used, base_pool)`` for the national spending budget.

    ``total_used`` is the national spending cost implied by ``allocations``;
    ``base_pool`` is the national revenue they fund. At baseline they are equal.
    """
    spending_keys = [key for key in SPENDING_PARAMS if key in param_baselines]
    tax_keys = [key for key in TAX_PARAMS if key in param_baselines]
    if not spending_keys:
        return 0.0, 0.0

    safe_baselines = {
        key: max(float(param_baselines[key]), 1e-9) for key in spending_keys + tax_keys
    }

    base_pool = 0.0
    total_used = 0.0
    for code, gdp in _gdp_by_province(current_state).items():
        prov = allocations.get(code, {})
        revenue_ratio = (
            sum(
                float(prov.get(key, safe_baselines[key])) / safe_baselines[key]
                for key in tax_keys
            )
            / max(len(tax_keys), 1)
        )
        base_pool += revenue_ratio * gdp * spending_intensity_pct / 100.0

        cost_ratio = (
            sum(
                float(prov.get(key, safe_baselines[key])) / safe_baselines[key]
                for key in spending_keys
            )
            / len(spending_keys)
        )
        total_used += cost_ratio * gdp * spending_intensity_pct / 100.0

    return total_used, base_pool


def apply_budget_constraint(
    allocations: Allocations,
    current_state: pd.DataFrame,
    spending_intensity_pct: float,
    param_baselines: Mapping[str, float],
    reserve_pool: float = 0.0,
) -> Allocations:
    """Scale spending levers down if national cost exceeds pool + reserve."""
    spending_keys = [key for key in SPENDING_PARAMS if key in param_baselines]
    if not spending_keys:
        return allocations

    total_used, base_pool = budget_usage(
        allocations, current_state, spending_intensity_pct, param_baselines
    )
    total_pool = base_pool + reserve_pool
    if total_pool <= 0 or total_used <= total_pool:
        return allocations

    scale = total_pool / total_used
    scaled = {code: dict(params) for code, params in allocations.items()}
    for code in scaled:
        for key in spending_keys:
            if key in scaled[code]:
                scaled[code][key] = scaled[code][key] * scale
    return scaled


def make_constraint_fn(
    spending_intensity_pct: float, param_baselines: Mapping[str, float]
) -> ConstraintFn:
    """Build a horizon-aware budget constraint that tracks the reserve carry-over."""
    baselines = dict(param_baselines)

    def constraint_fn(allocations: Allocations, state: pd.DataFrame, reserve: float):
        scaled = apply_budget_constraint(
            allocations, state, spending_intensity_pct, baselines, reserve
        )
        total_used, base_pool = budget_usage(
            allocations, state, spending_intensity_pct, baselines
        )
        spent = min(total_used, base_pool + reserve)
        new_reserve = max(0.0, reserve + base_pool - spent)
        return scaled, new_reserve

    return constraint_fn
