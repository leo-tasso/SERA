"""Tests for the national budget constraint in sera.twin.budget (package-level).

The bridge has its own thin-wrapper tests; these exercise the pure functions
directly, without importing the UI layer.
"""

import pandas as pd
import pytest

from sera.twin.budget import (
    SPENDING_PARAMS,
    TAX_PARAMS,
    apply_budget_constraint,
    budget_usage,
    make_constraint_fn,
)

INTENSITY = 19.0
# Two spending levers and two tax levers with a baseline of 100 each is enough
# to exercise the ratio arithmetic without depending on real lever metadata.
BASELINES = {
    "healthcare_spending_allocation": 100.0,
    "education_spending_allocation": 100.0,
    "income_tax_rate": 100.0,
    "corporate_tax_rate": 100.0,
}


def make_state():
    return pd.DataFrame(
        [
            {"area_code": "MI", "year": 2025, "gdp_per_capita": 300.0},
            {"area_code": "RC", "year": 2025, "gdp_per_capita": 100.0},
        ]
    )


def spending_scaled(factor):
    keys = [k for k in BASELINES if k in SPENDING_PARAMS]
    return {code: {k: 100.0 * factor for k in keys} for code in ("MI", "RC")}


def tax_scaled(factor):
    keys = [k for k in BASELINES if k in TAX_PARAMS]
    return {code: {k: 100.0 * factor for k in keys} for code in ("MI", "RC")}


def test_baseline_pool_is_gdp_share():
    _used, pool = budget_usage({}, make_state(), INTENSITY, BASELINES)
    assert pool == pytest.approx((300 + 100) * INTENSITY / 100)


def test_baseline_is_fully_utilised():
    used, pool = budget_usage({}, make_state(), INTENSITY, BASELINES)
    assert used == pytest.approx(pool)


def test_tax_cut_shrinks_pool():
    _used, base = budget_usage({}, make_state(), INTENSITY, BASELINES)
    _used, pool = budget_usage(tax_scaled(0.5), make_state(), INTENSITY, BASELINES)
    assert pool == pytest.approx(base * 0.5)


def test_overspending_is_scaled_to_pool():
    scaled = apply_budget_constraint(spending_scaled(2.0), make_state(), INTENSITY, BASELINES)
    used, pool = budget_usage(scaled, make_state(), INTENSITY, BASELINES)
    assert used == pytest.approx(pool)


def test_reserve_carries_over_when_underspending():
    fn = make_constraint_fn(INTENSITY, BASELINES)
    _scaled, reserve = fn(spending_scaled(0.5), make_state(), 0.0)
    _used, pool = budget_usage({}, make_state(), INTENSITY, BASELINES)
    assert reserve == pytest.approx(pool * 0.5)


def test_reserve_extends_pool_then_empties():
    fn = make_constraint_fn(INTENSITY, BASELINES)
    _used, pool = budget_usage({}, make_state(), INTENSITY, BASELINES)
    _scaled, reserve = fn(spending_scaled(2.0), make_state(), pool)
    assert reserve == 0.0
