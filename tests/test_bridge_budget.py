"""Tests for the UI bridge's national budget / reserve-pool accounting."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))

import backend_bridge  # noqa: E402
from backend_bridge import (  # noqa: E402
    SPENDING_PARAMS,
    _apply_budget_constraint,
    _budget_usage,
    _make_constraint_fn,
    build_parameters_frame,
    dataframe_records,
    format_label,
    parameter_limits,
)
from sera.twin.causal_graph import ANNUAL_PARAMETERS  # noqa: E402
from sera.twin.province_mapping import PROVINCE_SIGLAS_110  # noqa: E402

INTENSITY = 20.0  # % of GDP


def make_state():
    return pd.DataFrame(
        {
            "area_code": ["MI", "TO"],
            "year": [2025, 2025],
            "gdp_per_capita": [300.0, 100.0],
        }
    )


def allocations_scaled(factor):
    """Every spending lever at ``factor`` x its historical baseline."""
    levers = {key: parameter_limits(key)[0] * factor for key in SPENDING_PARAMS}
    return {"MI": dict(levers), "TO": dict(levers)}


class TestBudgetUsage:
    def test_baseline_allocations_use_exactly_the_pool(self):
        used, pool = _budget_usage({}, make_state(), INTENSITY)
        assert pool == pytest.approx((300 + 100) * INTENSITY / 100)
        assert used == pytest.approx(pool)

    def test_halved_spending_uses_half_the_pool(self):
        used, pool = _budget_usage(allocations_scaled(0.5), make_state(), INTENSITY)
        assert used == pytest.approx(pool * 0.5)

    def test_provinces_without_gdp_are_ignored(self):
        state = make_state()
        state.loc[1, "gdp_per_capita"] = 0.0
        used, pool = _budget_usage({}, state, INTENSITY)
        assert pool == pytest.approx(300 * INTENSITY / 100)
        assert used == pytest.approx(pool)


class TestBudgetConstraint:
    def test_within_budget_allocations_are_untouched(self):
        allocations = allocations_scaled(0.5)
        scaled = _apply_budget_constraint(allocations, make_state(), INTENSITY)
        assert scaled == allocations

    def test_over_budget_allocations_are_scaled_to_the_pool(self):
        scaled = _apply_budget_constraint(allocations_scaled(2.0), make_state(), INTENSITY)
        used, pool = _budget_usage(scaled, make_state(), INTENSITY)
        assert used == pytest.approx(pool)

    def test_reserve_extends_the_pool(self):
        _used, pool = _budget_usage({}, make_state(), INTENSITY)
        allocations = allocations_scaled(2.0)  # needs 2x pool
        scaled = _apply_budget_constraint(
            allocations, make_state(), INTENSITY, reserve_pool=pool
        )
        # base pool + an equal reserve covers the doubled spend exactly.
        assert scaled == allocations


class TestConstraintFn:
    def test_underspend_accrues_reserve(self):
        fn = _make_constraint_fn(INTENSITY)
        _scaled, reserve = fn(allocations_scaled(0.5), make_state(), 0.0)
        _used, pool = _budget_usage({}, make_state(), INTENSITY)
        assert reserve == pytest.approx(pool * 0.5)

    def test_overspend_without_reserve_leaves_none(self):
        fn = _make_constraint_fn(INTENSITY)
        _scaled, reserve = fn(allocations_scaled(2.0), make_state(), 0.0)
        assert reserve == 0.0

    def test_overspend_consumes_the_reserve(self):
        fn = _make_constraint_fn(INTENSITY)
        _used, pool = _budget_usage({}, make_state(), INTENSITY)
        _scaled, reserve = fn(allocations_scaled(2.0), make_state(), pool)
        assert reserve == 0.0

    def test_reserve_never_negative(self):
        fn = _make_constraint_fn(INTENSITY)
        _scaled, reserve = fn(allocations_scaled(10.0), make_state(), 1.0)
        assert reserve >= 0.0


class TestParametersFrame:
    def test_one_row_per_canonical_province(self):
        frame = build_parameters_frame(2026, {})
        assert list(frame["area_code"]) == PROVINCE_SIGLAS_110
        assert (frame["year"] == 2026).all()

    def test_values_clamped_and_defaulted(self):
        key = next(iter(ANNUAL_PARAMETERS))
        baseline, min_value, max_value, _step = parameter_limits(key)
        frame = build_parameters_frame(
            2026,
            {"MI": {key: 1e12}, "TO": {key: -1e12}, "NA": {key: "garbage"}},
        )
        by_code = frame.set_index("area_code")[key]
        assert by_code["MI"] == max_value
        assert by_code["TO"] == min_value
        assert by_code["NA"] == baseline
        assert by_code["RM"] == baseline  # untouched province


class TestBridgeHelpers:
    def test_format_label(self):
        assert format_label("gdp_per_capita") == "GDP Per Capita"
        assert format_label("rd_innovation_incentives") == "R&D Innovation Incentives"

    def test_dataframe_records_replaces_nan_with_none(self):
        frame = pd.DataFrame({"a": [1.0, float("nan")], "year": [2025, 2026]})
        records = dataframe_records(frame)
        assert records[0] == {"a": 1.0, "year": 2025}
        assert records[1]["a"] is None

    def test_emit_progress_is_prefix_framed_json(self, capsys):
        backend_bridge.emit_progress(150.0, "almost done")
        err = capsys.readouterr().err.strip()
        assert err.startswith("@@PROGRESS@@")
        import json

        payload = json.loads(err[len("@@PROGRESS@@"):])
        assert payload == {"percent": 100.0, "message": "almost done"}
