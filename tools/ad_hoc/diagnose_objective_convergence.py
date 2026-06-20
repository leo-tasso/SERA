"""Diagnose why most ethical objectives converge to the same policy.

Hypothesis: the budget-free levers (taxes, regulation) alone can push every
province to the +6%/year realism cap, so any growth-favoring objective shares
the optimum "max growth everywhere" and only relative-inequality objectives
(or indifferent ones, like Rawlsian maximin) can diverge.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "ui"))

from backend_bridge import (
    INDICATORS,
    SPENDING_PARAMS,
    _make_constraint_fn,
    get_spending_intensity_pct,
    parameter_metadata,
)
from sera.config import DATA_DIR
from sera.twin.causal_graph import ANNUAL_PARAMETERS, PARAMETER_EFFECT_DIRECTION
from sera.twin.cli import load_initial_state
from sera.twin.model_trainer import ModelTrainer
from sera.twin.policy import BaselinePolicy, ParamSpec, PolicyModel, RolloutEnv
from sera.twin.simulator import DigitalTwinSimulator
from sera.twin.province_mapping import PROVINCE_SIGLAS_110

HORIZON = 5


class FixedLeverPolicy(PolicyModel):
    """Hold a fixed lever dict for every province and year."""

    def __init__(self, param_specs, levers):
        super().__init__(param_specs)
        self.levers = levers

    def decide(self, state, step, env):
        return {code: dict(self.levers) for code in env.provinces}


def growth_stats(label, env, policy):
    trajectory, gdp_series, _w, _a, _r = env.rollout(policy)
    final = trajectory[trajectory["year"] == trajectory["year"].max()]
    start = env.initial_state
    merged = final[["area_code", "gdp_per_capita"]].merge(
        start[["area_code", "gdp_per_capita"]], on="area_code", suffixes=("_end", "_start")
    )
    annual = (merged["gdp_per_capita_end"] / merged["gdp_per_capita_start"]) ** (1 / HORIZON) - 1
    print(
        f"{label:<34} total_gdp_final={gdp_series[-1]:>10,.0f}  "
        f"annual_growth: min={annual.min():.3%} median={annual.median():.3%} max={annual.max():.3%}  "
        f"provinces_at_cap(>5.9%)={int((annual > 0.059).sum())}/110"
    )


def main():
    state = load_initial_state(DATA_DIR, INDICATORS, 2025)
    state = state.sort_values("area_code").reset_index(drop=True)
    indicator_cols = [c for c in state.columns if c not in {"area_code", "year"}]
    trainer = ModelTrainer.load(REPO_ROOT / "twin_models.joblib")
    metadata = parameter_metadata()
    specs = [ParamSpec(m["key"], m["baseline"], m["min"], m["max"]) for m in metadata]
    spec_by_key = {s.key: s for s in specs}

    free_levers = [k for k in ANNUAL_PARAMETERS if k not in SPENDING_PARAMS]
    print("free (non-budget) levers:", free_levers)

    baseline_levers = {s.key: s.baseline for s in specs}

    # Free levers pushed in their GDP-favourable direction, spending at baseline.
    free_best = dict(baseline_levers)
    for key in free_levers:
        direction = PARAMETER_EFFECT_DIRECTION.get(key, 0)
        spec = spec_by_key[key]
        free_best[key] = spec.max if direction > 0 else spec.min

    # Spending levers maxed, free levers at baseline (budget constraint bites).
    spend_max = dict(baseline_levers)
    for key in SPENDING_PARAMS:
        spend_max[key] = spec_by_key[key].max

    everything = dict(free_best)
    for key in SPENDING_PARAMS:
        everything[key] = spec_by_key[key].max

    scenarios = [
        ("baseline", baseline_levers),
        ("free levers only (taxes slashed)", free_best),
        ("spending maxed only", spend_max),
        ("everything", everything),
    ]

    combos = [
        (0.04, 0.5),  # current defaults
        (0.04, 0.0),  # rules only (model signal off)
        (0.0, 0.5),  # model signal only (rules off)
        (0.0, 0.04),  # capped signal only
        (0.02, 0.03),  # candidate calibrations
        (0.015, 0.02),
        (0.01, 0.015),
    ]
    for strength, signal_cap in combos:
        print(f"\n=== causal_rule_strength = {strength}, policy_signal_cap = {signal_cap} ===")
        simulator = DigitalTwinSimulator(
            trainer,
            indicator_cols,
            list(ANNUAL_PARAMETERS.keys()),
            causal_rule_strength=strength,
            policy_signal_cap=signal_cap,
        )
        env = RolloutEnv(
            simulator=simulator,
            initial_state=state,
            indicator_cols=indicator_cols,
            param_specs=specs,
            provinces=PROVINCE_SIGLAS_110,
            horizon=HORIZON,
            base_year=int(state["year"].max()),
            constraint_fn=_make_constraint_fn(get_spending_intensity_pct()),
        )
        for label, levers in scenarios:
            growth_stats(label, env, FixedLeverPolicy(specs, levers))


if __name__ == "__main__":
    main()
