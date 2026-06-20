"""Verify that per-province targeting now creates real equity trade-offs.

Compares three hand-crafted policies with the same lever vocabulary:
- growth-positive spending boosted in the POOREST 30 provinces (funded by
  slightly higher taxes in the richest 30);
- the mirror image favouring the RICHEST 30;
- uniform baseline.

If the fix worked, these must produce different total GDP / Gini / floor.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "ui"))

from backend_bridge import (
    INDICATORS,
    SPENDING_PARAMS,
    TAX_PARAMS,
    _make_constraint_fn,
    get_spending_intensity_pct,
    parameter_metadata,
)
from sera.config import DATA_DIR
from sera.twin.causal_graph import ANNUAL_PARAMETERS
from sera.twin.cli import load_initial_state
from sera.twin.model_trainer import ModelTrainer
from sera.twin.objectives import gini
from sera.twin.policy import BaselinePolicy, ParamSpec, PolicyModel, RolloutEnv
from sera.twin.province_mapping import PROVINCE_SIGLAS_110
from sera.twin.simulator import DigitalTwinSimulator

HORIZON = 5

# Spending programs with a positive GDP direction (boosted in targeted provinces).
GROWTH_SPENDING = [
    "education_spending_allocation",
    "infrastructure_investment_allocation",
    "rd_innovation_incentives",
    "small_business_support",
    "manufacturing_incentives",
]


class TargetedPolicy(PolicyModel):
    """Boost growth spending in target provinces, raise taxes in funder provinces."""

    def __init__(self, param_specs, targets, funders):
        super().__init__(param_specs)
        self.targets = set(targets)
        self.funders = set(funders)
        self._by_key = {spec.key: spec for spec in param_specs}

    def decide(self, state, step, env):
        allocations = {}
        for code in env.provinces:
            levers = {spec.key: spec.baseline for spec in self.param_specs}
            if code in self.targets:
                for key in GROWTH_SPENDING:
                    spec = self._by_key[key]
                    levers[key] = spec.baseline + 0.75 * (spec.max - spec.baseline)
            if code in self.funders:
                for key in TAX_PARAMS:
                    spec = self._by_key[key]
                    levers[key] = spec.baseline + 0.4 * (spec.max - spec.baseline)
            allocations[code] = levers
        return allocations


def report(label, env, policy):
    trajectory, gdp_series, _w, _a, _r = env.rollout(policy)
    final = trajectory[trajectory["year"] == trajectory["year"].max()]
    values = final["gdp_per_capita"].astype(float)
    print(
        f"{label:<28} total_gdp={gdp_series[-1]:>10,.0f}  gini={gini(values.to_numpy()):.4f}  "
        f"floor={values.min():,.1f}"
    )


def main():
    state = load_initial_state(DATA_DIR, INDICATORS, 2025)
    state = state.sort_values("area_code").reset_index(drop=True)
    indicator_cols = [c for c in state.columns if c not in {"area_code", "year"}]
    trainer = ModelTrainer.load(REPO_ROOT / "twin_models.joblib")
    simulator = DigitalTwinSimulator(trainer, indicator_cols, list(ANNUAL_PARAMETERS.keys()))
    specs = [ParamSpec(m["key"], m["baseline"], m["min"], m["max"]) for m in parameter_metadata()]

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

    ranked = state.sort_values("gdp_per_capita")["area_code"].tolist()
    poorest, richest = ranked[:30], ranked[-30:]

    report("baseline", env, BaselinePolicy(specs))
    report("favour poorest 30", env, TargetedPolicy(specs, poorest, richest))
    report("favour richest 30", env, TargetedPolicy(specs, richest, poorest))


if __name__ == "__main__":
    main()
