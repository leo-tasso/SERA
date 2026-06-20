import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sera.config import DATA_DIR
from sera.twin.budget import (
    SPENDING_PARAMS,
    TAX_PARAMS,
    apply_budget_constraint,
    budget_usage,
    make_constraint_fn,
)
from sera.twin.causal_graph import (
    ANNUAL_PARAMETERS,
    INDICATOR_BOUNDS,
    INDICATOR_TO_INDICATORS,
    get_parameter_reference,
)
from sera.twin.cli import load_initial_state
from sera.twin.data_loader import DataLoader
from sera.twin.model_trainer import ModelTrainer
from sera.twin.objectives import (
    DEFAULT_OBJECTIVE_ID,
    available_objectives,
    build_objective,
    gini,
)
from sera.twin.pareto import nsga2_front
from sera.twin.policy import (
    BlendedPolicy,
    ParamSpec,
    RolloutEnv,
    available_models,
    build_policy,
)
from sera.twin.province_mapping import PROVINCE_SIGLAS_110
from sera.twin.simulator import (
    CAUSAL_RULE_STRENGTH,
    DigitalTwinSimulator,
    documented_coupling_signs,
)

# SPENDING_PARAMS and TAX_PARAMS now live in sera.twin.budget (imported above)
# so the headless experiments don't have to import this UI module to get them.

INDICATORS = {
    "business_density": ("economic", 1),
    "gdp_per_capita": ("economic", 1),
    "income": ("demographic", 1),
    "poverty_rate": ("economic", -1),
    "self_employment": ("labor", 1),
    "unemployment_rate": ("labor", -1),
    "youth_employment": ("labor", 1),
    "completion_rates": ("education", 1),
    "school_enrollment": ("education", 1),
    "healthcare_spending_per_capita": ("healthcare_public_services", 1),
    "healthcare_worker_density": ("healthcare_public_services", 1),
    "life_expectancy": ("social_well_being", 1),
    "digital_infrastructure": ("innovation_infrastructure", 1),
    "patents": ("innovation_infrastructure", 1),
    "transportation_access": ("innovation_infrastructure", 1),
    "air_quality": ("environment", -1),
    "carbon_emissions": ("energy_resources", -1),
    "green_urban_space_per_capita": ("environmental_quality", 1),
    "renewable_energy_percentage": ("energy_resources", 1),
    "sustainability": ("environment", 1),
    "water_quality": ("environmental_quality", 1),
    "public_transportation_usage": ("transportation_mobility", 1),
    "traffic_congestion": ("transportation_mobility", -1),
    "crime_rate": ("social_well_being", -1),
}


def emit_progress(percent: float, message: str) -> None:
    """Stream a structured progress update on stderr for the Electron main process."""
    payload = json.dumps({"percent": round(min(max(percent, 0.0), 100.0), 1), "message": message})
    print(f"@@PROGRESS@@{payload}", file=sys.stderr, flush=True)


def format_label(key: str) -> str:
    parts = []
    for part in key.split("_"):
        if part == "gdp":
            parts.append("GDP")
        elif part == "rd":
            parts.append("R&D")
        else:
            parts.append(part.capitalize())
    return " ".join(parts)


def dataframe_records(frame: pd.DataFrame) -> list[dict]:
    sanitized = frame.astype(object).where(pd.notna(frame), None)
    return sanitized.to_dict(orient="records")


def get_spending_intensity_pct() -> float:
    """Return the public spending intensity as % of GDP from historical data."""
    path = (
        DATA_DIR
        / "public_finance"
        / "public_spending_efficiency"
        / "public_spending_efficiency_raw_2001_2025.csv"
    )
    try:
        frame = pd.read_csv(path)
        col = "public_spending_intensity_pct_gdp"
        if col in frame.columns:
            series = pd.to_numeric(frame[col], errors="coerce").dropna()
            if not series.empty:
                return float(series.iloc[-1])
    except Exception:
        pass
    return 19.0


def parameter_limits(key: str) -> tuple[float, float, float, float]:
    baseline, scale = get_parameter_reference(key)
    min_value = max(0.0, baseline - scale * 2)
    max_value = max(min_value + max(scale * 4, 1.0), baseline + scale * 2)
    step = max((max_value - min_value) / 100.0, 0.1)
    return float(baseline), float(min_value), float(max_value), round(step, 2)


def parameter_metadata() -> list[dict]:
    metadata = []
    for key, label in ANNUAL_PARAMETERS.items():
        baseline, min_value, max_value, step = parameter_limits(key)
        metadata.append(
            {
                "key": key,
                "label": label,
                "baseline": baseline,
                "min": min_value,
                "max": max_value,
                "step": step,
            }
        )
    return metadata


def default_allocations(metadata: list[dict]) -> dict[str, dict[str, float]]:
    defaults = {}
    for province in PROVINCE_SIGLAS_110:
        defaults[province] = {item["key"]: item["baseline"] for item in metadata}
    return defaults


def build_bootstrap(payload: dict) -> dict:
    indicator_keys = list(INDICATORS.keys())
    baseline_year = int(payload.get("baselineYear", 2025))
    latest_state = load_initial_state(DATA_DIR, INDICATORS, baseline_year)
    latest_state = latest_state.sort_values("area_code").reset_index(drop=True)
    available_indicator_keys = [key for key in indicator_keys if key in latest_state.columns]
    baseline_year = int(latest_state["year"].max())
    metadata = parameter_metadata()

    return {
        "baselineYear": baseline_year,
        "indicatorKeys": available_indicator_keys,
        "indicatorLabels": {key: format_label(key) for key in available_indicator_keys},
        "parameterMeta": metadata,
        "defaultAllocations": default_allocations(metadata),
        "latestStateRows": dataframe_records(latest_state),
        "provinces": PROVINCE_SIGLAS_110,
        "spendingIntensityPct": get_spending_intensity_pct(),
        "models": available_models(),
        "objectives": available_objectives(),
    }


def load_province_trends(payload: dict) -> dict:
    province_code = str(payload.get("provinceCode") or "").strip().upper()
    if not province_code:
        raise ValueError("provinceCode is required to load province trends.")

    requested_keys = [key for key in payload.get("indicatorKeys", []) if key in INDICATORS]
    if not requested_keys:
        requested_keys = list(INDICATORS.keys())[:4]

    start_year = int(payload.get("startYear", 2016))
    end_year = int(payload.get("endYear", 2025))

    loader = DataLoader(DATA_DIR)
    combined = None
    for key in requested_keys:
        category, _direction = INDICATORS[key]
        frame = loader.load_indicator(key, category)
        if frame.empty:
            continue

        frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
        frame = frame.dropna(subset=["year"])
        frame["year"] = frame["year"].astype(int)
        frame = frame[(frame["year"] >= start_year) & (frame["year"] <= end_year)].copy()
        if frame.empty:
            continue

        frame = loader.disaggregate_national_to_provincial(frame)
        frame = loader.disaggregate_regional_to_provincial(frame)
        frame = loader.standardize_to_province_level(frame, interpolate_missing=True)
        frame = frame[frame["area_code"] == province_code][["area_code", "year", "value"]].copy()
        frame = frame.rename(columns={"value": key})

        if combined is None:
            combined = frame
        else:
            combined = combined.merge(frame, on=["area_code", "year"], how="outer")

    if combined is None:
        combined = pd.DataFrame(columns=["area_code", "year"])

    combined = combined.sort_values(["year", "area_code"]).reset_index(drop=True)
    return {
        "provinceCode": province_code,
        "indicatorKeys": requested_keys,
        "rows": dataframe_records(combined),
    }


def _param_baselines() -> dict[str, float]:
    """Lever baselines (historical medians) keyed by lever name, for the budget."""
    return {key: parameter_limits(key)[0] for key in ANNUAL_PARAMETERS}


def _budget_usage(
    allocations: dict[str, dict[str, float]],
    current_state: pd.DataFrame,
    spending_intensity_pct: float = 19.0,
) -> tuple[float, float]:
    """Bridge wrapper around :func:`sera.twin.budget.budget_usage`."""
    return budget_usage(allocations, current_state, spending_intensity_pct, _param_baselines())


def _apply_budget_constraint(
    allocations: dict[str, dict[str, float]],
    current_state: pd.DataFrame,
    spending_intensity_pct: float = 19.0,
    reserve_pool: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Bridge wrapper around :func:`sera.twin.budget.apply_budget_constraint`."""
    return apply_budget_constraint(
        allocations, current_state, spending_intensity_pct, _param_baselines(), reserve_pool
    )


def _make_constraint_fn(spending_intensity_pct: float):
    """Bridge wrapper around :func:`sera.twin.budget.make_constraint_fn`."""
    return make_constraint_fn(spending_intensity_pct, _param_baselines())


def coupling_signs_for_mode(mode: str):
    """Inter-indicator propagation signs for an ablation mode.

    - ``reviewed`` (default): documented signs overlaid with panel-learned signs
      (``None`` -> the simulator's own resolution from data/learned_couplings.json);
    - ``documented``: the polarity-derived signs only, no panel review;
    - ``signless``: every edge +1, reproducing the original implementation that
      pushed each target in the direction of its source's change.
    """
    if mode == "signless":
        return {src: {tgt: 1 for tgt in tgts} for src, tgts in INDICATOR_TO_INDICATORS.items()}
    if mode == "documented":
        return documented_coupling_signs()
    return None  # 'reviewed' -> simulator loads learned overlay itself


def build_parameters_frame(
    next_year: int, allocations: dict[str, dict[str, float]]
) -> pd.DataFrame:
    parameter_keys = list(ANNUAL_PARAMETERS.keys())
    rows = []
    for province in PROVINCE_SIGLAS_110:
        province_allocations = allocations.get(province, {})
        row = {"area_code": province, "year": next_year}
        for key in parameter_keys:
            baseline, min_value, max_value, _step = parameter_limits(key)
            raw_value = province_allocations.get(key, baseline)
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = float(baseline)
            row[key] = min(max(value, float(min_value)), float(max_value))
        rows.append(row)
    return pd.DataFrame(rows)


def simulate_next_year(payload: dict) -> dict:
    current_state = pd.DataFrame(payload.get("currentStateRows") or [])
    if current_state.empty:
        raise ValueError("currentStateRows is required to run the simulation bridge.")

    if "area_code" not in current_state.columns or "year" not in current_state.columns:
        raise ValueError("currentStateRows must contain area_code and year columns.")

    current_state = current_state.copy()
    current_state["year"] = current_state["year"].astype(int)
    current_state = current_state.sort_values("area_code").reset_index(drop=True)
    current_year = int(payload.get("currentYear") or current_state["year"].max())
    next_year = current_year + 1

    emit_progress(10, "Loading trained twin...")
    model_path = Path(payload.get("modelPath") or REPO_ROOT / "twin_models.joblib")
    trainer = ModelTrainer.load(model_path)
    raw_allocations = payload.get("allocations") or {}
    spending_intensity_pct = float(
        payload.get("spendingIntensityPct") or get_spending_intensity_pct()
    )
    reserve_pool = float(payload.get("reservePool") or 0.0)
    constraint_fn = _make_constraint_fn(spending_intensity_pct)
    constrained_allocations, new_reserve = constraint_fn(
        raw_allocations, current_state, reserve_pool
    )
    parameters_frame = build_parameters_frame(next_year, constrained_allocations)

    indicator_columns = [
        column for column in current_state.columns if column not in {"area_code", "year"}
    ]
    simulator = DigitalTwinSimulator(trainer, indicator_columns, list(ANNUAL_PARAMETERS.keys()))
    emit_progress(50, f"Simulating year {next_year}...")
    next_state = simulator.simulate_year(
        current_state, parameters_frame, apply_rules=True, apply_bounds=True
    )
    next_state = next_state.sort_values("area_code").reset_index(drop=True)
    emit_progress(95, "Collecting results...")

    summary_keys = ["gdp_per_capita", "income", "unemployment_rate", "life_expectancy"]
    summary = {}
    for key in summary_keys:
        if key in next_state.columns:
            summary[key] = float(next_state[key].mean())

    return {
        "nextYear": next_year,
        "nextStateRows": dataframe_records(next_state),
        "summary": summary,
        "reservePool": float(new_reserve),
    }


def _series_payload(base_year: int, series: list[float]) -> list[dict]:
    return [
        {"year": base_year + index + 1, "value": float(value)} for index, value in enumerate(series)
    ]


# Causal-rule strength multipliers used for the sensitivity band: the same
# policy re-simulated with the hand-written rules at half and 1.5x strength.
SENSITIVITY_RULE_SCALES = [0.5, 1.5]


def optimize_policy(payload: dict) -> dict:
    """Train a policy on the chosen ethical objective and return graded candidates.

    Decision support, not decision making: nothing is applied automatically.
    The response carries three intervention candidates — full, moderate (levers
    halfway back toward baseline), and the historical baseline — each with its
    trajectory and equity metrics, and for the trained candidates a sensitivity
    band showing how the GDP path moves when the hand-written causal-rule
    strength is halved or increased by half. The human adopts (or rejects) a
    candidate in the UI.
    """
    current_state = pd.DataFrame(payload.get("currentStateRows") or [])
    if current_state.empty:
        raise ValueError("currentStateRows is required to run the policy optimizer.")
    if "area_code" not in current_state.columns or "year" not in current_state.columns:
        raise ValueError("currentStateRows must contain area_code and year columns.")

    current_state = current_state.copy()
    current_state["year"] = current_state["year"].astype(int)
    current_state = current_state.sort_values("area_code").reset_index(drop=True)
    current_year = int(payload.get("currentYear") or current_state["year"].max())

    horizon = int(payload.get("horizon") or 20)
    horizon = max(1, min(horizon, 50))
    iterations = int(payload.get("iterations") or 6)
    iterations = max(1, min(iterations, 40))
    model_id = str(payload.get("modelId") or "neural")
    seed = int(payload.get("seed") or 0)
    objective_id = str(payload.get("objectiveId") or DEFAULT_OBJECTIVE_ID)
    objective_params = payload.get("objectiveParams") or {}
    objective = build_objective(objective_id, **objective_params)
    spending_intensity_pct = float(
        payload.get("spendingIntensityPct") or get_spending_intensity_pct()
    )
    reserve_pool = float(payload.get("reservePool") or 0.0)
    final_year = current_year + horizon

    emit_progress(2, "Loading trained twin...")
    model_path = Path(payload.get("modelPath") or REPO_ROOT / "twin_models.joblib")
    trainer = ModelTrainer.load(model_path)

    indicator_columns = [
        column for column in current_state.columns if column not in {"area_code", "year"}
    ]
    param_specs = [
        ParamSpec(item["key"], item["baseline"], item["min"], item["max"])
        for item in parameter_metadata()
    ]

    def make_env(rule_scale: float = 1.0) -> RolloutEnv:
        simulator = DigitalTwinSimulator(
            trainer,
            indicator_columns,
            list(ANNUAL_PARAMETERS.keys()),
            causal_rule_strength=CAUSAL_RULE_STRENGTH * rule_scale,
        )
        return RolloutEnv(
            simulator=simulator,
            initial_state=current_state,
            indicator_cols=indicator_columns,
            param_specs=param_specs,
            provinces=PROVINCE_SIGLAS_110,
            horizon=horizon,
            base_year=current_year,
            constraint_fn=_make_constraint_fn(spending_intensity_pct),
            reserve_pool=reserve_pool,
            objective=build_objective(objective_id, **objective_params),
        )

    env = make_env()

    def candidate_payload(
        policy, candidate_id: str, label: str, description: str, with_band: bool
    ) -> dict:
        trajectory, gdp_series, welfare_series, allocations_by_year, reserve = env.rollout(policy)
        report = _trajectory_report(trajectory, gdp_series, current_year, final_year)
        report.pop("finalGdpByProvince", None)

        band = None
        if with_band and gdp_series:
            lows = list(gdp_series)
            highs = list(gdp_series)
            for scale in SENSITIVITY_RULE_SCALES:
                _t, scaled_gdp, _w, _a, _r = make_env(scale).rollout(policy)
                lows = [min(low, value) for low, value in zip(lows, scaled_gdp)]
                highs = [max(high, value) for high, value in zip(highs, scaled_gdp)]
            band = [
                {"year": current_year + index + 1, "low": float(low), "high": float(high)}
                for index, (low, high) in enumerate(zip(lows, highs))
            ]

        summary = {}
        if not trajectory.empty:
            final_state = trajectory[trajectory["year"] == final_year]
            for key in ["gdp_per_capita", "income", "unemployment_rate", "life_expectancy"]:
                if key in final_state.columns:
                    summary[key] = float(final_state[key].mean())

        report.update(
            {
                "id": candidate_id,
                "label": label,
                "description": description,
                "welfareByYear": _series_payload(current_year, welfare_series),
                "gdpBandByYear": band,
                "trajectoryRows": dataframe_records(trajectory),
                "finalAllocations": allocations_by_year.get(final_year, {}),
                "reservePool": float(reserve),
                "summary": summary,
            }
        )
        return report

    baseline_policy = build_policy("baseline", param_specs)
    baseline_description = (
        "Every lever stays at its historical value. The do-nothing reference: "
        "choosing it is also a policy decision."
    )

    train_info: dict = {}
    candidates: list[dict] = []
    explanation = None
    explainability = None
    if model_id == "baseline":
        emit_progress(50, "Rolling out baseline trajectory...")
        candidates.append(
            candidate_payload(
                baseline_policy,
                "baseline",
                "Baseline (historical levers)",
                baseline_description,
                False,
            )
        )
    else:
        policy = build_policy(model_id, param_specs)

        def progress(step: int, total: int, score: float) -> None:
            print(
                f"Training {model_id} on {objective.label}: iteration {step}/{total} "
                f"- best cumulative score {score:,.2f}",
                file=sys.stderr,
                flush=True,
            )
            # Training spans the 10% -> 78% window of the overall run.
            emit_progress(
                10 + 68 * step / max(total, 1),
                f"Training {model_id} ({objective.label}): iteration {step}/{total}",
            )

        emit_progress(10, f"Training {model_id} ({objective.label}): iteration 0/{iterations}")
        train_info = policy.fit(env, iterations=iterations, progress=progress)

        # Explanation of the trained policy (white-box parameters, gray-box
        # partial dependence, or black-box post-hoc audit, by model class).
        explainability = policy.explainability
        emit_progress(79, "Building the policy explanation...")
        try:
            explanation = policy.explain(env)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            explanation = None

        emit_progress(80, "Rolling out full intervention + sensitivity band...")
        candidates.append(
            candidate_payload(
                policy,
                "full",
                "Full intervention",
                "The levers exactly as the model optimized them. Largest projected "
                "effect, largest dependence on the twin's assumptions.",
                True,
            )
        )
        emit_progress(88, "Rolling out moderate intervention + sensitivity band...")
        candidates.append(
            candidate_payload(
                BlendedPolicy(policy, 0.5),
                "moderate",
                "Moderate intervention",
                "Same policy direction with every lever moved only halfway from "
                "baseline. Smaller projected effect, smaller bet on the model "
                "being right.",
                True,
            )
        )
        emit_progress(95, "Rolling out baseline reference...")
        candidates.append(
            candidate_payload(
                baseline_policy,
                "baseline",
                "Baseline (historical levers)",
                baseline_description,
                False,
            )
        )

    emit_progress(98, "Collecting results...")
    return {
        "modelId": model_id,
        "objectiveId": objective_id,
        "objectiveLabel": objective.label,
        "horizon": horizon,
        "finalYear": final_year,
        "trainInfo": train_info,
        "candidates": candidates,
        "sensitivityScales": SENSITIVITY_RULE_SCALES,
        "explainability": explainability,
        "explanation": explanation,
    }


def _equity_series(trajectory: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Per-year Gini and worst-off provincial GDP for a simulated trajectory."""
    gini_series: list[dict] = []
    worst_series: list[dict] = []
    if trajectory.empty or "gdp_per_capita" not in trajectory.columns:
        return gini_series, worst_series
    years = sorted(
        pd.to_numeric(trajectory["year"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    for year in years:
        group = trajectory[trajectory["year"] == year]
        values = pd.to_numeric(group["gdp_per_capita"], errors="coerce").dropna()
        if values.empty:
            continue
        gini_series.append({"year": year, "value": float(gini(values.to_numpy()))})
        worst_series.append({"year": year, "value": float(values.min())})
    return gini_series, worst_series


def _final_gdp_by_province(trajectory: pd.DataFrame, final_year: int) -> dict[str, float]:
    if trajectory.empty or "gdp_per_capita" not in trajectory.columns:
        return {}
    rows = trajectory[trajectory["year"] == final_year]
    result: dict[str, float] = {}
    for _, row in rows.iterrows():
        value = pd.to_numeric(pd.Series([row["gdp_per_capita"]]), errors="coerce").iloc[0]
        if pd.notna(value):
            result[str(row["area_code"]).strip().upper()] = float(value)
    return result


def _trajectory_report(
    trajectory: pd.DataFrame, gdp_series: list[float], current_year: int, final_year: int
) -> dict:
    """Equity-focused summary of one rollout, shared by baseline and objectives."""
    gini_series, worst_series = _equity_series(trajectory)
    return {
        "gdpByYear": _series_payload(current_year, gdp_series),
        "giniByYear": gini_series,
        "worstGdpByYear": worst_series,
        "finalGdpByProvince": _final_gdp_by_province(trajectory, final_year),
        "finalGdpTotal": float(gdp_series[-1]) if gdp_series else 0.0,
        "finalGini": float(gini_series[-1]["value"]) if gini_series else 0.0,
        "worstProvinceGdp": float(worst_series[-1]["value"]) if worst_series else 0.0,
    }


def compare_objectives(payload: dict) -> dict:
    """Train the selected model once per ethical objective and report equity outcomes.

    A read-only what-if analysis: the twin state is never advanced, the same
    starting state feeds every framework, and the response carries the GDP,
    Gini, and worst-off-province trajectories needed to compare them.
    """
    current_state = pd.DataFrame(payload.get("currentStateRows") or [])
    if current_state.empty:
        raise ValueError("currentStateRows is required to compare objectives.")
    if "area_code" not in current_state.columns or "year" not in current_state.columns:
        raise ValueError("currentStateRows must contain area_code and year columns.")

    current_state = current_state.copy()
    current_state["year"] = current_state["year"].astype(int)
    current_state = current_state.sort_values("area_code").reset_index(drop=True)
    current_year = int(payload.get("currentYear") or current_state["year"].max())

    horizon = max(1, min(int(payload.get("horizon") or 20), 50))
    iterations = max(1, min(int(payload.get("iterations") or 6), 40))
    model_id = str(payload.get("modelId") or "neural")
    seed = int(payload.get("seed") or 0)
    propagation_mode = str(payload.get("propagationMode") or "reviewed")
    spending_intensity_pct = float(
        payload.get("spendingIntensityPct") or get_spending_intensity_pct()
    )
    reserve_pool = float(payload.get("reservePool") or 0.0)
    final_year = current_year + horizon

    emit_progress(2, "Loading trained twin...")
    model_path = Path(payload.get("modelPath") or REPO_ROOT / "twin_models.joblib")
    trainer = ModelTrainer.load(model_path)
    indicator_columns = [
        column for column in current_state.columns if column not in {"area_code", "year"}
    ]
    simulator = DigitalTwinSimulator(
        trainer,
        indicator_columns,
        list(ANNUAL_PARAMETERS.keys()),
        coupling_signs=coupling_signs_for_mode(propagation_mode),
    )
    param_specs = [
        ParamSpec(item["key"], item["baseline"], item["min"], item["max"])
        for item in parameter_metadata()
    ]

    def make_env(objective):
        return RolloutEnv(
            simulator=simulator,
            initial_state=current_state,
            indicator_cols=indicator_columns,
            param_specs=param_specs,
            provinces=PROVINCE_SIGLAS_110,
            horizon=horizon,
            base_year=current_year,
            constraint_fn=_make_constraint_fn(spending_intensity_pct),
            reserve_pool=reserve_pool,
            objective=objective,
        )

    emit_progress(4, "Computing baseline trajectory...")
    baseline_env = make_env(None)
    baseline_policy = build_policy("baseline", param_specs)
    baseline_traj, baseline_gdp, _w, _a, _r = baseline_env.rollout(baseline_policy)
    baseline_report = _trajectory_report(baseline_traj, baseline_gdp, current_year, final_year)

    objectives_meta = available_objectives()
    requested_ids = payload.get("objectiveIds")
    if requested_ids:
        wanted = set(requested_ids)
        objectives_meta = [meta for meta in objectives_meta if meta["id"] in wanted]
    objective_params_by_id = payload.get("objectiveParams") or {}
    span = 88.0 / max(len(objectives_meta), 1)  # training spans the 8% -> 96% window
    results = []
    for index, meta in enumerate(objectives_meta):
        objective = build_objective(meta["id"], **(objective_params_by_id.get(meta["id"]) or {}))
        window_start = 8.0 + span * index
        env = make_env(objective)
        policy = build_policy(model_id, param_specs, seed=seed)

        train_info: dict = {}
        if policy.trainable:

            def progress(
                step: int, total: int, score: float, _start=window_start, _label=meta["label"]
            ) -> None:
                print(
                    f"[{index + 1}/{len(objectives_meta)}] Training {model_id} on {_label}: "
                    f"iteration {step}/{total} - best cumulative score {score:,.2f}",
                    file=sys.stderr,
                    flush=True,
                )
                emit_progress(
                    _start + (span - 4.0) * step / max(total, 1),
                    f"[{index + 1}/{len(objectives_meta)}] Training {model_id} on {_label}: "
                    f"iteration {step}/{total}",
                )

            train_info = policy.fit(env, iterations=iterations, progress=progress)
        emit_progress(window_start + span - 3.0, f'Rolling out {meta["label"]}...')
        trajectory, gdp_series, welfare_series, _allocations, _reserve = env.rollout(policy)

        report = _trajectory_report(trajectory, gdp_series, current_year, final_year)
        report.update(
            {
                "objectiveId": meta["id"],
                "objectiveLabel": meta["label"],
                "welfareByYear": _series_payload(current_year, welfare_series),
                "trainInfo": train_info,
            }
        )
        results.append(report)

    emit_progress(98, "Collecting comparison results...")
    return {
        "modelId": model_id,
        "horizon": horizon,
        "finalYear": final_year,
        "baseline": baseline_report,
        "results": results,
    }


def pareto_front(payload: dict) -> dict:
    """Map the efficiency–equity frontier with NSGA-II (read-only what-if).

    Evolves uniform national lever vectors against three objectives at once —
    total GDP, inter-provincial Gini, and the worst-off province's GDP — and
    returns the non-dominated front. Nothing is trained per framework and
    nothing is applied to the twin: the frontier shows what each ethical
    dropdown choice would be a corner of.
    """
    current_state = pd.DataFrame(payload.get("currentStateRows") or [])
    if current_state.empty:
        raise ValueError("currentStateRows is required to map the Pareto frontier.")
    if "area_code" not in current_state.columns or "year" not in current_state.columns:
        raise ValueError("currentStateRows must contain area_code and year columns.")

    current_state = current_state.copy()
    current_state["year"] = current_state["year"].astype(int)
    current_state = current_state.sort_values("area_code").reset_index(drop=True)
    current_year = int(payload.get("currentYear") or current_state["year"].max())

    horizon = max(1, min(int(payload.get("horizon") or 20), 50))
    generations = max(1, min(int(payload.get("iterations") or 6), 40))
    popsize = max(4, min(int(payload.get("popsize") or 12), 40))
    n_clusters = max(1, min(int(payload.get("nClusters") or 1), 12))
    spending_intensity_pct = float(
        payload.get("spendingIntensityPct") or get_spending_intensity_pct()
    )
    reserve_pool = float(payload.get("reservePool") or 0.0)
    final_year = current_year + horizon

    emit_progress(2, "Loading trained twin...")
    model_path = Path(payload.get("modelPath") or REPO_ROOT / "twin_models.joblib")
    trainer = ModelTrainer.load(model_path)
    indicator_columns = [
        column for column in current_state.columns if column not in {"area_code", "year"}
    ]
    simulator = DigitalTwinSimulator(trainer, indicator_columns, list(ANNUAL_PARAMETERS.keys()))
    param_specs = [
        ParamSpec(item["key"], item["baseline"], item["min"], item["max"])
        for item in parameter_metadata()
    ]
    env = RolloutEnv(
        simulator=simulator,
        initial_state=current_state,
        indicator_cols=indicator_columns,
        param_specs=param_specs,
        provinces=PROVINCE_SIGLAS_110,
        horizon=horizon,
        base_year=current_year,
        constraint_fn=_make_constraint_fn(spending_intensity_pct),
        reserve_pool=reserve_pool,
    )

    emit_progress(4, "Computing baseline reference point...")
    baseline_policy = build_policy("baseline", param_specs)
    baseline_traj, baseline_gdp, _w, _a, _r = env.rollout(baseline_policy)
    baseline_report = _trajectory_report(baseline_traj, baseline_gdp, current_year, final_year)
    baseline_report.pop("finalGdpByProvince", None)

    def progress(generation: int, total: int, evaluations: int) -> None:
        print(
            f"Pareto search: generation {generation}/{total} " f"({evaluations} rollouts so far)",
            file=sys.stderr,
            flush=True,
        )
        emit_progress(
            6 + 90 * generation / max(total, 1),
            f"Pareto search: generation {generation}/{total}",
        )

    result = nsga2_front(
        env,
        popsize=popsize,
        generations=generations,
        seed=0,
        n_clusters=n_clusters,
        progress=progress,
    )

    emit_progress(97, "Collecting frontier points...")
    points = []
    for point in result["points"]:
        x = point["x"]
        metrics = point["metrics"]
        levers = {
            spec.key: float(
                spec.min + float(np.clip(x[col], 0.0, 1.0)) * max(spec.max - spec.min, 0.0)
            )
            for col, spec in enumerate(param_specs)
        }
        entry = {
            "levers": levers,
            "finalGdpTotal": float(metrics["gdp_total"]),
            "finalGini": float(metrics["gini"]),
            "worstProvinceGdp": float(metrics["worst_gdp"]),
            "reservePool": float(metrics.get("reserve", 0.0)),
            "tags": list(point.get("tags", [])),
        }
        if point.get("clusters"):
            entry["clusters"] = point["clusters"]
        points.append(entry)

    return {
        "horizon": horizon,
        "finalYear": final_year,
        "generations": generations,
        "popsize": popsize,
        "nClusters": int(result.get("nClusters", n_clusters)),
        "evaluations": int(result["evaluations"]),
        "baseline": {
            "finalGdpTotal": baseline_report["finalGdpTotal"],
            "finalGini": baseline_report["finalGini"],
            "worstProvinceGdp": baseline_report["worstProvinceGdp"],
        },
        "points": points,
    }


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "bootstrap"
    raw_input = ""
    if not sys.stdin.isatty():
        raw_input = sys.stdin.read().strip()
    payload = json.loads(raw_input) if raw_input else {}

    if command == "bootstrap":
        result = build_bootstrap(payload)
    elif command == "province-trends":
        result = load_province_trends(payload)
    elif command == "simulate-next-year":
        print("Loading trained model and applying one-year policy scenario...", file=sys.stderr)
        result = simulate_next_year(payload)
    elif command == "optimize-policy":
        print("Loading trained twin and running policy model over the horizon...", file=sys.stderr)
        result = optimize_policy(payload)
    elif command == "compare-objectives":
        print("Training the policy model once per ethical objective...", file=sys.stderr)
        result = compare_objectives(payload)
    elif command == "pareto-front":
        print("Mapping the efficiency-equity frontier with NSGA-II...", file=sys.stderr)
        result = pareto_front(payload)
    else:
        raise ValueError(f"Unknown bridge command: {command}")

    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": str(error)}, allow_nan=False))
        sys.exit(1)
