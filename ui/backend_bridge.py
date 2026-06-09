import json
import sys
import traceback
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sera.config import DATA_DIR
from sera.twin.causal_graph import ANNUAL_PARAMETERS, INDICATOR_BOUNDS, get_parameter_reference
from sera.twin.cli import load_initial_state
from sera.twin.data_loader import DataLoader
from sera.twin.model_trainer import ModelTrainer
from sera.twin.policy import (
    ParamSpec,
    RolloutEnv,
    available_models,
    build_policy,
)
from sera.twin.province_mapping import PROVINCE_SIGLAS_110
from sera.twin.simulator import DigitalTwinSimulator

SPENDING_PARAMS = {
    'healthcare_spending_allocation',
    'education_spending_allocation',
    'infrastructure_investment_allocation',
    'social_welfare_spending_allocation',
    'rd_innovation_incentives',
    'green_energy_environment_investment',
    'pension_retirement_spending',
    'agriculture_support_level',
    'manufacturing_incentives',
    'tourism_support_level',
    'small_business_support',
    'public_sector_wage_levels',
    'housing_urban_development_support',
}

INDICATORS = {
    'business_density': ('economic', 1),
    'gdp_per_capita': ('economic', 1),
    'income': ('demographic', 1),
    'poverty_rate': ('economic', -1),
    'self_employment': ('labor', 1),
    'unemployment_rate': ('labor', -1),
    'youth_employment': ('labor', 1),
    'completion_rates': ('education', 1),
    'school_enrollment': ('education', 1),
    'healthcare_spending_per_capita': ('healthcare_public_services', 1),
    'healthcare_worker_density': ('healthcare_public_services', 1),
    'life_expectancy': ('social_well_being', 1),
    'digital_infrastructure': ('innovation_infrastructure', 1),
    'patents': ('innovation_infrastructure', 1),
    'transportation_access': ('innovation_infrastructure', 1),
    'air_quality': ('environment', -1),
    'carbon_emissions': ('energy_resources', -1),
    'green_urban_space_per_capita': ('environmental_quality', 1),
    'renewable_energy_percentage': ('energy_resources', 1),
    'sustainability': ('environment', 1),
    'water_quality': ('environmental_quality', 1),
    'public_transportation_usage': ('transportation_mobility', 1),
    'traffic_congestion': ('transportation_mobility', -1),
    'crime_rate': ('social_well_being', -1),
}


def emit_progress(percent: float, message: str) -> None:
    """Stream a structured progress update on stderr for the Electron main process."""
    payload = json.dumps(
        {'percent': round(min(max(percent, 0.0), 100.0), 1), 'message': message}
    )
    print(f'@@PROGRESS@@{payload}', file=sys.stderr, flush=True)


def format_label(key: str) -> str:
    parts = []
    for part in key.split('_'):
        if part == 'gdp':
            parts.append('GDP')
        elif part == 'rd':
            parts.append('R&D')
        else:
            parts.append(part.capitalize())
    return ' '.join(parts)


def dataframe_records(frame: pd.DataFrame) -> list[dict]:
    sanitized = frame.astype(object).where(pd.notna(frame), None)
    return sanitized.to_dict(orient='records')


def get_spending_intensity_pct() -> float:
    """Return the public spending intensity as % of GDP from historical data."""
    path = (
        DATA_DIR
        / 'public_finance'
        / 'public_spending_efficiency'
        / 'public_spending_efficiency_raw_2001_2025.csv'
    )
    try:
        frame = pd.read_csv(path)
        col = 'public_spending_intensity_pct_gdp'
        if col in frame.columns:
            series = pd.to_numeric(frame[col], errors='coerce').dropna()
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
                'key': key,
                'label': label,
                'baseline': baseline,
                'min': min_value,
                'max': max_value,
                'step': step,
            }
        )
    return metadata


def default_allocations(metadata: list[dict]) -> dict[str, dict[str, float]]:
    defaults = {}
    for province in PROVINCE_SIGLAS_110:
        defaults[province] = {item['key']: item['baseline'] for item in metadata}
    return defaults


def build_bootstrap(payload: dict) -> dict:
    indicator_keys = list(INDICATORS.keys())
    baseline_year = int(payload.get('baselineYear', 2025))
    latest_state = load_initial_state(DATA_DIR, INDICATORS, baseline_year)
    latest_state = latest_state.sort_values('area_code').reset_index(drop=True)
    available_indicator_keys = [key for key in indicator_keys if key in latest_state.columns]
    baseline_year = int(latest_state['year'].max())
    metadata = parameter_metadata()

    return {
        'baselineYear': baseline_year,
        'indicatorKeys': available_indicator_keys,
        'indicatorLabels': {key: format_label(key) for key in available_indicator_keys},
        'parameterMeta': metadata,
        'defaultAllocations': default_allocations(metadata),
        'latestStateRows': dataframe_records(latest_state),
        'provinces': PROVINCE_SIGLAS_110,
        'spendingIntensityPct': get_spending_intensity_pct(),
        'models': available_models(),
    }


def load_province_trends(payload: dict) -> dict:
    province_code = str(payload.get('provinceCode') or '').strip().upper()
    if not province_code:
        raise ValueError('provinceCode is required to load province trends.')

    requested_keys = [
        key for key in payload.get('indicatorKeys', [])
        if key in INDICATORS
    ]
    if not requested_keys:
        requested_keys = list(INDICATORS.keys())[:4]

    start_year = int(payload.get('startYear', 2016))
    end_year = int(payload.get('endYear', 2025))

    loader = DataLoader(DATA_DIR)
    combined = None
    for key in requested_keys:
        category, _direction = INDICATORS[key]
        frame = loader.load_indicator(key, category)
        if frame.empty:
            continue

        frame['year'] = pd.to_numeric(frame['year'], errors='coerce')
        frame = frame.dropna(subset=['year'])
        frame['year'] = frame['year'].astype(int)
        frame = frame[(frame['year'] >= start_year) & (frame['year'] <= end_year)].copy()
        if frame.empty:
            continue

        frame = loader.disaggregate_national_to_provincial(frame)
        frame = loader.disaggregate_regional_to_provincial(frame)
        frame = loader.standardize_to_province_level(frame, interpolate_missing=True)
        frame = frame[frame['area_code'] == province_code][['area_code', 'year', 'value']].copy()
        frame = frame.rename(columns={'value': key})

        if combined is None:
            combined = frame
        else:
            combined = combined.merge(frame, on=['area_code', 'year'], how='outer')

    if combined is None:
        combined = pd.DataFrame(columns=['area_code', 'year'])

    combined = combined.sort_values(['year', 'area_code']).reset_index(drop=True)
    return {
        'provinceCode': province_code,
        'indicatorKeys': requested_keys,
        'rows': dataframe_records(combined),
    }


def _budget_usage(
    allocations: dict[str, dict[str, float]],
    current_state: pd.DataFrame,
    spending_intensity_pct: float = 19.0,
) -> tuple[float, float]:
    """Return ``(total_used, base_pool)`` for the national spending budget.

    Cost formula (mirrors frontend ResourceMeter):
      province_cost = avg(val / baseline  for each spending param) * (intensity/100) * gdp
      province_base_limit = (intensity/100) * gdp
    At historical baseline values, cost == limit exactly (100% utilisation).
    """
    spending_keys = list(SPENDING_PARAMS)
    if not spending_keys:
        return 0.0, 0.0

    param_baselines = {key: max(parameter_limits(key)[0], 1e-9) for key in spending_keys}

    gdp_by_province: dict[str, float] = {}
    for _, row in current_state.iterrows():
        code = str(row.get('area_code', '')).strip().upper()
        gdp = float(row.get('gdp_per_capita', 0) or 0)
        if gdp > 0:
            gdp_by_province[code] = gdp

    base_pool = sum(gdp_by_province.values()) * spending_intensity_pct / 100.0

    total_used = 0.0
    for code, gdp in gdp_by_province.items():
        prov = allocations.get(code, {})
        ratio_sum = sum(
            float(prov.get(key, param_baselines[key])) / param_baselines[key]
            for key in spending_keys
        )
        avg_ratio = ratio_sum / len(spending_keys)
        total_used += avg_ratio * gdp * spending_intensity_pct / 100.0

    return total_used, base_pool


def _apply_budget_constraint(
    allocations: dict[str, dict[str, float]],
    current_state: pd.DataFrame,
    spending_intensity_pct: float = 19.0,
    reserve_pool: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Scale spending params down if total national cost exceeds the available budget.

    Unused budget from prior years (reserve_pool) extends the effective national pool.
    """
    spending_keys = list(SPENDING_PARAMS)
    if not spending_keys:
        return allocations

    total_used, base_pool = _budget_usage(allocations, current_state, spending_intensity_pct)
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


def _make_constraint_fn(spending_intensity_pct: float):
    """Build a horizon-aware budget constraint that tracks the reserve carry-over."""

    def constraint_fn(allocations, state, reserve):
        scaled = _apply_budget_constraint(allocations, state, spending_intensity_pct, reserve)
        total_used, base_pool = _budget_usage(allocations, state, spending_intensity_pct)
        spent = min(total_used, base_pool + reserve)
        new_reserve = max(0.0, reserve + base_pool - spent)
        return scaled, new_reserve

    return constraint_fn


def build_parameters_frame(next_year: int, allocations: dict[str, dict[str, float]]) -> pd.DataFrame:
    parameter_keys = list(ANNUAL_PARAMETERS.keys())
    rows = []
    for province in PROVINCE_SIGLAS_110:
        province_allocations = allocations.get(province, {})
        row = {'area_code': province, 'year': next_year}
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
    current_state = pd.DataFrame(payload.get('currentStateRows') or [])
    if current_state.empty:
        raise ValueError('currentStateRows is required to run the simulation bridge.')

    if 'area_code' not in current_state.columns or 'year' not in current_state.columns:
        raise ValueError('currentStateRows must contain area_code and year columns.')

    current_state = current_state.copy()
    current_state['year'] = current_state['year'].astype(int)
    current_state = current_state.sort_values('area_code').reset_index(drop=True)
    current_year = int(payload.get('currentYear') or current_state['year'].max())
    next_year = current_year + 1

    emit_progress(10, 'Loading trained twin...')
    model_path = Path(payload.get('modelPath') or REPO_ROOT / 'twin_models.joblib')
    trainer = ModelTrainer.load(model_path)
    raw_allocations = payload.get('allocations') or {}
    spending_intensity_pct = float(payload.get('spendingIntensityPct') or get_spending_intensity_pct())
    reserve_pool = float(payload.get('reservePool') or 0.0)
    constraint_fn = _make_constraint_fn(spending_intensity_pct)
    constrained_allocations, new_reserve = constraint_fn(
        raw_allocations, current_state, reserve_pool
    )
    parameters_frame = build_parameters_frame(next_year, constrained_allocations)

    indicator_columns = [column for column in current_state.columns if column not in {'area_code', 'year'}]
    simulator = DigitalTwinSimulator(trainer, indicator_columns, list(ANNUAL_PARAMETERS.keys()))
    emit_progress(50, f'Simulating year {next_year}...')
    next_state = simulator.simulate_year(current_state, parameters_frame, apply_rules=True, apply_bounds=True)
    next_state = next_state.sort_values('area_code').reset_index(drop=True)
    emit_progress(95, 'Collecting results...')

    summary_keys = ['gdp_per_capita', 'income', 'unemployment_rate', 'life_expectancy']
    summary = {}
    for key in summary_keys:
        if key in next_state.columns:
            summary[key] = float(next_state[key].mean())

    return {
        'nextYear': next_year,
        'nextStateRows': dataframe_records(next_state),
        'summary': summary,
        'reservePool': float(new_reserve),
    }


def _gdp_series_payload(base_year: int, gdp_series: list[float]) -> list[dict]:
    return [
        {'year': base_year + index + 1, 'value': float(value)}
        for index, value in enumerate(gdp_series)
    ]


def optimize_policy(payload: dict) -> dict:
    """Run a selected policy model over a multi-year horizon to maximise national GDP."""
    current_state = pd.DataFrame(payload.get('currentStateRows') or [])
    if current_state.empty:
        raise ValueError('currentStateRows is required to run the policy optimizer.')
    if 'area_code' not in current_state.columns or 'year' not in current_state.columns:
        raise ValueError('currentStateRows must contain area_code and year columns.')

    current_state = current_state.copy()
    current_state['year'] = current_state['year'].astype(int)
    current_state = current_state.sort_values('area_code').reset_index(drop=True)
    current_year = int(payload.get('currentYear') or current_state['year'].max())

    horizon = int(payload.get('horizon') or 20)
    horizon = max(1, min(horizon, 50))
    iterations = int(payload.get('iterations') or 6)
    iterations = max(1, min(iterations, 40))
    model_id = str(payload.get('modelId') or 'gdp_nn')
    spending_intensity_pct = float(payload.get('spendingIntensityPct') or get_spending_intensity_pct())
    reserve_pool = float(payload.get('reservePool') or 0.0)

    emit_progress(2, 'Loading trained twin...')
    model_path = Path(payload.get('modelPath') or REPO_ROOT / 'twin_models.joblib')
    trainer = ModelTrainer.load(model_path)

    indicator_columns = [column for column in current_state.columns if column not in {'area_code', 'year'}]
    simulator = DigitalTwinSimulator(trainer, indicator_columns, list(ANNUAL_PARAMETERS.keys()))

    param_specs = [
        ParamSpec(item['key'], item['baseline'], item['min'], item['max'])
        for item in parameter_metadata()
    ]
    constraint_fn = _make_constraint_fn(spending_intensity_pct)
    env = RolloutEnv(
        simulator=simulator,
        initial_state=current_state,
        indicator_cols=indicator_columns,
        param_specs=param_specs,
        provinces=PROVINCE_SIGLAS_110,
        horizon=horizon,
        base_year=current_year,
        constraint_fn=constraint_fn,
        reserve_pool=reserve_pool,
    )

    # Baseline (historical levers) reference trajectory for comparison.
    emit_progress(5, 'Computing baseline trajectory...')
    baseline_policy = build_policy('baseline', param_specs)
    _baseline_traj, baseline_gdp, _baseline_allocs, _baseline_reserve = env.rollout(baseline_policy)

    train_info: dict = {}
    if model_id == 'baseline':
        trajectory, gdp_series, allocations_by_year, final_reserve = (
            _baseline_traj, baseline_gdp, _baseline_allocs, _baseline_reserve
        )
        emit_progress(95, 'Collecting results...')
    else:
        policy = build_policy(model_id, param_specs)

        def progress(step: int, total: int, score: float) -> None:
            print(
                f'Training {model_id}: iteration {step}/{total} - best cumulative GDP {score:,.0f}',
                file=sys.stderr,
                flush=True,
            )
            # Training spans the 12% -> 90% window of the overall run.
            emit_progress(
                12 + 78 * step / max(total, 1),
                f'Training {model_id}: iteration {step}/{total}',
            )

        emit_progress(12, f'Training {model_id}: iteration 0/{iterations}')
        train_info = policy.fit(env, iterations=iterations, progress=progress)
        emit_progress(92, 'Rolling out optimized policy...')
        trajectory, gdp_series, allocations_by_year, final_reserve = env.rollout(policy)
        emit_progress(97, 'Collecting results...')

    final_year = current_year + horizon
    summary_keys = ['gdp_per_capita', 'income', 'unemployment_rate', 'life_expectancy']
    summary = {}
    if not trajectory.empty:
        final_state = trajectory[trajectory['year'] == final_year]
        for key in summary_keys:
            if key in final_state.columns:
                summary[key] = float(final_state[key].mean())

    # Per-province levers the model chose for the final year (loaded into the UI).
    final_allocations = allocations_by_year.get(final_year, {})

    return {
        'modelId': model_id,
        'horizon': horizon,
        'finalYear': final_year,
        'trajectoryRows': dataframe_records(trajectory),
        'baselineGdpByYear': _gdp_series_payload(current_year, baseline_gdp),
        'optimizedGdpByYear': _gdp_series_payload(current_year, gdp_series),
        'finalAllocations': final_allocations,
        'summary': summary,
        'trainInfo': train_info,
        'reservePool': float(final_reserve),
    }


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else 'bootstrap'
    raw_input = ''
    if not sys.stdin.isatty():
        raw_input = sys.stdin.read().strip()
    payload = json.loads(raw_input) if raw_input else {}

    if command == 'bootstrap':
        result = build_bootstrap(payload)
    elif command == 'province-trends':
        result = load_province_trends(payload)
    elif command == 'simulate-next-year':
        print('Loading trained model and applying one-year policy scenario...', file=sys.stderr)
        result = simulate_next_year(payload)
    elif command == 'optimize-policy':
        print('Loading trained twin and running policy model over the horizon...', file=sys.stderr)
        result = optimize_policy(payload)
    else:
        raise ValueError(f'Unknown bridge command: {command}')

    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({'error': str(error)}, allow_nan=False))
        sys.exit(1)