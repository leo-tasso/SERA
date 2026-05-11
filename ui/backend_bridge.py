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
from sera.twin.province_mapping import PROVINCE_SIGLAS_110
from sera.twin.simulator import DigitalTwinSimulator

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

    model_path = Path(payload.get('modelPath') or REPO_ROOT / 'twin_models.joblib')
    trainer = ModelTrainer.load(model_path)
    parameters_frame = build_parameters_frame(next_year, payload.get('allocations') or {})

    indicator_columns = [column for column in current_state.columns if column not in {'area_code', 'year'}]
    simulator = DigitalTwinSimulator(trainer, indicator_columns, list(ANNUAL_PARAMETERS.keys()))
    next_state = simulator.simulate_year(current_state, parameters_frame, apply_rules=True, apply_bounds=True)
    next_state = next_state.sort_values('area_code').reset_index(drop=True)

    summary_keys = ['gdp_per_capita', 'income', 'unemployment_rate', 'life_expectancy']
    summary = {}
    for key in summary_keys:
        if key in next_state.columns:
            summary[key] = float(next_state[key].mean())

    return {
        'nextYear': next_year,
        'nextStateRows': dataframe_records(next_state),
        'summary': summary,
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