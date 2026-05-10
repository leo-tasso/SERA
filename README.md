# SERA Data Downloader

Download historical data for the Italy Digital Twin project, starting with population indicators from ISTAT.

## Setup

### Python Environment (3.11)

This project uses Python 3.11. Set up a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Alternatively, using `pyproject.toml`:

```bash
pip install -e .
```

## Usage

### Download Population Data

Download resident population by province and region for 2001-2025:

```bash
# Set Python path to include src directory
$env:PYTHONPATH = "$pwd\src"

# Download population (2001-2025, or adjust years)
python src/sera/downloader.py --indicator population --start-year 2001 --end-year 2025

# Download specific year range
python src/sera/downloader.py --indicator population --start-year 2020 --end-year 2023

# Save to custom location
python src/sera/downloader.py --indicator population --output my_data.csv
```

### Output

Data is saved as CSV in `data/` directory:
- `population_raw_YYYY_YYYY.csv` - Raw population data by province and region

### Data Structure

Population CSV columns:
- `area_code`: ISTAT geographic code (IT for national, ITxxx for provinces/regions)
- `frequency`: Data frequency (A = annual)
- `year`: Year of observation
- `population`: Resident population count
- `sex`: Sex code (1 = total)
- `age_group`: Age group (TOTAL = all ages)
- `marital_status`: Marital status code (1 = total)

## Architecture

### Modules

- `sera.config`: Configuration and constants
- `sera.istat_client`: ISTAT SDMX API client with rate limiting
- `sera.downloaders.population`: Population indicator downloader
- `sera.downloader`: Main CLI entry point

### Rate Limiting

The downloader respects ISTAT's API rate limits:
- **Max**: 5 requests/minute per IP
- **Minimum delay**: 12 seconds between requests
- **Data caching**: Responses are cached locally to minimize API calls

## Data Sources

### Population (Indicator 1)
- **Source**: ISTAT (Istituto Nazionale di Statistica)
- **API**: SDMX Web Services (REST)
- **Endpoint**: https://esploradati.istat.it/SDMXWS
- **Dataflow**: `22_289_DF_DCIS_POPRES1_1`
- **Coverage**: 
  - National level (IT)
  - Regional level (IT regions)
  - Provincial level (~107 provinces)
- **Time span**: 2019-2025 (available from endpoint)
- **Frequency**: Annual (January snapshot)

## Implementation Notes

### Constraints & Workarounds

1. **ISTAT Rate Limiting**: The API enforces 5 requests/minute. Exceeding this causes IP bans (1-2 days recovery).
   - Mitigation: 12-second minimum delay between requests
   - Local caching to avoid redundant API calls

2. **ISTAT API Bug**: The `endPeriod` parameter returns year+1 data.
   - Workaround: We subtract 1 from the requested end year

3. **Data Availability**: The population endpoint only has data from 2019 onwards.
   - Reason: This dataflow version was introduced in 2019
   - Solution: Check alternative ISTAT dataflows for older data (2001+)

## Next Steps

1. **Additional Indicators**: Implement downloaders for other 108 indicators (birth rate, GDP, unemployment, etc.)
2. **Historical Data**: Source older population data (pre-2019) from alternative ISTAT dataflows
3. **Data Validation**: Add reconciliation with multiple sources to ensure data quality
4. **Parameters**: Download government policy parameters and exogenous drivers

## Files Generated

- `pyproject.toml`: Python project configuration (Python 3.11)
- `src/sera/`: Main downloader package
- `data/population_raw_*.csv`: Downloaded population data

## Testing

Quick test with recent years only:

```bash
$env:PYTHONPATH = "$pwd\src"
python src/sera/downloader.py --indicator population --start-year 2024 --end-year 2025
```

This downloads ~274 records in seconds and validates the complete pipeline.
