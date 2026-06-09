# SERA — Italy Digital Twin

SERA is a digital twin of the Italian provinces. It has three parts:

1. **Data downloader** — fetches ~88 historical socioeconomic indicators (2001-2025) for the 110 Italian provinces from the ISTAT SDMX API into `data/`.
2. **Twin engine** (`sera.twin`) — trains one ML model per indicator and simulates provincial indicators forward year by year under user-chosen policy levers, combining the trained models with a hand-written causal graph.
3. **Control-room UI** (`ui/`) — an Electron + React dashboard with a clickable province map, per-province policy allocators, a national budget meter, and an AI policy optimizer that picks levers to maximize national GDP over a multi-year horizon.

## Setup

Python 3.11, virtual environment at `.venv`:

```powershell
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt # or: pip install -e .
```

## Data downloader

```powershell
$env:PYTHONPATH = "$pwd\src"

# One indicator
python src/sera/downloader.py --indicator population --start-year 2001 --end-year 2025

# Everything (88 indicators; slow — see rate limits below)
python src/sera/downloader.py --indicator all
```

Output goes to `data/<category>/<indicator>/<indicator>_raw_<start>_<end>.csv`, with a
`.mapping.json` next to each CSV documenting the source dataflow and column mapping.

### ISTAT rate limits and caching

- ISTAT enforces a hard limit of **5 requests/minute per IP**; exceeding it can get the IP
  banned for 1-2 days (`ISTAT_RATE_LIMIT_PER_MINUTE` in `sera.config`).
- The client stays well under that with a **25-second minimum delay** between requests
  (`ISTAT_MIN_DELAY_SECONDS`), counting failed requests too.
- Responses are cached in `data/cache/`, keyed by dataflow, dimensional key, and year range.
- Known ISTAT API quirk (documented across all ISTAT sources): the `endPeriod` parameter
  returns year+1, so the client subtracts 1 from the requested end year.

## Twin engine

Train the per-indicator models and run a baseline simulation:

```powershell
$env:PYTHONPATH = "$pwd\src"
python -m sera.twin.cli --mode train-and-simulate --model-type ridge --sim-years 5
```

This writes `twin_models.joblib` (used by the UI) and `simulation_results.csv`.
Useful flags: `--mode train|simulate|train-and-simulate`, `--baseline-year`,
`--initial-state <csv>`, `--parameters-file <csv>`.

How a simulated year works (`sera.twin.simulator`):

1. Each indicator's trained model predicts from lagged indicators + policy levers; the
   prediction is anchored to last year's value and only the *policy signal* (prediction under
   chosen levers vs. baseline levers) is applied.
2. Hand-written causal rules from `sera.twin.causal_graph` add a second, deliberate layer of
   lever→indicator elasticity (see `CAUSAL_RULE_STRENGTH`).
3. Inter-indicator effects propagate (e.g. income → poverty), bounds are enforced, and a
   realism speed limit caps year-over-year change of any indicator at ±6%.

`sera.twin.policy` defines the pluggable policy models used by the UI's optimizer:
`baseline` (historical levers) and `gdp_nn`, a small NumPy MLP trained with evolution
strategies to maximize cumulative national GDP, subject to the national budget constraint
(unspent budget carries over as a reserve).

## UI

```powershell
cd ui
npm install
npm start
```

Requirements: `twin_models.joblib` at the repo root (train it first), the `data/` directory,
and `ui/province_provinces.geojson` (committed). All JS libraries (React, Chart.js, Babel)
are vendored in `ui/vendor/` — the app makes **no network requests at runtime**.

The Electron main process talks to Python through `ui/backend_bridge.py` (one process per
command: `bootstrap`, `province-trends`, `simulate-next-year`, `optimize-policy`), streaming
progress over stderr.

## Tests

```powershell
python -m pytest tests -q
```

## Repository layout

- `src/sera/config.py` — paths, ISTAT constants, indicator→category map
- `src/sera/istat_client.py` — rate-limited, cached SDMX client
- `src/sera/downloader.py` + `src/sera/downloaders/<category>/<indicator>.py` — CLI + one module per indicator
- `src/sera/twin/` — data loading, model training, causal graph, simulator, policy models, CLI
- `ui/` — Electron app (main.js, preload.js, renderer.js, backend_bridge.py)
- `data/` — downloaded indicator CSVs (110 provinces, 2-letter sigle)
- `tools/ad_hoc/` — one-off validation and demo scripts
