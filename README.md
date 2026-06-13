<p align="center">
  <img src="ui/assets/icon.svg" width="128" alt="SERA icon — a sad sun setting behind a frowning horizon" />
</p>

# SERA — Italy Digital Twin

SERA is a digital twin of the Italian provinces. It has three parts:

1. **Data downloader** — fetches ~88 historical socioeconomic indicators (2001-2025) for the 110 Italian provinces from the ISTAT SDMX API into `data/`.
2. **Twin engine** (`sera.twin`) — trains one ML model per indicator and simulates provincial indicators forward year by year under user-chosen policy levers, combining the trained models with a hand-written causal graph.
3. **Control-room UI** (`ui/`) — an Electron + React dashboard with a clickable province map, per-province policy allocators, a national budget meter, and an AI policy optimizer that picks levers over a multi-year horizon to maximize a user-selected **ethical objective** (utilitarian GDP, Rawlsian maximin, egalitarian Sen welfare, or multi-indicator wellbeing).

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

`sera.twin.policy` defines the pluggable policy models used by the UI's optimizer,
spanning a deliberate **explainability spectrum** (each model carries a badge and an
`explain()` artifact the UI renders next to its candidates):

- `baseline` — historical levers (white box, trivially)
- `neural` — a small NumPy MLP trained with evolution strategies, emitting per-province
  levers (black box; audited post hoc via `sera.twin.explain`: permutation importance +
  a distilled surrogate tree with an honest fidelity score)
- `linear` — the same per-province setup with no hidden layer, same ES training; the
  signed weight matrix is the explanation, so the score gap vs. `neural` is a measured
  price of transparency (white box)
- `rules` — one IF/THEN threshold rule per lever, found with CEM; the whole policy
  prints as sentences (white box)
- `cluster_cem` — provinces k-means-clustered on their indicators, one lever vector per
  cluster: regional policy packages (white box)
- `uniform_cem` — one shared national lever vector found with the cross-entropy method
  (white box)
- `uniform_bayes` — the same shared vector found with Gaussian-process Bayesian
  optimization: far fewer twin rollouts, plus per-lever partial-dependence curves with
  the GP's own uncertainty (gray box)

All run subject to the national budget
constraint: spending levers consume the pool, **tax levers fund it** (cutting taxes
below baseline shrinks the budget available for programs), and unspent budget carries
over as a reserve. Effect sizes (`CAUSAL_RULE_STRENGTH`, `POLICY_SIGNAL_CAP`) are
calibrated so no single lever family can push a province to the growth cap on its own —
policies face real fiscal trade-offs, which is what lets different ethical objectives
reach genuinely different optima.

What the optimizer maximizes is a separate, explicit choice: `sera.twin.objectives`
defines pluggable **ethical objectives** that score each simulated year, and any
trainable model can be paired with any objective:

- `utilitarian` — total national GDP (sum across provinces; the classic default)
- `rawlsian` — maximin: only the worst-off province counts (Rawls' difference principle)
- `cvar` — smoothed maximin: the mean of the worst `alpha` fraction of provinces
  (a denser-signal Rawlsian cousin; `alpha` is a slider)
- `prioritarian` — sum of a concave transform `y^(1−rho)` of provincial GDP; the
  concavity `rho` sweeps the continuum from utilitarian (`rho`=0) toward maximin
- `egalitarian` — Sen welfare: total GDP × (1 − Gini across provinces)
- `sufficientarian` — negative shortfall below a threshold set relative to the
  starting median province (the threshold is a slider, not a hard-coded constant)
- `wellbeing` — multi-indicator composite (GDP, life expectancy, unemployment, poverty)
  measured as percent change from the starting year

Parameterized objectives expose their tunable value (`alpha`, `rho`, threshold) as a
slider in the UI next to the objective dropdown, so the value judgment stays explicit.

The UI exposes both choices (model and objective) in the header control panel and plots the
objective's welfare trajectory against the baseline scenario, so the distributional
consequences of each ethical framework are directly comparable.

The **ethics equity dashboard** goes one step further: the `compare-objectives` bridge
command trains the selected model once per ethical framework from the same starting
state (a read-only what-if — the twin is never advanced) and the UI compares the
outcomes side by side: total GDP (efficiency), inter-provincial Gini (inequality), the
worst-off province's GDP (the floor), and per-objective "who gains, who loses" maps of
final-year provincial GDP versus the baseline scenario.

The **efficiency–equity frontier** (`sera.twin.pareto`, `pareto-front` bridge command)
drops the framework dropdown entirely: an NSGA-II search evolves uniform national lever
vectors against total GDP, inter-provincial Gini, and the worst-off province
*simultaneously* and the UI plots the resulting Pareto frontier — every point a
non-dominated policy whose levers can be inspected, with the utilitarian, egalitarian,
and Rawlsian optima tagged as corners of the same curve. Read-only, like the dashboard.

**Human oversight & uncertainty.** The optimizer never applies anything automatically:
each run returns three graded candidates — full intervention, moderate (levers halfway
back toward baseline, via `BlendedPolicy`), and the historical baseline — with their
efficiency/equity trade-offs, and the user explicitly adopts one (or none). Trained
candidates also carry a **sensitivity band**: the same policy re-simulated with the
hand-written causal rules at 0.5× and 1.5× strength (`causal_rule_strength` on
`DigitalTwinSimulator`), so the UI shows how much of the projection rests on hand-tuned
assumptions.

## Ethics & documentation

SERA is decision *support*, not decision *making*, and ships the documentation a
responsible AI system needs:

- [ETHICS.md](ETHICS.md) — intended use, embedded value judgments, affected groups,
  EU AI Act positioning
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md) — model card for the twin and policy models,
  including honest validation status
- [docs/DATASHEET.md](docs/DATASHEET.md) — datasheet for the ISTAT dataset, including
  who is missing from the data

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
command: `bootstrap`, `province-trends`, `simulate-next-year`, `optimize-policy`,
`compare-objectives`, `pareto-front`), streaming progress over stderr.

## Tests

```powershell
python -m pytest tests -q
```

## Repository layout

- `src/sera/config.py` — paths, ISTAT constants, indicator→category map
- `src/sera/istat_client.py` — rate-limited, cached SDMX client
- `src/sera/downloader.py` + `src/sera/downloaders/<category>/<indicator>.py` — CLI + one module per indicator
- `src/sera/twin/` — data loading, model training, causal graph, simulator, the national
  budget constraint (`budget.py`), policy models, ethical objectives, post-hoc explanations
  (`explain.py`), Pareto frontier search (`pareto.py`, uniform or per-cluster), CLI
- `ui/` — Electron app (main.js, preload.js, renderer.js, backend_bridge.py)
- `data/` — downloaded indicator CSVs (110 provinces, 2-letter sigle)
- `tools/ad_hoc/` — one-off validation and demo scripts
