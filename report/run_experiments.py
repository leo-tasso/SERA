"""Run the ethical-objective comparison and Pareto-front experiments for the report.

Calls the same bridge functions the UI uses (compare-objectives, pareto-front)
from the latest historical state, and dumps the raw JSON next to this script so
the plotting script can build the report figures.

What changed relative to the single-run version:

* **Multi-seed replication** (improvement #1): every framework is trained under
  several seeds so the report can show mean +/- spread and separate genuine
  framework effects from gradient-free search luck.
* **CVaR vs hard maximin** (improvement #2): the smoothed-maximin objective is
  included alongside Rawlsian maximin to test whether maximin's poor floor is
  the theory or just its sparse one-province training signal.
* **Clustered Pareto** (improvement #4): a second NSGA-II run over per-cluster
  lever vectors, to see whether province targeting recovers the real
  efficiency--equity trade-off that uniform national policy cannot express.
* **Indicator provenance** (improvement #7): the measured-vs-disaggregated label
  of every panel indicator is dumped so the report can quantify how much of the
  equity result rests on real provincial variation.
"""

import json
import os
import sys
import time
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPORT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "ui"))

import backend_bridge as bridge  # noqa: E402
from sera.twin.data_loader import DataLoader  # noqa: E402

HORIZON = 10  # simulated years
ITERATIONS = 6  # training iterations per objective (compare) / generations (pareto)
PARETO_POPSIZE = 12
PARETO_CLUSTERS = 4  # per-cluster Pareto: how many regional packages to evolve
MODEL_ID = "neural"  # the equity dashboard's default policy model

# Multi-seed replication. Override with SERA_SEEDS=... in the environment.
# Six seeds give usable bootstrap confidence intervals on the framework deltas
# at a tractable runtime; raise for tighter intervals.
SEEDS = [int(s) for s in os.environ.get("SERA_SEEDS", "0,1,2,3,4,5").split(",") if s != ""]

# The frameworks compared head-to-head. CVaR sits next to Rawlsian so the
# report can test the "maximin under-optimizes its own floor" hypothesis.
OBJECTIVE_IDS = ["utilitarian", "rawlsian", "cvar", "egalitarian", "wellbeing"]


def main() -> None:
    out_dir = REPORT_DIR / "results"
    out_dir.mkdir(exist_ok=True)

    print("Loading latest historical state...", file=sys.stderr, flush=True)
    state = bridge.load_initial_state(bridge.DATA_DIR, bridge.INDICATORS, 2025)
    state = state.sort_values("area_code").reset_index(drop=True)
    current_year = int(state["year"].max())
    print(f"State: {len(state)} provinces, year {current_year}", file=sys.stderr, flush=True)

    base_payload = {
        "currentStateRows": state.to_dict("records"),
        "currentYear": current_year,
        "horizon": HORIZON,
        "iterations": ITERATIONS,
        "modelId": MODEL_ID,
        "objectiveIds": OBJECTIVE_IDS,
    }

    # ---- Indicator provenance (improvement #7) ------------------------------ #
    print("=== indicator provenance ===", file=sys.stderr, flush=True)
    loader = DataLoader(bridge.DATA_DIR)
    provenance = loader.panel_provenance(bridge.INDICATORS)
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    # ---- Multi-seed framework comparison (improvements #1, #2) -------------- #
    baseline = None
    per_seed = []
    for seed in SEEDS:
        t0 = time.time()
        print(f"=== compare-objectives (seed {seed}) ===", file=sys.stderr, flush=True)
        comparison = bridge.compare_objectives({**base_payload, "seed": seed})
        print(
            f"seed {seed} done in {time.time() - t0:.0f}s",
            file=sys.stderr,
            flush=True,
        )
        if baseline is None:
            baseline = comparison["baseline"]  # seed-independent (no training)
        per_seed.append({"seed": seed, "results": comparison["results"]})

    comparison_out = {
        "modelId": MODEL_ID,
        "horizon": HORIZON,
        "finalYear": current_year + HORIZON,
        "seeds": SEEDS,
        "objectiveIds": OBJECTIVE_IDS,
        "baseline": baseline,
        "perSeed": per_seed,
    }
    (out_dir / "compare_objectives.json").write_text(json.dumps(comparison_out, indent=2))

    # ---- Pareto frontiers: uniform vs clustered (improvement #4) ------------ #
    t1 = time.time()
    print("=== pareto-front (uniform) ===", file=sys.stderr, flush=True)
    uniform = bridge.pareto_front({**base_payload, "popsize": PARETO_POPSIZE})
    print(f"uniform pareto done in {time.time() - t1:.0f}s", file=sys.stderr, flush=True)
    (out_dir / "pareto_front.json").write_text(json.dumps(uniform, indent=2))

    t2 = time.time()
    print("=== pareto-front (clustered) ===", file=sys.stderr, flush=True)
    clustered = bridge.pareto_front(
        {**base_payload, "popsize": PARETO_POPSIZE, "nClusters": PARETO_CLUSTERS}
    )
    print(f"clustered pareto done in {time.time() - t2:.0f}s", file=sys.stderr, flush=True)
    (out_dir / "pareto_front_clustered.json").write_text(json.dumps(clustered, indent=2))

    print("ALL EXPERIMENTS DONE", flush=True)


if __name__ == "__main__":
    main()
