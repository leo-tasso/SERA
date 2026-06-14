"""Propagation-mode ablation (report improvement #3).

Quantifies how much the data-reviewed inter-indicator signs actually change the
results, by re-running the headline comparison under three propagation modes:

* ``signless``   --- the original implementation (every target moves with its source);
* ``documented`` --- polarity-derived signs, no panel review;
* ``reviewed``   --- documented signs corrected by the panel (what ships).

Writes report/results/ablation.json.
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

HORIZON = 10
ITERATIONS = 6
MODEL_ID = "neural"
MODES = ["signless", "documented", "reviewed"]
OBJECTIVE_IDS = ["utilitarian", "rawlsian", "cvar", "egalitarian", "wellbeing"]
SEEDS = [int(s) for s in os.environ.get("SERA_ABL_SEEDS", "0,1,2").split(",") if s != ""]


def main() -> None:
    out_dir = REPORT_DIR / "results"
    out_dir.mkdir(exist_ok=True)

    state = bridge.load_initial_state(bridge.DATA_DIR, bridge.INDICATORS, 2025)
    state = state.sort_values("area_code").reset_index(drop=True)
    current_year = int(state["year"].max())
    base_payload = {
        "currentStateRows": state.to_dict("records"),
        "currentYear": current_year,
        "horizon": HORIZON,
        "iterations": ITERATIONS,
        "modelId": MODEL_ID,
        "objectiveIds": OBJECTIVE_IDS,
    }

    by_mode = {}
    for mode in MODES:
        baseline = None
        per_seed = []
        for seed in SEEDS:
            t0 = time.time()
            print(f"=== mode={mode} seed={seed} ===", file=sys.stderr, flush=True)
            res = bridge.compare_objectives({**base_payload, "seed": seed, "propagationMode": mode})
            if baseline is None:
                baseline = res["baseline"]
            per_seed.append({"seed": seed, "results": res["results"]})
            print(f"    done in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        by_mode[mode] = {"baseline": baseline, "perSeed": per_seed}

    (out_dir / "ablation.json").write_text(json.dumps({
        "modes": MODES, "seeds": SEEDS, "objectiveIds": OBJECTIVE_IDS,
        "horizon": HORIZON, "finalYear": current_year + HORIZON, "byMode": by_mode,
    }, indent=2))
    print("ABLATION DONE", flush=True)


if __name__ == "__main__":
    main()
