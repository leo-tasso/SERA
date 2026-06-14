"""Prioritarian continuum sweep (report improvement #2).

The prioritarian objective W = sum y^(1-rho) interpolates between utilitarian
(rho=0) and maximin (rho->1). This trains it at a ladder of concavities, several
seeds each, and dumps the GDP/Gini/floor trajectory across rho so the report can
*show* the continuum it otherwise only asserts.

Writes report/results/prioritarian_sweep.json.
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

import numpy as np  # noqa: E402
import backend_bridge as bridge  # noqa: E402

HORIZON = 10
ITERATIONS = 6
MODEL_ID = "neural"
RHOS = [float(r) for r in os.environ.get("SERA_RHOS", "0.0,0.25,0.5,0.75,0.95").split(",")]
SEEDS = [int(s) for s in os.environ.get("SERA_SWEEP_SEEDS", "0,1,2").split(",") if s != ""]


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
        "objectiveIds": ["prioritarian"],
    }

    points = []
    baseline = None
    for rho in RHOS:
        gdps, ginis, worsts = [], [], []
        for seed in SEEDS:
            t0 = time.time()
            print(f"=== prioritarian rho={rho} seed={seed} ===", file=sys.stderr, flush=True)
            res = bridge.compare_objectives({
                **base_payload,
                "seed": seed,
                "objectiveParams": {"prioritarian": {"rho": rho}},
            })
            if baseline is None:
                baseline = res["baseline"]
            r = res["results"][0]
            gdps.append(r["finalGdpTotal"])
            ginis.append(r["finalGini"])
            worsts.append(r["worstProvinceGdp"])
            print(f"    done in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        points.append({
            "rho": rho,
            "gdp_mean": float(np.mean(gdps)), "gdp_std": float(np.std(gdps)),
            "gini_mean": float(np.mean(ginis)), "gini_std": float(np.std(ginis)),
            "worst_mean": float(np.mean(worsts)), "worst_std": float(np.std(worsts)),
        })

    (out_dir / "prioritarian_sweep.json").write_text(json.dumps({
        "rhos": RHOS, "seeds": SEEDS, "horizon": HORIZON,
        "finalYear": current_year + HORIZON, "baseline": baseline, "points": points,
    }, indent=2))
    print("PRIORITARIAN SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
