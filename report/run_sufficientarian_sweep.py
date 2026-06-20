"""Sufficientarian threshold sweep (report improvement #4).

The sufficientarian objective minimizes the shortfall below a threshold theta
(set as a fraction of the starting median province). theta *is* the theory, so
this sweeps where the "enough" line sits and records how the optimizer's
GDP/Gini/floor respond --- the parallel to the prioritarian rho sweep, and the
only thing that exercises the second tunable objective.

Writes report/results/sufficientarian_sweep.json.
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
THETAS = [float(t) for t in os.environ.get("SERA_THETAS", "0.5,0.65,0.8,0.95,1.1").split(",")]
SEEDS = [int(s) for s in os.environ.get("SERA_SUFF_SEEDS", "0,1,2").split(",") if s != ""]


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
        "objectiveIds": ["sufficientarian"],
    }

    points = []
    baseline = None
    for theta in THETAS:
        gdps, ginis, worsts = [], [], []
        for seed in SEEDS:
            t0 = time.time()
            print(f"=== sufficientarian theta={theta} seed={seed} ===", file=sys.stderr, flush=True)
            res = bridge.compare_objectives(
                {
                    **base_payload,
                    "seed": seed,
                    "objectiveParams": {"sufficientarian": {"threshold_ratio": theta}},
                }
            )
            if baseline is None:
                baseline = res["baseline"]
            r = res["results"][0]
            gdps.append(r["finalGdpTotal"])
            ginis.append(r["finalGini"])
            worsts.append(r["worstProvinceGdp"])
            print(f"    done in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        points.append(
            {
                "theta": theta,
                "gdp_mean": float(np.mean(gdps)),
                "gini_mean": float(np.mean(ginis)),
                "worst_mean": float(np.mean(worsts)),
            }
        )

    (out_dir / "sufficientarian_sweep.json").write_text(
        json.dumps(
            {
                "thetas": THETAS,
                "seeds": SEEDS,
                "horizon": HORIZON,
                "finalYear": current_year + HORIZON,
                "baseline": baseline,
                "points": points,
            },
            indent=2,
        )
    )
    print("SUFFICIENTARIAN SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
