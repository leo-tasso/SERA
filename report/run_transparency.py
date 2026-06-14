"""Price-of-transparency experiment (report improvement #1).

The report sells the policy-model spectrum on the claim that the black-box vs.
white-box score gap is a *measured* price of transparency, not an assumed one ---
but the headline experiments only ever use the neural policy. This trains every
trainable model under one fixed objective (utilitarian) and the same budget,
several seeds each, and records the cumulative objective score and equity
metrics next to each model's explainability badge, so the price is finally on
the table.

Writes report/results/transparency.json.
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
OBJECTIVE = "utilitarian"
# Every trainable model, spanning the explainability spectrum.
MODELS = ["neural", "linear", "rules", "cluster_cem", "uniform_cem", "uniform_bayes"]
SEEDS = [int(s) for s in os.environ.get("SERA_TRANSPARENCY_SEEDS", "0,1,2").split(",") if s != ""]


def main() -> None:
    out_dir = REPORT_DIR / "results"
    out_dir.mkdir(exist_ok=True)

    state = bridge.load_initial_state(bridge.DATA_DIR, bridge.INDICATORS, 2025)
    state = state.sort_values("area_code").reset_index(drop=True)
    current_year = int(state["year"].max())
    rows = state.to_dict("records")

    results = []
    for model_id in MODELS:
        scores, gdps, ginis, worsts, imprs = [], [], [], [], []
        label = explain = None
        for seed in SEEDS:
            t0 = time.time()
            print(f"=== {model_id} seed={seed} ===", file=sys.stderr, flush=True)
            res = bridge.optimize_policy({
                "currentStateRows": rows, "currentYear": current_year,
                "horizon": HORIZON, "iterations": ITERATIONS,
                "modelId": model_id, "objectiveId": OBJECTIVE, "seed": seed,
            })
            info = res.get("trainInfo") or {}
            explain = res.get("explainability") or explain
            full = next((c for c in res.get("candidates", []) if c.get("id") == "full"), None)
            if info.get("best_score") is not None:
                scores.append(float(info["best_score"]))
                imprs.append(float(info.get("improvement_pct", 0.0)))
            if full:
                label = res.get("objectiveLabel")
                if full.get("finalGdpTotal") is not None:
                    gdps.append(float(full["finalGdpTotal"]))
                    ginis.append(float(full["finalGini"]))
                    worsts.append(float(full["worstProvinceGdp"]))
            print(f"    done in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        results.append({
            "modelId": model_id,
            "explainability": explain,
            "score_mean": float(np.mean(scores)) if scores else None,
            "score_std": float(np.std(scores)) if scores else None,
            "improvement_mean": float(np.mean(imprs)) if imprs else None,
            "gdp_mean": float(np.mean(gdps)) if gdps else None,
            "gini_mean": float(np.mean(ginis)) if ginis else None,
            "worst_mean": float(np.mean(worsts)) if worsts else None,
        })

    (out_dir / "transparency.json").write_text(json.dumps({
        "objective": OBJECTIVE, "horizon": HORIZON, "iterations": ITERATIONS,
        "seeds": SEEDS, "finalYear": current_year + HORIZON, "results": results,
    }, indent=2))
    print("TRANSPARENCY DONE", flush=True)


if __name__ == "__main__":
    main()
