"""Smoke-test the explainable policy models and the Pareto bridge end to end.

Runs against the real trained twin (twin_models.joblib): one tiny
optimize-policy run per new model (checking the explanation payload is
JSON-serialisable) and one tiny pareto-front run.

Usage:  python tools/ad_hoc/check_explainable_policies.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "ui"))

import backend_bridge  # noqa: E402


def main() -> None:
    bootstrap = backend_bridge.build_bootstrap({})
    state_rows = bootstrap["latestStateRows"]
    base_payload = {
        "currentStateRows": state_rows,
        "currentYear": bootstrap["baselineYear"],
        "horizon": 2,
        "iterations": 1,
    }

    for model_id in ["linear", "rules", "cluster_cem", "uniform_bayes", "neural"]:
        print(f"--- optimize-policy: {model_id} ---", flush=True)
        result = backend_bridge.optimize_policy({**base_payload, "modelId": model_id})
        explanation = result.get("explanation")
        assert result["explainability"] is not None, model_id
        assert explanation is not None, f"{model_id}: no explanation"
        json.dumps(result, allow_nan=False)  # must be JSON-clean for the bridge
        print(
            f"OK  explainability={result['explainability']} "
            f"explanation_type={explanation['type']} "
            f"candidates={len(result['candidates'])}"
        )

    print("--- pareto-front ---", flush=True)
    pareto = backend_bridge.pareto_front({**base_payload, "popsize": 4})
    json.dumps(pareto, allow_nan=False)
    tags = {tag for point in pareto["points"] for tag in point["tags"]}
    print(
        f"OK  points={len(pareto['points'])} evaluations={pareto['evaluations']} "
        f"tags={sorted(tags)}"
    )


if __name__ == "__main__":
    main()
