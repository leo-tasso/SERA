"""Estimate twin structure from the panel and quantify the gap to the hand-written graph.

Runs the analysis layer in :mod:`sera.twin.panel_estimation` on the real data and
writes a findings JSON plus a human-readable summary. This is what turns the
"the ML layer doesn't carry the structure" admission into measured numbers:

* indicator->indicator sign agreement vs the hand-written graph, pooled vs with
  fixed effects (does adding the omitted entity/year contrast help?);
* honest temporal backtest vs the leaky random split the production trainer uses;
* lever data granularity (why lever response can't be learned at all).

Usage:
    .venv/Scripts/python.exe tools/ad_hoc/estimate_twin_structure.py
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "ui"))

import numpy as np  # noqa: E402

from sera.config import DATA_DIR  # noqa: E402
from sera.twin import panel_estimation as pe  # noqa: E402
from backend_bridge import INDICATORS  # noqa: E402

OUT = REPO_ROOT / "docs" / "twin_structure_findings.json"


def run_level(level: str) -> dict:
    panel = pe.load_panel(DATA_DIR, INDICATORS, level=level)
    learned_fe = pe.learn_couplings(panel, fe=True)
    learned_pool = pe.learn_couplings(panel, fe=False)
    ag_fe = pe.coupling_agreement(learned_fe)
    ag_pool = pe.coupling_agreement(learned_pool)
    bt_pool = pe.temporal_backtest(panel, fe=False)  # matches production spec
    bt_fe = pe.temporal_backtest(panel, fe=True)
    contradicted = [
        {"source": s, "target": t, "documented_sign": int(w), "data_coef": round(c, 4)}
        for s, t, w, c in ag_fe["per_edge"]
        if np.sign(c) != w
    ]
    return {
        "entities": int(panel["entity"].nunique()),
        "years": int(panel["year"].nunique()),
        "rows": int(len(panel)),
        "sign_agreement_fe": ag_fe["sign_agreement"],
        "sign_agreement_pooled": ag_pool["sign_agreement"],
        "edges_scored": ag_fe["edges_scored"],
        "edge_precision_fe": ag_fe["edge_precision"],
        "edge_recall_fe": ag_fe["edge_recall"],
        "temporal_r2_pooled": bt_pool["temporal_r2_median"],
        "random_r2_pooled": bt_pool["random_r2_mean"],
        "temporal_r2_fe": bt_fe["temporal_r2_median"],
        "direction_accuracy": bt_pool["direction_accuracy_mean"],
        "cutoff_year": bt_pool["cutoff_year"],
        "contradicted_edges": contradicted,
    }


def main() -> None:
    levers = pe.lever_data_granularity(DATA_DIR)
    reviewed = pe.reviewed_coupling_signs(DATA_DIR, INDICATORS, level="province")
    findings = {
        "lever_granularity": {
            "levers": levers["levers"],
            "national_or_regional_only": levers["national_or_regional_only"],
            "province_measured": levers["province_measured"],
        },
        "wiring": {  # what was changed in the simulator from the panel review
            "edges_total": sum(len(v) for v in reviewed["signs"].values()),
            "flipped": reviewed["flipped"],
            "dropped": reviewed["dropped"],
            "flipped_count": len(reviewed["flipped"]),
            "dropped_count": len(reviewed["dropped"]),
        },
        "province": run_level("province"),
        "region": run_level("region"),
    }
    OUT.write_text(json.dumps(findings, indent=2))

    print("\n=== TWIN STRUCTURE: estimated vs hand-written ===\n")
    lg = findings["lever_granularity"]
    print(f"Levers: {lg['national_or_regional_only']}/{lg['levers']} are national/regional-only "
          f"(0..{lg['province_measured']} province-measured) -> provincial lever response is unidentifiable.\n")
    for level in ("province", "region"):
        r = findings[level]
        print(f"[{level}] {r['entities']} entities x {r['years']} years")
        print(f"  indicator->indicator SIGN agreement vs hand-written graph "
              f"({r['edges_scored']} in-panel edges):")
        print(f"      pooled (production spec) {r['sign_agreement_pooled']:.0%}   "
              f"with fixed effects {r['sign_agreement_fe']:.0%}")
        print(f"  edge-set recovery: precision {r['edge_precision_fe']:.2f}  recall {r['edge_recall_fe']:.2f}")
        print(f"  honest validation: temporal-holdout R2 {r['temporal_r2_pooled']:+.2f} "
              f"vs leaky random-split R2 {r['random_r2_pooled']:+.2f}; "
              f"direction accuracy {r['direction_accuracy']:.0%} (cutoff {r['cutoff_year']})")
        if r["contradicted_edges"]:
            print("  hand-written edges the data contradicts:")
            for e in r["contradicted_edges"]:
                print(f"      {e['source']} -> {e['target']}: documented "
                      f"{'+' if e['documented_sign'] > 0 else '-'}, data {e['data_coef']:+.3f}")
        print()
    print(f"Findings written to {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
