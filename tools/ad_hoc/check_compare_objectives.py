"""Smoke-check the compare-objectives bridge command output (reads JSON on stdin)."""

import json
import sys

d = json.load(sys.stdin)
b = d["baseline"]
print(
    "baseline     gdp={:,.0f} gini={:.4f} worst={:,.0f} provinces={}".format(
        b["finalGdpTotal"], b["finalGini"], b["worstProvinceGdp"], len(b["finalGdpByProvince"])
    )
)
for r in d["results"]:
    print(
        "{:<12} gdp={:,.0f} gini={:.4f} worst={:,.0f} welfare_pts={} gini_pts={} provinces={}".format(
            r["objectiveId"],
            r["finalGdpTotal"],
            r["finalGini"],
            r["worstProvinceGdp"],
            len(r["welfareByYear"]),
            len(r["giniByYear"]),
            len(r["finalGdpByProvince"]),
        )
    )
