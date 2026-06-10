"""Smoke-check the optimize-policy candidate response (reads JSON on stdin)."""

import json
import sys

d = json.load(sys.stdin)
print('model:', d['modelId'], '| objective:', d['objectiveId'], '| finalYear:', d['finalYear'])
print('sensitivity scales:', d['sensitivityScales'])
for c in d['candidates']:
    band = c.get('gdpBandByYear')
    band_txt = 'band=none'
    if band:
        widths = [b['high'] - b['low'] for b in band]
        band_txt = 'band_width_final={:,.0f}'.format(widths[-1])
    print(
        '{:<10} gdp={:,.0f} gini={:.4f} worst={:,.0f} reserve={:,.0f} rows={} {}'.format(
            c['id'], c['finalGdpTotal'], c['finalGini'], c['worstProvinceGdp'],
            c['reservePool'], len(c['trajectoryRows']), band_txt,
        )
    )
    assert c['finalAllocations'], f"candidate {c['id']} has no final allocations"
print('OK')
