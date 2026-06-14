# Twin structure: estimated vs. hand-written

This note quantifies how much of the digital twin's causal structure is backed
by the historical panel, versus asserted by hand. It is the honest-validation
companion to the model card. Reproduce with:

```powershell
.venv\Scripts\python.exe tools\ad_hoc\estimate_twin_structure.py
```

which writes [`twin_structure_findings.json`](twin_structure_findings.json) and
prints the summary below. The estimation layer is
[`sera.twin.panel_estimation`](../src/sera/twin/panel_estimation.py); it is an
analysis layer only and does not change the production twin.

## What was measured

The production trainer fits one pooled ridge per indicator on **all 20 national
levers + a lag**, with **no entity/year fixed effects** and a **random** train/
test split. Three things follow that this analysis pins down with numbers.

### 1. The policy levers carry no provincial variation (so their effect can't be learned)

All **20/20 levers are published only at national (or, after spreading, regional)
granularity** — zero are measured at province level. The per-province lever
values the trainer sees are pure population-share artifacts of disaggregation,
mutually collinear and uninformative about policy. **Provincial lever→indicator
response is unidentifiable from this data at any granularity** — not a tuning
problem, a data-availability fact. (Independently corroborated by the trained
models: the ridge coefficients agree with the documented lever direction only
~54% of the time — a coin flip — and overshoot persistence ~10×, which is what
saturates the ±3% policy-signal cap.)

### 2. The indicator→indicator graph IS mostly data-backed — and fixed effects matter

The inter-indicator couplings (income→poverty, unemployment→crime, …), which the
data *can* support, are nonetheless hand-written in
`causal_graph.INDICATOR_TO_INDICATORS`. Re-estimated from the panel as a
fixed-effects panel regression, the **directions agree with the hand-written
graph 76%** of the in-panel edges — well above chance, and a reassuring vote of
confidence in the domain knowledge. Crucially, adding the entity/year fixed
effects the production model omits **raises agreement from ~59% (pooled) to 76%**:
the omitted contrast is part of why the production ML looks like noise.

| Granularity | sign agreement (pooled) | sign agreement (fixed effects) |
|---|---|---|
| Province (110) | 59% | **76%** |
| Region (20) | 65% | **76%** |

The *exact topology* is less well recovered (edge-set precision/recall ≈ 0.2):
the data would pick a somewhat different set of edges, but the directions of the
asserted ones are largely right. Four hand-written edges are **contradicted** by
the data (worth review): `school_enrollment→completion_rates`,
`completion_rates→youth_employment`, and the two near-zero
`digital_infrastructure→business_density` / `transportation_access→business_density`
(both source indicators are themselves disaggregated, so carry little real
provincial signal).

### 3. The random split is badly optimistic — honest validation is much weaker

Same models, two evaluation protocols:

| Protocol | R² (per-indicator dynamics) |
|---|---|
| **Random split** (what the trainer reports) | **+0.79** |
| **Temporal holdout** (train ≤2018, test ≥2019) | **strongly negative** (worse than predicting the mean) |
| Direction accuracy, out-of-time | **~60%** (barely above chance) |

The flattering +0.79 is a leakage artifact of shuffling adjacent province-years;
out-of-time the level models do not generalize, and they get the sign of next
year's change right only ~60% of the time. This is the number the model card's
"no forecast validation" caveat should cite explicitly.

## Implications

- Keep the hand-written **indicator→indicator** graph — the data endorses its
  directions (76%) — but consider revisiting the four contradicted edges, and
  add entity/year fixed effects if these couplings are ever estimated for real.
- Stop expecting the **lever→indicator** ML to carry signal: it cannot, by data
  availability. The honest framing is "persistence + documented elasticities,"
  with the rules owning lever response explicitly.
- Report the **temporal** R² and direction accuracy, not the random-split R², as
  the twin's validation status.
- A **region-level** reformulation is where any data-driven lever response could
  eventually live (spending is administered regionally), but is out of scope of
  the current province-only lever data.
