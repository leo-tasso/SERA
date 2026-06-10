# Model Card — SERA Digital Twin & Policy Models

Following the structure of Mitchell et al., *Model Cards for Model Reporting* (2019).
Companion documents: [../ETHICS.md](../ETHICS.md), [DATASHEET.md](DATASHEET.md).

## Model details

SERA contains three model layers:

1. **Indicator models** (`sera.twin.model_trainer`) — one scikit-learn regressor per
   socioeconomic indicator (ridge regression by default), predicting next-year provincial
   values from lagged indicators plus the year's policy levers. Persisted to
   `twin_models.joblib`.
2. **Simulator** (`sera.twin.simulator`) — wraps the indicator models with hand-written
   causal rules (`CAUSAL_RULE_STRENGTH`), inter-indicator propagation, indicator bounds,
   and a ±6 %/year realism cap. Predictions are anchored to the previous year's value;
   only the *policy signal* (prediction under chosen levers minus prediction under
   baseline levers, capped at `POLICY_SIGNAL_CAP`) is applied. An honest finding: the
   trained models' relative response saturates that cap for almost any lever deviation,
   so the cap value *is* the effective size of the ML layer's contribution, and the
   differentiated lever→indicator structure comes mostly from the hand-written rules.
3. **Policy models** (`sera.twin.policy`) — gradient-free optimizers that drive the twin:
   - `neural`: a small pure-NumPy MLP applied per province, trained with mirrored-sampling
     Evolution Strategies;
   - `uniform_cem`: one shared national lever vector, found with the Cross-Entropy Method;
   - `baseline`: historical levers (no optimization);
   - `BlendedPolicy`: any policy's levers scaled toward baseline (used for the "moderate
     intervention" candidate).

   All trainable policies maximize a pluggable **ethical objective**
   (`sera.twin.objectives`): utilitarian total GDP, Rawlsian maximin, egalitarian Sen
   welfare, or a multi-indicator wellbeing composite.

- **Developers:** SERA project (educational/research prototype).
- **Model date:** 2026.
- **License/availability:** trained weights are reproducible from the repository and the
  public ISTAT data.

## Intended use

- Exploring how policy levers *might* propagate through provincial indicators.
- Comparing the recommendations of different ethical frameworks on the same model.
- Teaching: a worked example of value-laden objective functions, sensitivity analysis,
  and human-oversight UX in an AI decision-support system.

**Out of scope:** real policy decisions, forecasting, ranking provinces for funding, or
any administrative use affecting actual people. See ETHICS.md for the full statement.

## Training data

ISTAT (Italian National Institute of Statistics) SDMX API: ~88 indicators, 110 provinces,
2001–2025 (the UI uses a 24-indicator subset). National- and regional-level series are
disaggregated to provinces and gaps interpolated — see the datasheet for methods and
caveats.

## Evaluation and validation

Honest status: **there is no held-out forecast validation.** Specifically:

- The indicator models are fit on the full historical window; reported quality is
  in-sample. The simulator design (anchoring + capped policy signal) deliberately uses
  the models only for *relative* policy response, because their absolute calibration is
  not trusted.
- The causal-rule layer is hand-written domain knowledge, not estimated from data. Every
  optimizer run re-simulates the chosen policy with these rules at 0.5× and 1.5× strength
  and reports the spread (the UI's sensitivity band).
- Policy training quality is reported per run (start vs. best objective score).
- Unit tests cover mechanics (bounds, budget constraint, objective math, determinism),
  not predictive accuracy.

Any claim of real-world accuracy is therefore unsupported; treat outputs as internally
consistent scenarios, not predictions.

## Limitations

- **Hand-tuned dynamics dominate aggressive scenarios.** Lever effects flow through
  `CAUSAL_RULE_STRENGTH`, `POLICY_SIGNAL_CAP`, a propagation factor, and growth caps —
  all chosen constants, calibrated so that trade-offs between policies exist at all
  (with looser caps, every intervention saturates the growth limit and all
  growth-favoring objectives collapse onto the same optimum).
- **The budget is stylized.** The national pool equals a fixed share of GDP scaled by
  the average tax-lever level; spending costs are average lever ratios. Real fiscal
  multipliers, debt, and EU rules are absent.
- **No shocks or regime changes.** The ±6 %/year cap structurally excludes crises,
  pandemics, or booms.
- **Province-level only.** Within-province distribution is invisible; "national GDP" is
  the unweighted sum of provincial GDP per capita.
- **No demographic feedback.** Population, migration, and ageing do not respond to policy.
- **Static world.** No EU/global economy, no interactions with national fiscal policy
  beyond the budget constraint.
- **Optimizers exploit the simulator, not reality.** A policy that scores well has found
  a good point in the *model*; transferring that claim to Italy is a category error.

## Ethical considerations

The objective function is a user-facing ethical choice; the UI presents candidates rather
than answers and shows equity metrics (Gini, worst-off province) alongside GDP. See
[ETHICS.md](../ETHICS.md) for value judgments, affected groups, and EU AI Act positioning.
