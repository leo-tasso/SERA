# SERA — Ethics Statement

SERA simulates Italian provincial socioeconomics and lets an AI optimizer propose policy.
That combination — an AI system advising on the allocation of public resources — is exactly
the kind of system that deserves explicit ethical scaffolding. This document states what
the system is for, what it must not be used for, which value judgments it embeds, and how
its design tries to keep the human, not the model, in charge.

Companion documents: [docs/MODEL_CARD.md](docs/MODEL_CARD.md) (the models),
[docs/DATASHEET.md](docs/DATASHEET.md) (the data).

## Intended use

SERA is an **educational and research prototype** for exploring how policy choices *might*
propagate through provincial indicators, and how different *ethical frameworks* change what
an optimizer recommends. It is decision **support**, never decision **making**:

- The optimizer returns **candidate policies with stated trade-offs** (efficiency,
  inequality, the worst-off province, unspent budget). A human must explicitly adopt one —
  including, as a first-class option, the baseline "do nothing".
- Nothing the optimizer produces is applied to the twin automatically.

## Out-of-scope / prohibited uses

- Real-world resource allocation or any administrative decision affecting actual people or
  provinces, without the validation, governance, and legal review such use would require.
- Presenting SERA projections as forecasts. They are model estimates conditioned on strong,
  explicitly hand-written assumptions (see "Honesty about uncertainty").
- Removing or hiding the candidate-selection step, the sensitivity band, or the baseline
  option to produce a single "the AI says do X" answer.

## Value judgments are explicit, not hidden

An optimizer always embodies an ethical framework; the only question is whether it is
visible. In SERA the framework is a user-facing choice (`sera.twin.objectives`):

| Objective | Ethical position | What it ignores |
| --- | --- | --- |
| Utilitarian (total GDP) | Sum of outcomes; a euro counts the same everywhere | Distribution entirely |
| Rawlsian (maximin) | Only the worst-off province counts | Growth everywhere else |
| Egalitarian (Sen welfare) | Growth discounted by inter-provincial Gini | Within-province inequality |
| Multi-objective wellbeing | GDP, life expectancy, unemployment, poverty | Everything not in the composite |

The **ethics equity dashboard** trains the same model under all four frameworks and shows
the diverging futures side by side — including per-province "who gains, who loses" maps —
so the distributional consequences of the choice are inspectable rather than implied.

Two further value judgments worth naming:

- **The unit of fairness is the province.** Within-province inequality (rich and poor
  households in Milan) is invisible to every objective.
- **GDP per capita summed across provinces** is the "national GDP" proxy. It is not
  population-weighted; small provinces weigh as much as large ones.

## Honesty about uncertainty

- The twin's lever→indicator response comes in part from **hand-written causal rules**
  (`CAUSAL_RULE_STRENGTH` in `sera.twin.simulator`) layered on top of statistically weak
  signals in the historical data. These are assumptions, not measurements.
- Every optimizer run therefore re-simulates the proposed policy with those rules at **half
  and 1.5× strength** and shows the spread as a shaded band. A wide band means the
  projection is mostly assumption.
- A ±6 %/year realism cap and a policy-signal cap bound the dynamics; they also mean the
  model structurally cannot predict shocks, crises, or regime changes.
- The UI labels simulated values as model estimates and marks where history ends and
  simulation begins. No prediction intervals in the statistical sense are provided —
  the band is a sensitivity analysis, not a confidence interval.

## Human oversight by construction

- Candidates, not commands: full intervention, moderate intervention (levers halfway back
  toward baseline), and baseline are always presented together with their trade-offs.
- Adoption is explicit, single-shot, and recorded in the UI; after adopting, fresh
  candidates must be generated from the new state.
- The objective dropdown forces the value judgment to be made *before* optimization, by a
  person.

## Who could be harmed

- **People invisible in the data.** Official statistics under-represent the informal
  economy, undocumented residents, and the homeless; an optimizer maximizing measured
  indicators optimizes for the measured population (see the datasheet).
- **Residents of "unprofitable" provinces** under the utilitarian objective, which will
  happily concentrate gains where returns are highest. This is why the Rawlsian and
  egalitarian objectives and the equity dashboard exist.
- **Decision-makers themselves**, via automation bias: a polished dashboard invites more
  trust than a hand-tuned simulation deserves. The uncertainty banner, sensitivity band,
  and mandatory candidate choice are mitigations, not solutions.

## EU AI Act positioning

As distributed, SERA is a research/educational prototype, not a deployed AI system.
However, if a system like SERA were used by a public authority to inform the allocation of
public funds or essential services, it would plausibly fall in the **high-risk** category
(Regulation (EU) 2024/1689, Annex III), triggering obligations that this project's design
anticipates deliberately:

| AI Act obligation (Art. 9–15) | Where SERA addresses it |
| --- | --- |
| Risk management & known limitations | This document; model card "Limitations" |
| Data governance | Datasheet (provenance, gaps, preprocessing) |
| Technical documentation | README, model card, datasheet |
| Transparency to users | In-UI uncertainty banner; objective descriptions |
| Human oversight (Art. 14) | Candidate/adopt flow; baseline as first-class choice |
| Accuracy & robustness | Sensitivity band; realism caps; documented validation gaps |

Anyone moving SERA toward real use must treat that table as a starting checklist, not as
compliance.

## Contact

Questions or concerns about the ethics of this project: open an issue in the repository.
