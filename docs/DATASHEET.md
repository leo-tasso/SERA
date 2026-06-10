# Datasheet — SERA Provincial Indicator Dataset

Following the structure of Gebru et al., *Datasheets for Datasets* (2021).
Companion documents: [../ETHICS.md](../ETHICS.md), [MODEL_CARD.md](MODEL_CARD.md).

## Motivation

Assembled to train and drive the SERA digital twin: a multi-decade, multi-domain panel of
socioeconomic indicators for the 110 Italian provinces, suitable for simulating policy
scenarios. Created for education and research, not for administrative use.

## Composition

- **Unit of observation:** province-year (110 provinces by 2-letter sigla, 2001–2025).
- **Coverage:** ~88 indicators across economic, labor, education, healthcare, social
  wellbeing, environment, energy, innovation/infrastructure, transportation, and public
  finance categories (`src/sera/config.py` holds the indicator→category map).
- **Format:** one CSV per indicator under `data/<category>/<indicator>/`, with a
  `.mapping.json` sidecar documenting the source dataflow and column mapping.
- The dataset contains **aggregate statistics only** — no personal data. People are
  present only as counted populations.

## Collection process

- **Source:** ISTAT (Italian National Institute of Statistics) SDMX API, fetched by
  `sera.downloader` / `sera.istat_client`.
- **Rate limiting:** ISTAT allows 5 requests/minute per IP; the client enforces a
  25-second minimum delay and caches responses in `data/cache/`.
- **Known source quirk:** ISTAT's `endPeriod` returns year+1; the client compensates.
- Each `.mapping.json` records exactly which ISTAT dataflow and dimensions produced each
  CSV, so every series is traceable to its official source.

## Preprocessing

Performed at load time by `sera.twin.data_loader` (raw CSVs are kept as downloaded):

- **Disaggregation:** indicators published only at national or regional level are spread
  down to provinces (population/share-based). These provincial values are *estimates*,
  not measurements — provincial variation for such indicators is partly an artifact.
- **Interpolation:** missing province-years are interpolated when standardizing to the
  110-province panel.
- **Standardization:** province codes normalized to the 110 two-letter sigle
  (`sera.twin.province_mapping`), including post-reform province changes.

Consumers cannot currently distinguish measured from disaggregated/interpolated values in
the loaded panel; treat fine-grained provincial differences with suspicion for any
indicator published above province level.

## Who is missing

Official statistics measure the officially visible. Under- or un-represented:

- the informal/shadow economy (significant and regionally uneven in Italy);
- undocumented residents and recent migrants not in registries;
- homeless and institutionalized populations;
- within-province heterogeneity (every value is a provincial aggregate).

Any optimization on this data optimizes for the *measured* population. This is a primary
ethical caveat of the whole project (see ETHICS.md).

## Uses

- Used by: twin training (`sera.twin.cli`), the simulator's initial state, and the UI's
  historical trend charts.
- **Unsuitable for:** individual-level inference (ecological fallacy), official
  statistics reporting (use ISTAT directly), or precise inter-provincial comparisons of
  disaggregated indicators.

## Distribution and maintenance

- Raw ISTAT data is **not redistributed** with the repository; each user fetches it from
  the ISTAT API under ISTAT's terms (ISTAT data is generally CC BY 4.0 — verify per
  dataflow).
- The dataset is a snapshot ending at 2025; ISTAT revises history, so re-downloads may
  not reproduce it exactly (the `data/cache/` layer preserves a given snapshot).
- Maintained on a best-effort basis as part of the SERA project.
