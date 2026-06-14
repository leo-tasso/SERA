"""Estimate twin structure from the historical panel instead of asserting it.

The production twin (:mod:`sera.twin.model_trainer`) fits one pooled ridge per
indicator on *all* national levers plus a lag, with no entity/time effects and a
random train/test split. Two consequences, established by analysis:

* the **lever -> indicator** response is unidentifiable, because the policy
  levers are national time series with no cross-province variation; and
* the **indicator -> indicator** couplings, which the data *can* support, are
  nonetheless hand-written in :data:`sera.twin.causal_graph.INDICATOR_TO_INDICATORS`.

This module supplies the missing, defensible estimation so the gap can be
quantified rather than assumed:

* :func:`load_panel` builds a wide ``(entity, year)`` panel at province *or*
  region granularity (regions are where spending policy actually varies);
* :func:`two_way_within` applies entity + year fixed effects (the contrast the
  pooled model omits);
* :func:`learn_couplings` estimates the indicator -> indicator graph from the
  panel under fixed effects, and :func:`coupling_agreement` scores the learned
  signs against the hand-written graph;
* :func:`temporal_backtest` reports honest, time-ordered generalization and the
  optimism of the random split the production trainer uses;
* :func:`lever_cross_section_variation` quantifies, directly from the data, that
  the levers do not vary across entities --- the reason their response cannot be
  learned at any granularity.

Nothing here touches the production twin or the report; it is an analysis layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sera.twin.causal_graph import (
    ANNUAL_PARAMETERS,
    INDICATOR_EFFECT_DIRECTION,
    INDICATOR_TO_INDICATORS,
)
from sera.twin.data_loader import DataLoader
from sera.twin.province_mapping import PROVINCE_TO_REGION

ENTITY = "entity"
YEAR = "year"


# --------------------------------------------------------------------------- #
# Panel construction
# --------------------------------------------------------------------------- #
def load_panel(
    data_dir: Path,
    indicators: Dict[str, object],
    *,
    level: str = "province",
    min_year: int = 2001,
    max_year: int = 2025,
) -> pd.DataFrame:
    """Load a wide ``(entity, year, <indicators>)`` panel.

    ``indicators`` is the ``{name: (category, direction)}`` map the UI uses.
    ``level`` is ``"province"`` (110 entities) or ``"region"`` (20 entities,
    provinces aggregated by unweighted mean --- a documented simplification, fine
    for estimating relationship *structure*).
    """
    loader = DataLoader(Path(data_dir))
    indicator_cats = {
        name: meta[0] if isinstance(meta, (tuple, list)) else meta
        for name, meta in indicators.items()
    }
    combined_ind, _params = loader.prepare_training_data(
        {name: (cat, 1) for name, cat in indicator_cats.items()},
        {},  # parameters handled separately; not needed for indicator structure
        min_year=min_year,
        max_year=max_year,
    )
    if combined_ind is None or combined_ind.empty:
        return pd.DataFrame(columns=[ENTITY, YEAR])

    df = combined_ind.rename(columns={"area_code": ENTITY})
    df[YEAR] = pd.to_numeric(df[YEAR], errors="coerce")
    df = df.dropna(subset=[YEAR])
    df[YEAR] = df[YEAR].astype(int)

    if level == "region":
        df[ENTITY] = df[ENTITY].map(lambda code: PROVINCE_TO_REGION.get(str(code).upper()))
        df = df.dropna(subset=[ENTITY])
        value_cols = [c for c in df.columns if c not in (ENTITY, YEAR)]
        df = df.groupby([ENTITY, YEAR], as_index=False)[value_cols].mean()
    elif level != "province":
        raise ValueError(f"level must be 'province' or 'region', got {level!r}")

    return df.sort_values([ENTITY, YEAR]).reset_index(drop=True)


def add_lags(panel: pd.DataFrame, columns: Sequence[str], lag: int = 1) -> pd.DataFrame:
    """Append ``<col>_lag{lag}`` columns, shifted within each entity by year."""
    out = panel.sort_values([ENTITY, YEAR]).copy()
    for col in columns:
        out[f"{col}_lag{lag}"] = out.groupby(ENTITY)[col].shift(lag)
    return out


# --------------------------------------------------------------------------- #
# Fixed effects (the contrast the pooled model omits)
# --------------------------------------------------------------------------- #
def two_way_within(
    frame: pd.DataFrame, value_cols: Sequence[str], *, entity_means=True, year_means=True
) -> pd.DataFrame:
    """Two-way within transform: subtract entity and year means, add grand mean.

    This removes additive entity fixed effects (a rich province is always rich)
    and common-year shocks, leaving the within-entity, within-year variation that
    actually identifies a relationship --- exactly what a pooled regression on raw
    levels confounds.
    """
    out = frame.copy()
    for col in value_cols:
        series = out[col].astype(float)
        grand = series.mean()
        adjusted = series.copy()
        if entity_means:
            adjusted = adjusted - out.groupby(ENTITY)[col].transform("mean") + grand
        if year_means:
            adjusted = adjusted - out.groupby(YEAR)[col].transform("mean") + grand
        out[col] = adjusted
    return out


def _design(
    lagged: pd.DataFrame, target: str, source_lag_cols: Sequence[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Rows where the target exists, with missing lagged predictors mean-imputed.

    Mirrors the production trainer's imputation (mean-fill) so the panel's
    sparsity does not collapse the sample, while keeping the target authentic.
    """
    cols = [ENTITY, YEAR, target] + list(source_lag_cols)
    data = lagged[cols].copy()
    data = data[data[target].notna()]
    usable = []
    for col in source_lag_cols:
        if data[col].notna().any():
            mean = data[col].mean()
            data[col] = data[col].fillna(mean if pd.notna(mean) else 0.0)
            usable.append(col)
    data = data.dropna(subset=[target] + usable)
    return data, usable


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge on standardized X (returns coefficients on standardized X)."""
    Xc = X - X.mean(axis=0)
    Xs = Xc / (X.std(axis=0) + 1e-9)
    yc = y - y.mean()
    d = Xs.shape[1]
    beta = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(d), Xs.T @ yc)
    return beta


def expected_sign(source: str, target: str) -> int:
    """Documented sign of a source->target indicator edge.

    Mirrors the heuristic the simulator's rule layer uses: the product of the two
    indicators' "higher is better" directions, so two same-polarity indicators
    move together (+) and opposite-polarity ones move against each other (-).
    """
    return int(
        np.sign(
            INDICATOR_EFFECT_DIRECTION.get(source, 0) * INDICATOR_EFFECT_DIRECTION.get(target, 0)
        )
    )


# --------------------------------------------------------------------------- #
# Learning the indicator -> indicator graph
# --------------------------------------------------------------------------- #
def learn_couplings(
    panel: pd.DataFrame,
    *,
    fe: bool = True,
    alpha: float = 1.0,
    lag: int = 1,
    min_samples: int = 30,
) -> Dict[str, Dict[str, float]]:
    """Estimate, per target indicator, its dependence on the lagged others.

    For each target ``k`` we regress ``k_t`` on the lag-1 values of every other
    indicator (optionally after the two-way within transform), and return the
    standardized ridge coefficient of each candidate source. The sign and
    magnitude of those coefficients are the data's vote on the edges that
    :data:`INDICATOR_TO_INDICATORS` asserts by hand.
    """
    indicators = [c for c in panel.columns if c not in (ENTITY, YEAR)]
    lagged = add_lags(panel, indicators, lag=lag)
    learned: Dict[str, Dict[str, float]] = {}

    for target in indicators:
        sources = [s for s in indicators if s != target]
        lag_cols = [f"{s}_lag{lag}" for s in sources]
        data, usable = _design(lagged, target, lag_cols)
        if len(data) < min_samples or not usable:
            continue
        if fe:
            data = two_way_within(data, [target] + usable)
        X = data[usable].to_numpy(float)
        y = data[target].to_numpy(float)
        keep = X.std(axis=0) > 1e-9  # drop constants (e.g. after demeaning)
        if not keep.any():
            continue
        beta = _ridge_fit(X[:, keep], y, alpha)
        coefs: Dict[str, float] = {}
        bi = 0
        for col, k in zip(usable, keep):
            src = col[: -len(f"_lag{lag}")]
            coefs[src] = float(beta[bi]) if k else 0.0
            bi += int(bool(k))
        learned[target] = coefs
    return learned


def coupling_agreement(
    learned: Dict[str, Dict[str, float]],
    graph: Optional[Dict[str, List[str]]] = None,
    *,
    top_k_relative: float = 0.0,
) -> dict:
    """Score the learned couplings against the hand-written graph.

    Two questions:

    * **Sign agreement** on the asserted edges: for every hand-written edge
      ``source -> target`` present in the panel, does the learned coefficient's
      sign match the documented :func:`expected_sign`?
    * **Edge recovery**: treating each target's strongest learned sources as the
      data-driven predecessors, how much do they overlap the hand-written ones
      (precision/recall)?
    """
    graph = graph if graph is not None else INDICATOR_TO_INDICATORS
    agree = disagree = missing = 0
    per_edge = []
    for source, targets in graph.items():
        for target in targets:
            coef = learned.get(target, {}).get(source)
            if coef is None:
                missing += 1
                continue
            want = expected_sign(source, target)
            got = int(np.sign(coef))
            if want == 0 or got == 0:
                missing += 1
                continue
            if got == want:
                agree += 1
            else:
                disagree += 1
            per_edge.append((source, target, want, coef))

    # Edge recovery: data-driven top sources per target vs hand-written.
    tp = fp = fn = 0
    for target, coefs in learned.items():
        hand = {s for s, ts in graph.items() if target in ts}
        if not hand:
            continue
        ranked = sorted(coefs.items(), key=lambda kv: -abs(kv[1]))
        n = max(len(hand), int(round(top_k_relative * len(coefs)))) if top_k_relative else len(hand)
        data_driven = {s for s, _ in ranked[:n]}
        tp += len(hand & data_driven)
        fp += len(data_driven - hand)
        fn += len(hand - data_driven)

    scored = agree + disagree
    return {
        "edges_scored": scored,
        "edges_missing": missing,
        "sign_agree": agree,
        "sign_disagree": disagree,
        "sign_agreement": (agree / scored) if scored else float("nan"),
        "edge_precision": (tp / (tp + fp)) if (tp + fp) else float("nan"),
        "edge_recall": (tp / (tp + fn)) if (tp + fn) else float("nan"),
        "per_edge": per_edge,
    }


# --------------------------------------------------------------------------- #
# Honest temporal validation (#4)
# --------------------------------------------------------------------------- #
def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def temporal_backtest(
    panel: pd.DataFrame,
    *,
    fe: bool = True,
    alpha: float = 1.0,
    lag: int = 1,
    cutoff_frac: float = 0.75,
    min_samples: int = 40,
) -> dict:
    """Time-ordered backtest of the indicator dynamics, vs. a random split.

    For each target, train on the earliest ``cutoff_frac`` of years and test on
    the rest (entity-demeaned to honour fixed effects). Reports the mean held-out
    R^2 and the directional accuracy (does the model get the sign of the
    year-over-year change right?), alongside the R^2 of a *random* split of the
    same rows --- the optimistic number the production trainer reports.
    """
    indicators = [c for c in panel.columns if c not in (ENTITY, YEAR)]
    lagged = add_lags(panel, indicators, lag=lag)
    years = np.sort(panel[YEAR].unique())
    if len(years) < 4:
        return {"targets": 0}
    cutoff = years[int(len(years) * cutoff_frac)]

    temporal_r2, random_r2, dir_acc = [], [], []
    rng = np.random.default_rng(0)
    for target in indicators:
        source_cols = [f"{s}_lag{lag}" for s in indicators if s != target]
        data, sources = _design(lagged, target, source_cols)
        if len(data) < min_samples or not sources:
            continue
        if fe:  # one-way entity demeaning (year FE would not extrapolate to test years)
            data = two_way_within(data, [target] + sources, year_means=False)

        def split_fit_score(train_mask):
            tr, te = data[train_mask], data[~train_mask]
            if len(tr) < min_samples // 2 or len(te) < 5:
                return None
            Xtr, ytr = tr[sources].to_numpy(float), tr[target].to_numpy(float)
            Xte, yte = te[sources].to_numpy(float), te[target].to_numpy(float)
            keep = Xtr.std(axis=0) > 1e-9
            if not keep.any():
                return None
            mu, sd = Xtr[:, keep].mean(0), Xtr[:, keep].std(0) + 1e-9
            beta = _ridge_fit(Xtr[:, keep], ytr, alpha)
            pred = ((Xte[:, keep] - mu) / sd) @ beta + ytr.mean()
            return yte, pred

        temporal = split_fit_score((data[YEAR] < cutoff).to_numpy())
        if temporal is not None:
            yte, pred = temporal
            temporal_r2.append(_r2(yte, pred))
            # Directional accuracy vs the demeaned target's own mean as reference.
            dir_acc.append(float(np.mean(np.sign(pred - pred.mean()) == np.sign(yte - yte.mean()))))

        rand_mask = rng.random(len(data)) < cutoff_frac
        rnd = split_fit_score(rand_mask)
        if rnd is not None:
            random_r2.append(_r2(*rnd))

    return {
        "targets": len(temporal_r2),
        "cutoff_year": int(cutoff),
        "temporal_r2_mean": float(np.nanmean(temporal_r2)) if temporal_r2 else float("nan"),
        "temporal_r2_median": float(np.nanmedian(temporal_r2)) if temporal_r2 else float("nan"),
        "random_r2_mean": float(np.nanmean(random_r2)) if random_r2 else float("nan"),
        "direction_accuracy_mean": float(np.nanmean(dir_acc)) if dir_acc else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Why levers can't be learned (#2, stated as a measured fact)
# --------------------------------------------------------------------------- #
def reviewed_coupling_signs(
    data_dir: Path,
    indicators: Dict[str, object],
    *,
    level: str = "province",
    graph: Optional[Dict[str, List[str]]] = None,
    eps: float = 1e-3,
) -> dict:
    """Data-reviewed signs for the hand-written indicator->indicator edges.

    For every edge ``source -> target`` in the hand-written graph, returns a sign
    in ``{-1, 0, +1}`` to drive the simulator's propagation:

    * the learned fixed-effects coupling's sign where the edge is estimable and
      non-negligible (``|coef| > eps``);
    * ``0`` (drop the edge) where it is estimable but the data shows no link;
    * the documented :func:`expected_sign` where the edge cannot be tested
      (an indicator outside the panel).

    This replaces the simulator's previous *sign-less* propagation (which pushed
    every target in the same direction as its source, mis-signing every
    opposite-polarity edge) with directions the panel endorses.
    """
    graph = graph if graph is not None else INDICATOR_TO_INDICATORS
    panel = load_panel(data_dir, indicators, level=level)
    learned = learn_couplings(panel, fe=True) if not panel.empty else {}

    signs: Dict[str, Dict[str, int]] = {}
    flipped: List[str] = []
    dropped: List[str] = []
    for source, targets in graph.items():
        for target in targets:
            coef = learned.get(target, {}).get(source)
            documented = expected_sign(source, target)
            if coef is None:
                sign = documented  # not estimable: trust domain knowledge
            elif abs(coef) <= eps:
                sign = 0  # estimable but no signal: drop
                dropped.append(f"{source}->{target}")
            else:
                sign = int(np.sign(coef))
                if documented != 0 and sign != documented:
                    flipped.append(f"{source}->{target}")
            signs.setdefault(source, {})[target] = sign
    return {"signs": signs, "flipped": flipped, "dropped": dropped, "level": level}


def save_coupling_signs(
    data_dir: Path, indicators: Dict[str, object], out_path: Path, *, level: str = "province"
) -> dict:
    """Compute :func:`reviewed_coupling_signs` and persist to JSON for the simulator."""
    import json

    reviewed = reviewed_coupling_signs(data_dir, indicators, level=level)
    Path(out_path).write_text(json.dumps(reviewed, indent=2))
    return reviewed


def lever_data_granularity(data_dir: Path) -> dict:
    """Measure the native granularity of the policy-lever source data.

    The cleanest statement of why lever response cannot be learned at any entity
    granularity: the raw lever files have no sub-national observations to begin
    with. This classifies each of the 20 levers' provenance
    (:meth:`DataLoader.classify_provenance`) and counts how many are published
    only at the national level --- i.e. carry zero genuine cross-entity
    variation, so the per-province values the trainer sees are pure
    population-share artifacts of disaggregation, mutually collinear and
    uninformative about policy.
    """
    loader = DataLoader(Path(data_dir))
    provenance = {p: loader.classify_provenance(p, "annual_parameters") for p in ANNUAL_PARAMETERS}
    national = sum(
        1
        for v in provenance.values()
        if v in (DataLoader.PROVENANCE_DISAGG_NATIONAL, DataLoader.PROVENANCE_DISAGG_REGIONAL)
    )
    measured = sum(1 for v in provenance.values() if v == DataLoader.PROVENANCE_MEASURED)
    return {
        "levers": len(provenance),
        "national_or_regional_only": national,
        "province_measured": measured,
        "provenance": provenance,
    }
