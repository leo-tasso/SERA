"""Post-hoc explanation tools for black-box policy models.

The neural policy performs well but cannot be read; these tools produce an
honest audit trail for it (or any per-province feature policy with the same
interface — ``feature_keys``, ``_feature_matrix``, ``decide``):

- :func:`permutation_importance` — shuffle one indicator across provinces and
  measure how much the policy's lever choices move. Answers "which indicators
  is this policy actually reacting to?".
- :func:`distill_to_tree` — fit a shallow decision tree that imitates the
  policy's decisions and report its **fidelity** (R² on held-out decisions).
  The tree's leaves read as province segments with lever packages; the
  fidelity score says how much of the black box the segments actually capture.

Both operate on the policy's *decisions* only — no extra twin rollouts — so
they are cheap enough to run after every optimisation.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def _decision_matrix(policy, state: pd.DataFrame, env) -> np.ndarray:
    """Policy decisions as a ``(n_provinces, n_levers)`` matrix in [0, 1]."""
    allocations = policy.decide(state, 0, env)
    matrix = np.zeros((len(env.provinces), len(policy.param_specs)), dtype=float)
    for row, code in enumerate(env.provinces):
        levers = allocations.get(code, {})
        for col, spec in enumerate(policy.param_specs):
            span = max(spec.max - spec.min, 1e-9)
            try:
                value = float(levers.get(spec.key, spec.baseline))
            except (TypeError, ValueError):
                value = float(spec.baseline)
            matrix[row, col] = (value - spec.min) / span
    return matrix


def permutation_importance(policy, env, *, n_repeats: int = 3, seed: int = 0) -> List[dict]:
    """Mean absolute lever shift (in lever-range units) per shuffled indicator."""
    state = env.initial_state
    base = _decision_matrix(policy, state, env)
    rng = np.random.default_rng(seed)
    importances = []
    for key in list(getattr(policy, "feature_keys", [])):
        if key not in state.columns:
            continue
        shifts = []
        for _ in range(n_repeats):
            shuffled = state.copy()
            shuffled[key] = rng.permutation(shuffled[key].to_numpy())
            shifts.append(float(np.abs(_decision_matrix(policy, shuffled, env) - base).mean()))
        importances.append({"feature": key, "importance": float(np.mean(shifts))})
    importances.sort(key=lambda item: -item["importance"])
    return importances


def _segments_from_tree(tree, feature_names: List[str], param_specs, top_levers: int) -> List[dict]:
    """Flatten a fitted multi-output tree into readable province segments."""
    inner = tree.tree_
    segments: List[dict] = []

    def walk(node: int, conditions: List[str]) -> None:
        if inner.children_left[node] == -1:  # leaf
            values = inner.value[node].reshape(-1)
            levers = []
            for col, spec in enumerate(param_specs):
                span = max(spec.max - spec.min, 1e-9)
                decoded = spec.min + float(np.clip(values[col], 0.0, 1.0)) * span
                levers.append(
                    {
                        "lever": spec.key,
                        "value": float(decoded),
                        "baseline": float(spec.baseline),
                        "shift": abs(decoded - spec.baseline) / span,
                    }
                )
            levers.sort(key=lambda item: -item["shift"])
            segments.append(
                {
                    "conditions": list(conditions) if conditions else ["all provinces"],
                    "n_samples": int(inner.n_node_samples[node]),
                    "levers": [
                        {key: value for key, value in lever.items() if key != "shift"}
                        for lever in levers[:top_levers]
                    ],
                }
            )
            return
        feature = feature_names[inner.feature[node]]
        # Features are relative deviations from the national reference mean.
        threshold_pct = float(inner.threshold[node]) * 100.0
        walk(
            inner.children_left[node],
            conditions + [f"{feature} ≤ national {threshold_pct:+.1f}%"],
        )
        walk(
            inner.children_right[node],
            conditions + [f"{feature} > national {threshold_pct:+.1f}%"],
        )

    walk(0, [])
    return segments


def distill_to_tree(
    policy,
    env,
    *,
    max_depth: int = 3,
    n_jitter: int = 8,
    noise: float = 0.15,
    seed: int = 0,
    top_levers: int = 5,
) -> Optional[dict]:
    """Distil the policy into a shallow surrogate tree with a fidelity score.

    The training set is the policy's own decisions on the initial state plus
    ``n_jitter`` perturbed copies of it (each indicator multiplied by
    ``1 + N(0, noise)``), so the tree sees how the policy reacts to nearby
    provincial states, not just the 110 observed ones. Fidelity is the R² of
    the tree's imitation on held-out rows — report it next to the segments.
    """
    from sklearn.metrics import r2_score
    from sklearn.tree import DecisionTreeRegressor

    feature_keys = list(getattr(policy, "feature_keys", []))
    if not feature_keys:
        return None

    rng = np.random.default_rng(seed)
    states = [env.initial_state]
    for _ in range(n_jitter):
        jittered = env.initial_state.copy()
        for key in feature_keys:
            if key not in jittered.columns:
                continue
            column = pd.to_numeric(jittered[key], errors="coerce").to_numpy(dtype=float)
            jittered[key] = column * (1.0 + noise * rng.standard_normal(len(column)))
        states.append(jittered)

    feature_rows = []
    decision_rows = []
    for state in states:
        feature_rows.append(policy._feature_matrix(state, 0, env)[:, 2:])
        decision_rows.append(_decision_matrix(policy, state, env))
    X = np.vstack(feature_rows)
    y = np.vstack(decision_rows)

    n_rows = X.shape[0]
    order = rng.permutation(n_rows)
    n_test = max(1, n_rows // 4)
    test_idx, train_idx = order[:n_test], order[n_test:]

    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=seed)
    tree.fit(X[train_idx], y[train_idx])
    try:
        fidelity = float(r2_score(y[test_idx], tree.predict(X[test_idx])))
    except ValueError:
        fidelity = 0.0
    if not np.isfinite(fidelity):
        fidelity = 0.0

    return {
        "fidelity_r2": fidelity,
        "max_depth": int(max_depth),
        "n_samples": int(n_rows),
        "segments": _segments_from_tree(tree, feature_keys, policy.param_specs, top_levers),
    }


def explain_policy_posthoc(policy, env, *, seed: int = 0) -> dict:
    """The full post-hoc audit bundle the UI renders for black-box policies."""
    return {
        "type": "neural_posthoc",
        "importances": permutation_importance(policy, env, seed=seed),
        "surrogate": distill_to_tree(policy, env, seed=seed),
        "note": (
            "The network itself is not interpretable. The importances show "
            "which indicators its decisions actually depend on; the segments "
            "are a shallow decision tree imitating it, honest only up to the "
            "stated fidelity."
        ),
    }
