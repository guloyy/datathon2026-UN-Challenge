"""Default weights, MIP optimizer, and blocking-row diagnostics.

DEFAULT_WEIGHTS — 0.5 coverage_gap + 0.5 need_scale.
    Justification: equal emphasis on "underfunded-ness" and "scale of need",
    matching the existing gap_score formula reframed as a weighted sum.

optimize_mip() — find w* = argmin ||w - w_default||² s.t. rank(target) ≤ k
    under a big-M MIP with binary z_j indicators.

optimize_lp_fallback() — LP relaxation with threshold + post-verify.

blocking_rows() — given an attempted weight vector, return the n rows that
    most beat the target and their dominant contributing features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cvxpy as cp
except ImportError as e:  # pragma: no cover
    raise ImportError("cvxpy is required for the prioritize optimizer. pip install cvxpy") from e


DEFAULT_WEIGHTS: dict[str, float] = {
    "coverage_gap": 0.5,
    "need_scale":   0.5,
}


class InfeasibleError(Exception):
    """Raised when no weight vector can place the target in the top k."""

    def __init__(self, reason: str, blocking_rows: Optional[list[dict]] = None):
        super().__init__(reason)
        self.reason = reason
        self.blocking_rows = blocking_rows or []


def default_weight_vector(feature_names: list[str]) -> np.ndarray:
    """Project DEFAULT_WEIGHTS onto the feature-name ordering."""
    w = np.zeros(len(feature_names), dtype=float)
    for i, name in enumerate(feature_names):
        w[i] = DEFAULT_WEIGHTS.get(name, 0.0)
    total = w.sum()
    if total <= 0:
        raise ValueError("DEFAULT_WEIGHTS do not overlap with the feature list.")
    return w / total


def _score_ranking(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Return score array (n,) = X @ w."""
    return X @ w


RANK_TIE_EPS = 1e-9


def target_rank(X: np.ndarray, target_idx: int, w: np.ndarray, eps: float = RANK_TIE_EPS) -> int:
    """1-based rank of the target row under weights w.

    Ties within `eps` are not counted as beating — prevents floating-point
    noise in QP-binding constraints from inflating the apparent rank.
    """
    scores = _score_ranking(X, w)
    target_score = scores[target_idx]
    better = int(np.sum(scores > target_score + eps))
    return better + 1


@dataclass
class OptResult:
    w: np.ndarray
    status: str
    rank_achieved: int


QP_SOLVER_ORDER = ("CLARABEL", "SCS")


def _solve_constrained_qp(
    X_other: np.ndarray,
    x_t: np.ndarray,
    must_not_beat_idx: np.ndarray,
    w_default: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Solve min ||w − w_default||² s.t. X_must · w ≤ x_t · w, w ≥ 0, sum(w) = 1.

    Tries CLARABEL first, falls back to SCS. OSQP is not used — it frequently
    hits its iteration limit on this problem shape.
    """
    d = w_default.shape[0]
    installed = set(cp.installed_solvers())
    last_status = "unknown"
    last_err: Optional[Exception] = None

    for solver in QP_SOLVER_ORDER:
        if solver not in installed:
            continue
        w = cp.Variable(d, nonneg=True)
        constraints = [cp.sum(w) == 1]
        if must_not_beat_idx.size > 0:
            constraints.append(X_other[must_not_beat_idx] @ w <= x_t @ w)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(w - w_default)), constraints)
        try:
            problem.solve(solver=solver)
        except cp.error.SolverError as e:
            last_err = e
            last_status = f"{solver}_error"
            continue

        last_status = problem.status
        if problem.status in ("infeasible", "infeasible_inaccurate"):
            raise InfeasibleError(f"QP status: {problem.status}")
        if problem.status in ("optimal", "optimal_inaccurate"):
            return np.asarray(w.value, dtype=float).flatten(), problem.status

    if last_err is not None:
        raise InfeasibleError(f"QP solver error: {last_err}")
    raise InfeasibleError(f"QP did not converge: last status={last_status}")


def optimize_mip(
    X: np.ndarray,
    target_idx: int,
    feature_names: list[str],
    k: int = 3,
    big_m: float = 10.0,
    solver: Optional[str] = None,
    max_iterations: int = 20,
) -> OptResult:
    """Rank-constrained optimizer via fixed-point iteration on the "allowed blockers" set.

    Because HIGHS (cvxpy's default free MILP solver) does not handle quadratic
    objectives on mixed-integer problems, the formal MIQP:

        min ||w - w_default||²
        s.t. X[j]·w − x_t·w ≤ M · z_j        ∀j≠target
             sum(z_j) ≤ k−1
             z_j ∈ {0,1}, w ≥ 0, sum(w) = 1

    is solved by iterating:

        allowed ← top-(k-1) rows by score under current w
        w       ← argmin ||w - w_default||²
                  s.t. X[j] · w ≤ x_t · w   for all j ∉ allowed,
                       w ≥ 0, sum(w) = 1
        repeat until `allowed` stops changing.

    At convergence, the "allowed" set equals the top-(k-1) under w, which is
    exactly the rank-k constraint. The final w is the QP-optimal minimum-L2
    departure from DEFAULT_WEIGHTS for that blocker set.
    """
    n, d = X.shape
    if target_idx < 0 or target_idx >= n:
        raise ValueError(f"target_idx {target_idx} out of range [0, {n})")
    if n == 1:
        w_def = default_weight_vector(feature_names)
        return OptResult(w=w_def, status="optimal_trivial", rank_achieved=1)

    w_default = default_weight_vector(feature_names)
    x_t = X[target_idx]
    mask = np.ones(n, dtype=bool)
    mask[target_idx] = False
    X_other = X[mask]
    m_rows = X_other.shape[0]

    w_current = w_default.copy()
    prev_allowed: Optional[frozenset[int]] = None
    status = "optimal"

    for _ in range(max_iterations):
        scores_other = X_other @ w_current
        target_score = float(x_t @ w_current)
        # Rows that currently beat target under w_current:
        beating = np.where(scores_other > target_score)[0]
        # Allowed-to-beat set = top-(k-1) rows by current score.
        if m_rows <= k - 1:
            allowed_idx = np.arange(m_rows, dtype=int)
        else:
            order = np.argsort(-scores_other)
            allowed_idx = order[: k - 1]
        allowed = frozenset(int(i) for i in allowed_idx)

        must_not_beat_idx = np.array(
            [j for j in range(m_rows) if j not in allowed], dtype=int
        )

        w_new, status = _solve_constrained_qp(
            X_other, x_t, must_not_beat_idx, w_default
        )
        w_new = np.clip(w_new, 0.0, None)
        s = w_new.sum()
        if s > 0:
            w_new = w_new / s

        if prev_allowed is not None and allowed == prev_allowed and np.allclose(
            w_new, w_current, atol=1e-6
        ):
            w_current = w_new
            break
        prev_allowed = allowed
        w_current = w_new

    rank = target_rank(X, target_idx, w_current)
    if rank > k:
        raise InfeasibleError(
            f"Iterative optimizer did not place target in top {k} (achieved rank {rank})"
        )
    return OptResult(w=w_current, status=status, rank_achieved=rank)


def optimize_lp_fallback(
    X: np.ndarray,
    target_idx: int,
    feature_names: list[str],
    k: int = 3,
) -> OptResult:
    """LP relaxation fallback: require target score ≥ threshold τ (score of k-th under default).

    Post-verifies the actual rank — raises InfeasibleError if final rank > k.
    """
    n, d = X.shape
    w_default = default_weight_vector(feature_names)

    scores_default = X @ w_default
    sorted_desc = np.sort(scores_default)[::-1]
    tau = float(sorted_desc[min(k - 1, n - 1)])

    w = cp.Variable(d, nonneg=True)
    x_t = X[target_idx]
    constraints = [cp.sum(w) == 1, x_t @ w >= tau]
    objective = cp.Minimize(cp.sum_squares(w - w_default))
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise InfeasibleError(f"LP fallback status: {problem.status}")

    w_val = np.clip(np.asarray(w.value, dtype=float).flatten(), 0.0, None)
    s = w_val.sum()
    if s > 0:
        w_val = w_val / s

    rank = target_rank(X, target_idx, w_val)
    return OptResult(w=w_val, status=problem.status, rank_achieved=rank)


def blocking_rows(
    X: np.ndarray,
    target_idx: int,
    attempted_w: np.ndarray,
    feature_names: list[str],
    rows_meta,  # pd.DataFrame — kept untyped to avoid a hard pandas import at this layer
    n: int = 2,
    top_dominant_features: int = 2,
) -> list[dict]:
    """Return the n rows that most beat the target under attempted_w.

    For each blocker, list the features that contribute most to the score gap.
    """
    scores = X @ attempted_w
    target_score = scores[target_idx]
    gaps = scores - target_score
    gaps[target_idx] = -np.inf

    beating_idx = np.where(gaps > 0)[0]
    if beating_idx.size == 0:
        return []

    order = beating_idx[np.argsort(-gaps[beating_idx])]
    picked = order[:n]

    x_t = X[target_idx]
    out: list[dict] = []
    for idx in picked:
        per_feature_gap = (X[idx] - x_t) * attempted_w
        top_feat_idx = np.argsort(-per_feature_gap)[:top_dominant_features]
        dominant = []
        for fi in top_feat_idx:
            if per_feature_gap[fi] <= 0:
                continue
            dominant.append({
                "feature":      feature_names[fi],
                "row_value":    float(X[idx][fi]),
                "target_value": float(x_t[fi]),
                "contribution_gap": float(per_feature_gap[fi]),
            })

        meta = {}
        row = rows_meta.iloc[int(idx)]
        for col in ("country_iso3", "country_name", "sector", "year"):
            if col in rows_meta.columns:
                val = row[col]
                if hasattr(val, "item"):
                    try:
                        val = val.item()
                    except Exception:
                        pass
                meta[col] = val

        out.append({
            **meta,
            "score_gap": float(gaps[idx]),
            "dominant_features": dominant,
        })

    return out
