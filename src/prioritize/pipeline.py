"""End-to-end /prioritize pipeline.

Orchestrates: intent parsing → filter → short-circuit check → optimizer →
pros/cons + counterfactual + LLM prose → structured response dict.

The response dict matches the schema in the implementation plan §10.
"""

from __future__ import annotations

import numpy as np

from src.prioritize.features import (
    FeatureMatrix,
    FEATURE_LABELS,
    filter_rows,
    get_feature_matrix,
    get_target_row_index,
)
from src.prioritize.intent import Intent, parse_intent
from src.prioritize.explain import (
    build_ranking_slice,
    compute_counterfactual,
    compute_pros_cons,
    llm_prose,
)
from src.prioritize.weights import (
    DEFAULT_WEIGHTS,
    InfeasibleError,
    blocking_rows as compute_blocking_rows,
    default_weight_vector,
    optimize_mip,
    target_rank,
)

K_DEFAULT = 3
TOP_N_RANKING = 10


REGION_TO_ISO3 = {
    "SSA":  ["SOM", "SSD", "COD", "CAF", "NER", "MLI", "BFA", "TCD", "ETH",
             "MOZ", "ZWE", "NGA", "CMR", "KEN", "UGA", "BDI", "RWA", "MDG",
             "MWI", "ZMB", "LBR", "SLE", "GIN", "SEN"],
    "MENA": ["YEM", "SYR", "IRQ", "LBY", "PSE", "LBN", "SDN", "EGY", "JOR",
             "TUN", "OPT"],
    "APAC": ["AFG", "MMR", "BGD", "PRK", "PAK", "PHL", "IDN", "NPL", "LAO",
             "KHM", "LKA", "TLS"],
    "LAC":  ["HTI", "VEN", "COL", "GTM", "HND", "SLV", "ECU", "PER"],
    "EECA": ["UKR", "TJK", "TKM", "TUR"],
}


def _weights_to_dict(w: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    return {name: float(w[i]) for i, name in enumerate(feature_names)}


def _target_meta(rows, target_idx: int) -> dict:
    row = rows.iloc[int(target_idx)]
    out = {}
    for col in ("country_iso3", "country_name", "sector", "sector_name", "year"):
        if col in rows.columns:
            val = row[col]
            if hasattr(val, "item"):
                try:
                    val = val.item()
                except Exception:
                    pass
            out[col] = val
    return out


def _score_of(X: np.ndarray, idx: int, w: np.ndarray) -> float:
    return float(X[idx] @ w)


def _default_ranking_indices(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    w_def = default_weight_vector(feature_names)
    scores = X @ w_def
    return np.argsort(-scores)


def _short_circuit_response(
    X: np.ndarray,
    rows,
    target_idx: int,
    intent: Intent,
    feature_names: list[str],
    k: int,
    use_llm_prose: bool,
) -> dict:
    w_def = default_weight_vector(feature_names)
    x_t = X[target_idx]
    pros, cons = compute_pros_cons(w_def, x_t, feature_names)
    counterfactual = compute_counterfactual(X, rows, target_idx, w_def, feature_names, k=k)
    ranking = build_ranking_slice(X, rows, w_def, target_idx, top_n=TOP_N_RANKING)
    target_meta = _target_meta(rows, target_idx)
    weights_dict = _weights_to_dict(w_def, feature_names)

    explanation = (
        llm_prose(target_meta, weights_dict, pros, cons, counterfactual, short_circuited=True)
        if use_llm_prose else ""
    )

    return {
        "mode": intent.mode,
        "intent": intent.__dict__,
        "weights": weights_dict,
        "weight_deviation_from_default": 0.0,
        "short_circuited": True,
        "reason": "Target is already in the top-k under default scoring",
        "target": {
            **target_meta,
            "rank": target_rank(X, target_idx, w_def),
            "score": _score_of(X, target_idx, w_def),
        },
        "ranking": ranking,
        "pros": pros,
        "cons": cons,
        "counterfactual": counterfactual,
        "explanation": explanation,
    }


def _filter_only_response(
    fm: FeatureMatrix,
    intent: Intent,
    k_top: int = 10,
    use_llm_prose: bool = False,
) -> dict:
    X_f, rows_f = filter_rows(
        fm,
        sector=intent.sector,
        region=intent.region,
        emergency_group=intent.emergency_group,
        year=intent.year,
        region_to_iso3=REGION_TO_ISO3,
    )
    if rows_f.empty:
        return {
            "mode": intent.mode,
            "intent": intent.__dict__,
            "weights": {name: float(v) for name, v in DEFAULT_WEIGHTS.items()},
            "weight_deviation_from_default": 0.0,
            "short_circuited": False,
            "target": None,
            "ranking": [],
            "pros": [],
            "cons": [],
            "counterfactual": None,
            "explanation": "No rows matched the given filter.",
        }

    w_def = default_weight_vector(fm.feature_names)
    scores = X_f @ w_def
    order = np.argsort(-scores)
    ranking = []
    for rank_pos, idx in enumerate(order[:k_top], start=1):
        idx = int(idx)
        entry = {"rank": rank_pos, "score": float(scores[idx])}
        for col in ("country_iso3", "country_name", "sector", "sector_name", "year"):
            if col in rows_f.columns:
                val = rows_f.iloc[idx][col]
                if hasattr(val, "item"):
                    try:
                        val = val.item()
                    except Exception:
                        pass
                entry[col] = val
        ranking.append(entry)

    return {
        "mode": intent.mode,
        "intent": intent.__dict__,
        "weights": _weights_to_dict(w_def, fm.feature_names),
        "weight_deviation_from_default": 0.0,
        "short_circuited": False,
        "target": None,
        "ranking": ranking,
        "pros": [],
        "cons": [],
        "counterfactual": None,
        "explanation": "Ranked by default weights; no target was specified.",
    }


def prioritize(
    query: str,
    k: int = K_DEFAULT,
    fm: FeatureMatrix | None = None,
    intent: Intent | None = None,
    use_llm_prose: bool = True,
) -> dict:
    """Run the full /prioritize flow for a natural-language query.

    Args:
        query:          NL query text.
        k:              rank target (default 3).
        fm:             optional pre-built FeatureMatrix (tests inject synthetic data here).
        intent:         optional pre-parsed intent (tests skip the LLM this way).
        use_llm_prose:  call Claude for the explanation string; False for tests/offline.

    Returns:
        Response dict matching implementation plan §10.

    Raises:
        InfeasibleError: when the target cannot be placed in the top k.
    """
    if fm is None:
        fm = get_feature_matrix()
    if intent is None:
        intent = parse_intent(query)

    if intent.mode == "filter_only":
        return _filter_only_response(fm, intent, use_llm_prose=use_llm_prose)

    if intent.country_iso3 is None:
        raise InfeasibleError(f"Target mode requires country_iso3; got intent={intent}")

    # target_only mode implicitly compares country-ALL rows. Without this filter,
    # "prioritize Yemen" would rank Yemen-ALL against Sudan-FSC, Somalia-HEA, etc.
    # — a grain mix that isn't meaningful.
    effective_sector = intent.sector if intent.mode == "target_sector" else "ALL"

    X_f, rows_f = filter_rows(
        fm,
        sector=effective_sector,
        year=intent.year,
        region_to_iso3=REGION_TO_ISO3,
    )
    if rows_f.empty:
        raise InfeasibleError(
            f"No rows found for filter sector={effective_sector} year={intent.year}"
        )

    target_idx = get_target_row_index(
        rows_f,
        country_iso3=intent.country_iso3,
        sector=effective_sector,
        year=intent.year,
    )
    if target_idx is None:
        raise InfeasibleError(
            f"No row found for {intent.country_iso3}"
            + (f"×{effective_sector}" if effective_sector else "")
        )

    w_def = default_weight_vector(fm.feature_names)
    default_rank = target_rank(X_f, target_idx, w_def)
    if default_rank <= k:
        return _short_circuit_response(
            X_f, rows_f, target_idx, intent, fm.feature_names, k, use_llm_prose
        )

    try:
        result = optimize_mip(X_f, target_idx, fm.feature_names, k=k)
    except InfeasibleError as e:
        blockers = compute_blocking_rows(
            X_f, target_idx, w_def, fm.feature_names, rows_f, n=2
        )
        raise InfeasibleError(
            reason=f"Could not place target in top {k}: {e.reason}",
            blocking_rows=blockers,
        )

    final_rank = result.rank_achieved
    if final_rank > k:
        blockers = compute_blocking_rows(
            X_f, target_idx, result.w, fm.feature_names, rows_f, n=2
        )
        raise InfeasibleError(
            reason=(
                f"Optimizer reported {result.status} but target rank = {final_rank} > {k}"
            ),
            blocking_rows=blockers,
        )

    x_t = X_f[target_idx]
    pros, cons = compute_pros_cons(result.w, x_t, fm.feature_names)
    counterfactual = compute_counterfactual(
        X_f, rows_f, target_idx, result.w, fm.feature_names, k=k
    )
    ranking = build_ranking_slice(X_f, rows_f, result.w, target_idx, top_n=TOP_N_RANKING)
    target_meta = _target_meta(rows_f, target_idx)
    weights_dict = _weights_to_dict(result.w, fm.feature_names)
    l2_dev = float(np.linalg.norm(result.w - w_def))

    explanation = (
        llm_prose(target_meta, weights_dict, pros, cons, counterfactual, short_circuited=False)
        if use_llm_prose else ""
    )

    return {
        "mode": intent.mode,
        "intent": intent.__dict__,
        "weights": weights_dict,
        "weight_deviation_from_default": l2_dev,
        "short_circuited": False,
        "target": {
            **target_meta,
            "rank": final_rank,
            "score": _score_of(X_f, target_idx, result.w),
        },
        "ranking": ranking,
        "pros": pros,
        "cons": cons,
        "counterfactual": counterfactual,
        "explanation": explanation,
    }
