"""Pros / cons extraction + deterministic counterfactuals + LLM prose.

No LLM is used for counterfactual *selection*. The LLM is only used to turn
the structured output into a natural-language explanation string.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from src.prioritize.features import FEATURE_LABELS
from src.prioritize.weights import default_weight_vector, target_rank

PROS_CONTRIBUTION_THRESHOLD = 0.05   # feature contributes >5% of target score
CONS_WEIGHT_THRESHOLD = 0.02         # weight is essentially zero
CONS_TARGET_VALUE_THRESHOLD = 0.5    # target scores at least 0.5 on the feature


def _label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def compute_pros_cons(
    w: np.ndarray,
    x_t: np.ndarray,
    feature_names: list[str],
) -> tuple[list[dict], list[dict]]:
    """Return (pros, cons) lists.

    pros: features where contribution > 5% of total target score, sorted desc
    cons: features with weight < 0.02 AND target_value > 0.5, sorted by target_value desc
    """
    contrib = w * x_t
    total = float(contrib.sum())

    pros: list[dict] = []
    cons: list[dict] = []

    for i, name in enumerate(feature_names):
        contribution = float(contrib[i])
        weight = float(w[i])
        target_value = float(x_t[i])

        if total > 0 and contribution > PROS_CONTRIBUTION_THRESHOLD * total:
            pros.append({
                "feature":      name,
                "weight":       weight,
                "target_value": target_value,
                "contribution": contribution,
                "label":        _label(name),
            })

        if weight < CONS_WEIGHT_THRESHOLD and target_value > CONS_TARGET_VALUE_THRESHOLD:
            cons.append({
                "feature":      name,
                "weight":       weight,
                "target_value": target_value,
                "label":        _label(name),
            })

    pros.sort(key=lambda p: p["contribution"], reverse=True)
    cons.sort(key=lambda c: c["target_value"], reverse=True)
    return pros, cons


def _row_meta(rows: pd.DataFrame, idx: int, keys=("country_iso3", "country_name", "sector", "year")) -> dict:
    row = rows.iloc[int(idx)]
    out: dict = {}
    for k in keys:
        if k in rows.columns:
            val = row[k]
            if hasattr(val, "item"):
                try:
                    val = val.item()
                except Exception:
                    pass
            out[k] = val
    return out


def _top_index(scores: np.ndarray, exclude: Optional[int] = None) -> Optional[int]:
    order = np.argsort(-scores)
    for idx in order:
        if exclude is not None and int(idx) == exclude:
            continue
        return int(idx)
    return None


def _severity_max_weights(feature_names: list[str]) -> np.ndarray:
    severity_set = {"need_scale", "pin", "targeted", "per_capita_need"}
    w = np.zeros(len(feature_names), dtype=float)
    for i, name in enumerate(feature_names):
        if name in severity_set:
            w[i] = 1.0
    s = w.sum()
    if s == 0:
        return default_weight_vector(feature_names)
    return w / s


def _gap_max_weights(feature_names: list[str]) -> np.ndarray:
    w = np.zeros(len(feature_names), dtype=float)
    if "coverage_gap" in feature_names:
        w[feature_names.index("coverage_gap")] = 1.0
        return w
    return default_weight_vector(feature_names)


def compute_counterfactual(
    X: np.ndarray,
    rows: pd.DataFrame,
    target_idx: int,
    solved_w: np.ndarray,
    feature_names: list[str],
    k: int = 3,
) -> dict:
    """Return the deterministic counterfactual struct.

    Sources:
      a) default_top — top row under DEFAULT_WEIGHTS if ≠ target
      b) displaced   — rows that were in top-k under defaults but dropped out under solved_w
      c) archetypes  — top row under severity-max and gap-max weight vectors (if ≠ target)
    """
    w_default = default_weight_vector(feature_names)
    scores_default = X @ w_default
    scores_solved = X @ solved_w

    default_top_idx = _top_index(scores_default, exclude=target_idx)
    default_top = None
    if default_top_idx is not None:
        default_top = {
            **_row_meta(rows, default_top_idx),
            "score": float(scores_default[default_top_idx]),
        }

    order_default = np.argsort(-scores_default)
    order_solved = np.argsort(-scores_solved)
    top_default = set(int(i) for i in order_default[:k])
    top_solved = set(int(i) for i in order_solved[:k])
    displaced_idx = top_default - top_solved - {target_idx}

    def _rank_of(order: np.ndarray, idx: int) -> int:
        pos = int(np.where(order == idx)[0][0])
        return pos + 1

    displaced = []
    for idx in sorted(displaced_idx, key=lambda i: scores_default[i] - scores_solved[i], reverse=True)[:2]:
        displaced.append({
            **_row_meta(rows, idx),
            "default_rank": _rank_of(order_default, idx),
            "new_rank":     _rank_of(order_solved, idx),
            "default_score": float(scores_default[idx]),
            "new_score":     float(scores_solved[idx]),
            "score_delta":   float(scores_solved[idx] - scores_default[idx]),
        })

    w_sev = _severity_max_weights(feature_names)
    w_gap = _gap_max_weights(feature_names)
    scores_sev = X @ w_sev
    scores_gap = X @ w_gap
    sev_idx = _top_index(scores_sev, exclude=target_idx)
    gap_idx = _top_index(scores_gap, exclude=target_idx)

    archetypes = {
        "severity_max": (
            {**_row_meta(rows, sev_idx), "score": float(scores_sev[sev_idx])}
            if sev_idx is not None else None
        ),
        "gap_max": (
            {**_row_meta(rows, gap_idx), "score": float(scores_gap[gap_idx])}
            if gap_idx is not None else None
        ),
    }

    return {
        "default_top": default_top,
        "displaced": displaced,
        "archetypes": archetypes,
    }


def build_ranking_slice(
    X: np.ndarray,
    rows: pd.DataFrame,
    w: np.ndarray,
    target_idx: int,
    top_n: int = 10,
) -> list[dict]:
    """Return the top-n ranked rows, plus the target row if it's outside the top-n."""
    scores = X @ w
    order = np.argsort(-scores)
    picked = list(order[:top_n])
    if target_idx not in picked:
        picked.append(target_idx)

    out = []
    for rank_pos, idx in enumerate(picked, start=1):
        idx = int(idx)
        entry = {
            "rank": int(np.where(order == idx)[0][0]) + 1,
            **_row_meta(rows, idx),
            "score": float(scores[idx]),
        }
        out.append(entry)
    return out


def fallback_explanation(
    target_meta: dict,
    pros: list[dict],
    cons: list[dict],
    counterfactual: Optional[dict],
    short_circuited: bool,
) -> str:
    """Deterministic natural-language string used if the LLM call is disabled/unavailable."""
    target_label = target_meta.get("country_name") or target_meta.get("country_iso3") or "target"
    sector = target_meta.get("sector")
    sector_label = f" × {sector}" if sector and sector != "ALL" else ""

    if short_circuited:
        lead = f"{target_label}{sector_label} is already in the top rank under default scoring."
    else:
        lead = f"To place {target_label}{sector_label} in the top rank, the weights emphasized:"

    pros_s = ", ".join(p["label"] for p in pros[:4]) if pros else "no dominant features"
    cons_s = ", ".join(c["label"] for c in cons[:3]) if cons else ""

    parts = [lead, f"Optimized for: {pros_s}."]
    if cons_s:
        parts.append(f"Not prioritized (despite being high for the target): {cons_s}.")

    if counterfactual:
        default_top = counterfactual.get("default_top")
        if default_top and default_top.get("country_iso3") != target_meta.get("country_iso3"):
            parts.append(f"Under default weights, {default_top.get('country_name') or default_top.get('country_iso3')} would rank first.")
        archetypes = counterfactual.get("archetypes", {}) or {}
        sev = archetypes.get("severity_max")
        if sev and sev.get("country_iso3") != target_meta.get("country_iso3"):
            parts.append(f"Under a severity-max view, {sev.get('country_name') or sev.get('country_iso3')} wins.")

    return " ".join(parts)


def llm_prose(
    target_meta: dict,
    weights: dict[str, float],
    pros: list[dict],
    cons: list[dict],
    counterfactual: Optional[dict],
    short_circuited: bool,
    llm_client=None,
) -> str:
    """Generate a natural-language explanation using Claude.

    The LLM is only a translator — it is given the structured numbers and asked
    to write 2–3 sentences. Falls back to the deterministic template on any error.
    """
    try:
        import anthropic  # local import, optional at runtime
    except ImportError:
        return fallback_explanation(target_meta, pros, cons, counterfactual, short_circuited)

    if llm_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return fallback_explanation(target_meta, pros, cons, counterfactual, short_circuited)
        llm_client = anthropic.Anthropic(api_key=api_key)

    payload = {
        "target":          target_meta,
        "weights":         weights,
        "pros":            pros,
        "cons":            cons,
        "counterfactual":  counterfactual,
        "short_circuited": short_circuited,
    }
    system = (
        "You translate humanitarian prioritization analysis into 2–3 concise sentences. "
        "Ground every claim in the JSON numbers you're given. Do not invent features or "
        "countries. Mention the top 2–3 pros, at most 2 cons if present, and the most "
        "relevant counterfactual (default_top or one archetype) if present. "
        "No markdown, no bullet points, no commentary."
    )
    try:
        msg = llm_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=350,
            system=system,
            messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
            temperature=0,
        )
        text = msg.content[0].text.strip()
        return text or fallback_explanation(target_meta, pros, cons, counterfactual, short_circuited)
    except Exception:
        return fallback_explanation(target_meta, pros, cons, counterfactual, short_circuited)
