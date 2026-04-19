"""
Multi-Criteria Decision Analysis (MCDA) for humanitarian crisis prioritization.

Pipeline:
  1. User describes their mandate in natural language
  2. LLM extracts a weight vector across 12 dimensions
  3. Dimensions combine existing gap metrics + external vulnerability indices
  4. Pro/Con analysis explains why each country should or should not be prioritized
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.loader import get_fact, get_dim_country

PROCESSED_DIR  = Path(__file__).resolve().parents[2] / "data" / "processed"
PRIMARY_MODEL  = "databricks-llama-4-maverick"
FALLBACK_MODEL = "databricks-meta-llama-3-3-70b-instruct"


# ── Dimension registry ────────────────────────────────────────────────────────

DIMENSIONS: dict[str, dict[str, str]] = {
    "need_scale": {
        "label":  "People in Need (Scale)",
        "desc":   "Absolute number of people in humanitarian need — log-normalised.",
        "group":  "Humanitarian Need",
    },
    "funding_gap": {
        "label":  "Funding Gap",
        "desc":   "Share of humanitarian requirements that remain unfunded (1 − coverage ratio).",
        "group":  "Funding",
    },
    "structural_neglect": {
        "label":  "Structural Neglect",
        "desc":   "Proportion of years the crisis has been chronically underfunded (below 30% coverage).",
        "group":  "Funding",
    },
    "trend_worsening": {
        "label":  "Worsening Trend",
        "desc":   "Whether funding coverage is declining year-over-year.",
        "group":  "Funding",
    },
    "targeting_gap": {
        "label":  "Targeting Gap",
        "desc":   "Share of people in need who are not targeted for any humanitarian response.",
        "group":  "Humanitarian Need",
    },
    "water_stress": {
        "label":  "Water Scarcity",
        "desc":   "Baseline water stress: demand relative to available supply. High = severe scarcity.",
        "group":  "Vulnerability",
    },
    "food_insecurity_risk": {
        "label":  "Food Insecurity Risk",
        "desc":   "Structural risk of food insecurity based on production shortfalls and import dependency.",
        "group":  "Vulnerability",
    },
    "displacement_risk": {
        "label":  "Displacement & Refugee Risk",
        "desc":   "Risk from internal displacement and cross-border refugee flows.",
        "group":  "Vulnerability",
    },
    "health_fragility": {
        "label":  "Health System Fragility",
        "desc":   "Weakness of national health infrastructure, capacity, and access.",
        "group":  "Vulnerability",
    },
    "climate_vulnerability": {
        "label":  "Climate Vulnerability",
        "desc":   "Exposure and sensitivity to climate change impacts, adjusted for adaptive capacity.",
        "group":  "Vulnerability",
    },
    "governance_fragility": {
        "label":  "Governance Fragility",
        "desc":   "Weakness of institutions, rule of law, and state capacity to respond to crises.",
        "group":  "Vulnerability",
    },
    "disaster_risk": {
        "label":  "Natural Disaster Exposure",
        "desc":   "Physical exposure to floods, droughts, earthquakes, and cyclones.",
        "group":  "Vulnerability",
    },
    "inform_severity": {
        "label":  "Crisis Severity (INFORM)",
        "desc":   "INFORM Global Crisis Severity Index — composite of impact, conditions of affected people, and crisis complexity. Normalised 0–1 from 1–10 scale.",
        "group":  "Severity",
    },
    "mismatch_score": {
        "label":  "Severity-Funding Mismatch",
        "desc":   "severity × (1 − coverage_ratio): highest where a very severe crisis receives the least funding. Identifies the top-left-quadrant 'most overlooked' crises.",
        "group":  "Severity",
    },
}

DEFAULT_WEIGHTS: dict[str, float] = {k: round(1 / len(DIMENSIONS), 4) for k in DIMENSIONS}
DEFAULT_IMPORTANCE: dict[str, int] = {k: 5 for k in DIMENSIONS}

_DIM_LIST = "\n".join(
    f'  "{k}" ({v["group"]}): {v["desc"]}' for k, v in DIMENSIONS.items()
)

_SCORE_SYSTEM_BASE = f"""You are an expert humanitarian prioritization assistant.

Score the importance of each dimension from 1 to 10.
- 8–10 = explicitly important to the user
- 4–7  = somewhat relevant or implied
- 1–3  = not mentioned or explicitly dismissed
- DEFAULT for any dimension the user does not mention: 1

This matters: scores are squared before normalisation, so a 10 carries 100× the
weight of a 1. Only give high scores to dimensions the user actually cares about.

Dimensions:
{_DIM_LIST}

Rules:
- Return ONLY a valid JSON object mapping each key to an integer 1–10
- No explanation, no markdown fences
- All 14 keys must be present

Example — user says "severity mismatch and chronic neglect":
{{"need_scale":2,"funding_gap":3,"structural_neglect":9,"trend_worsening":2,\
"targeting_gap":1,"water_stress":2,"food_insecurity_risk":2,"displacement_risk":1,\
"health_fragility":1,"climate_vulnerability":2,"governance_fragility":3,"disaster_risk":1,\
"inform_severity":8,"mismatch_score":10}}
"""

SCORE_SYSTEM_INITIAL = _SCORE_SYSTEM_BASE

SCORE_SYSTEM_REFINE = _SCORE_SYSTEM_BASE + """
You will be given the current scores and the user's refinement feedback.
Adjust only what the user asks to change; keep all other scores stable.
"""

NARRATIVE_SYSTEM = """You are a humanitarian analyst assistant. Given a country's data and a user's stated priorities, write two short analytical paragraphs.

Return ONLY a valid JSON object with exactly two keys:
- "why_fund": 2-3 sentences making the strongest case FOR prioritizing this country given the user's mandate. Cite specific numbers (people in need, funding %, years underfunded). Be direct.
- "why_not": 2-3 sentences making the strongest case AGAINST. Call out where this country underperforms relative to other crisis contexts, and name a stronger alternative if one clearly exists.

No markdown, no extra keys, no explanation outside the JSON."""


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _get_client():
    import os
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    host  = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    if not host or not token:
        raise RuntimeError(
            "DATABRICKS_HOST and DATABRICKS_TOKEN are required for AI features. "
            "Add them to your .env file."
        )
    return OpenAI(api_key=token, base_url=f"{host}/serving-endpoints")


def _call_llm(messages: list[dict], max_tokens: int = 512) -> str:
    client = _get_client()
    for model in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=0
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            if model == PRIMARY_MODEL:
                print(f"  ⚠ {PRIMARY_MODEL} unavailable ({exc}), trying fallback")
                continue
            raise


# ── Weight helpers ────────────────────────────────────────────────────────────

def weights_from_scores(scores: dict[str, int]) -> dict[str, float]:
    """Square-normalise 1–10 importance scores into a weight vector summing to 1."""
    sq = {k: v ** 2 for k, v in scores.items()}
    total = sum(sq.values()) or 1.0
    return {k: round(v / total, 4) for k, v in sq.items()}


# ── Weight extraction ─────────────────────────────────────────────────────────

def extract_weights(
    prompt: str,
    previous_scores: dict[str, int] | None = None,
) -> tuple[dict[str, float], dict[str, int], str]:
    """
    NL → importance scores (1–10) → normalised weight vector.

    Parameters
    ----------
    prompt           : User's natural-language priority description or refinement.
    previous_scores  : If provided, treated as a follow-up — LLM adjusts existing
                       scores rather than starting fresh.

    Returns
    -------
    weights          : Normalised floats summing to 1.0 (used for math).
    importance_scores: Raw 1–10 integers (shown to users).
    interpretation   : Human-readable summary of the top priorities.
    """
    if previous_scores:
        scores_text = "\n".join(
            f"  {k}: {previous_scores.get(k, 5)}/10  ({DIMENSIONS[k]['label']})"
            for k in DIMENSIONS
        )
        user_msg = f"Current importance scores:\n{scores_text}\n\nRefinement request: {prompt}"
        system   = SCORE_SYSTEM_REFINE
    else:
        user_msg = prompt
        system   = SCORE_SYSTEM_INITIAL

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_msg},
    ]
    raw = _call_llm(messages)
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip().strip("`")

    try:
        parsed: dict = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        parsed = json.loads(m.group()) if m else {}

    # Clamp to [1, 10] integers; fall back to previous or 1 (not 5 — see prompt)
    fallback = previous_scores or {k: 1 for k in DIMENSIONS}
    importance: dict[str, int] = {
        k: max(1, min(10, int(round(float(parsed.get(k, fallback.get(k, 1)))))))
        for k in DIMENSIONS
    }

    weights = weights_from_scores(importance)

    top3 = sorted(importance.items(), key=lambda x: -x[1])[:3]
    interpretation = (
        "Top priorities: "
        + ", ".join(f"{DIMENSIONS[k]['label']} ({v}/10)" for k, v in top3)
        + (". Scores adjusted from previous." if previous_scores else ".")
    )
    return weights, importance, interpretation


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_vulnerability() -> tuple[pd.DataFrame, set[str]]:
    """Returns (vuln_df, set_of_iso3_with_real_data)."""
    path = PROCESSED_DIR / "dim_vulnerability.parquet"
    df = pd.read_parquet(path) if path.exists() else _builtin_vulnerability()
    return df, set(df["country_iso3"].tolist())


def _compute_temporal(fact: pd.DataFrame, sector: str = "ALL") -> pd.DataFrame:
    """Per-country temporal features. sector-specific when provided."""
    base = fact[fact["sector"] == sector][["country_iso3", "year", "coverage_ratio"]].copy()
    rows = []
    for iso3, grp in base.groupby("country_iso3"):
        cov      = grp.set_index("year")["coverage_ratio"].dropna().sort_index()
        mean_gap = float(1 - cov.mean()) if len(cov) >= 2 else np.nan
        slope    = float(np.polyfit(cov.index.astype(float), cov.values, 1)[0]) \
                   if len(cov) >= 2 else np.nan
        rows.append({"country_iso3": iso3, "mean_funding_gap": mean_gap,
                     "coverage_slope": slope, "n_years": len(cov)})
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["country_iso3", "mean_funding_gap", "coverage_slope", "n_years"])


# ── Scoring ───────────────────────────────────────────────────────────────────

SECTOR_LABELS: dict[str, str] = {
    "ALL":  "All Sectors",
    "WSH":  "WASH",
    "FSC":  "Food Security",
    "HEA":  "Health",
    "NUT":  "Nutrition",
    "PRO":  "Protection",
    "SHL":  "Shelter",
    "EDU":  "Education",
    "CCCM": "Camp Coordination",
    "ETC":  "Emergency Telecom",
    "LOG":  "Logistics",
}


def score_all(
    weights: dict[str, float],
    year: int = 2024,
    sector: str = "ALL",
    compare_year: int | None = None,
) -> pd.DataFrame:
    """
    Score every country for the given year using the weight vector.

    sector       : filter to a specific cluster sector (default "ALL").
    compare_year : if given, also scores that year and adds rank_delta
                   (positive = rank improved vs. compare_year).
    """
    fact  = get_fact()
    dim_c = get_dim_country()
    vuln, vuln_iso3s = _load_vulnerability()

    from src.data.inform_loader import get_inform_severity
    inform = get_inform_severity()
    inform_year = (
        inform[inform["year"] == year][["country_iso3", "inform_severity"]]
        if year in inform["year"].values
        else inform.groupby("country_iso3")["inform_severity"].max().reset_index()
    )

    base = (
        fact[(fact["sector"] == sector) & (fact["year"] == year)]
        .merge(dim_c[["country_iso3", "country_name", "continent", "region_name"]],
               on="country_iso3", how="left")
        .merge(vuln, on="country_iso3", how="left")
        .merge(inform_year, on="country_iso3", how="left")
    )

    temporal = _compute_temporal(fact, sector)
    base     = base.merge(temporal, on="country_iso3", how="left")

    # Flag whether vulnerability scores are from a real source or a 0.5 default
    base["has_vulnerability"] = base["country_iso3"].isin(vuln_iso3s)

    # ── Humanitarian dimensions ───────────────────────────────────────────────
    log_pin = np.log1p(base["pin"].clip(lower=0).fillna(0))
    base["need_scale"] = log_pin / (log_pin.max() or 1.0)

    # funding_gap: no funding received = 0 is a fact → gap = 1.0 when untracked.
    base["funding_gap"] = (1 - base["coverage_ratio"].clip(0, 1)).fillna(1.0)

    base["structural_neglect"] = base["mean_funding_gap"]            # NaN if <2 years

    decline = (-base["coverage_slope"]).clip(lower=0)                # NaN if <2 years
    max_decline = decline.max()
    base["trend_worsening"] = (decline / max_decline) if (max_decline and max_decline > 0) else np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        tgt_gap = 1 - (base["targeted"] / base["pin"]).clip(0, 1)
    base["targeting_gap"] = tgt_gap                                  # NaN if no PIN/targeted

    # ── External vulnerability columns — NaN if not available ────────────────
    for dim in ("water_stress", "food_insecurity_risk", "displacement_risk",
                "health_fragility", "climate_vulnerability",
                "governance_fragility", "disaster_risk"):
        if dim not in base.columns:
            base[dim] = np.nan

    # ── INFORM severity & mismatch ────────────────────────────────────────────
    # No fill — countries without INFORM data stay NaN; mismatch_score is NaN too.
    base["mismatch_score"] = (base["inform_severity"] * base["funding_gap"]).clip(0, 1)

    # ── Weighted MCDA score ───────────────────────────────────────────────────
    # For each country, sum only over dimensions that have real data;
    # normalise by the actual weight sum for that country so NaN dims don't penalise it.
    def _row_score(row: pd.Series) -> float:
        num = denom = 0.0
        for dim in DIMENSIONS:
            w = weights.get(dim, 0)
            v = row.get(dim)
            if w > 0 and v is not None and not pd.isna(v):
                num   += w * float(v)
                denom += w
        return num / denom if denom > 0 else 0.0

    base["mcda_score"] = base.apply(_row_score, axis=1)

    # Percentile rank per dimension (1.0 = highest need/risk)
    for dim in DIMENSIONS:
        base[f"pct_{dim}"] = base[dim].rank(pct=True, ascending=True)

    base = base.sort_values("mcda_score", ascending=False).reset_index(drop=True)
    base["mcda_rank"] = range(1, len(base) + 1)

    for col in ("pin", "coverage_ratio", "requirements_usd", "fts_funding_usd",
                "targeted", "years_underfunded", "coverage_slope", "n_years"):
        if col not in base.columns:
            base[col] = None

    # ── Year-over-year rank delta ─────────────────────────────────────────────
    if compare_year is not None:
        prev = score_all(weights=weights, year=compare_year, sector=sector)
        prev_map = prev.set_index("country_iso3")["mcda_rank"].to_dict()
        base["prev_rank"]   = base["country_iso3"].map(prev_map)
        base["rank_delta"]  = (base["prev_rank"] - base["mcda_rank"]).where(
            base["prev_rank"].notna()
        )
    else:
        base["rank_delta"] = np.nan

    return base


# ── Pro/Con analysis ──────────────────────────────────────────────────────────

def build_pro_con(
    country_iso3: str,
    scored: pd.DataFrame,
    weights: dict[str, float],
    user_prompt: str = "",
) -> dict[str, Any]:
    """
    For a single country, return structured pro-prioritize and caution cards
    based on its percentile rank on each weighted dimension.
    """
    row = scored[scored["country_iso3"] == country_iso3]
    if row.empty:
        return {"error": f"{country_iso3} not in scored dataset for this year"}
    row = row.iloc[0]

    pros, cons, neutrals = [], [], []

    for dim, meta in DIMENSIONS.items():
        w   = weights.get(dim, 0)
        if w < 0.03:
            continue  # dimension effectively ignored by user

        pct = float(row.get(f"pct_{dim}", 0.5))
        val = row.get(dim)
        val_f = float(val) if val is not None and not pd.isna(val) else None

        card: dict[str, Any] = {
            "dimension":  dim,
            "label":      meta["label"],
            "group":      meta["group"],
            "weight":     round(w, 3),
            "value":      round(val_f, 3) if val_f is not None else None,
            "percentile": round(pct, 2),
            "narrative":  _dimension_narrative(dim, val_f, pct, w, meta, row),
        }

        if pct >= 0.67:
            pros.append(card)
        elif pct <= 0.33:
            # Name the country that outperforms on this dimension
            if dim in scored.columns:
                col = scored[dim].dropna()
                if not col.empty:
                    best = scored.loc[col.idxmax()]
                    if best["country_iso3"] != country_iso3:
                        bv = float(best[dim])
                        card["better_country"] = {
                            "country_name": str(best["country_name"]),
                            "country_iso3": str(best["country_iso3"]),
                            "value":        round(bv, 3) if math.isfinite(bv) else None,
                            "mcda_rank":    int(best["mcda_rank"]),
                        }
            cons.append(card)
        else:
            neutrals.append(card)

    for lst in (pros, cons, neutrals):
        lst.sort(key=lambda c: -c["weight"])

    # LLM split narrative
    narrative = _llm_summary(country_iso3, row, weights, user_prompt, pros, cons)

    return {
        "country_iso3": country_iso3,
        "country_name": str(row.get("country_name", country_iso3)),
        "continent":    str(row.get("continent", "")),
        "region_name":  str(row.get("region_name", "")),
        "mcda_score":   round(float(row["mcda_score"]), 4),
        "mcda_rank":    int(row["mcda_rank"]),
        "n_countries":  len(scored),
        "why_fund":     narrative["why_fund"],
        "why_not":      narrative["why_not"],
        "pros":         pros,
        "cons":         cons,
        "neutral":      neutrals,
        # Reference metrics
        "pin":              _safe(row.get("pin")),
        "coverage_ratio":   _safe(row.get("coverage_ratio")),
        "requirements_usd": _safe(row.get("requirements_usd")),
        "years_underfunded": _safe(row.get("years_underfunded")),
        # Raw dimension scores — lets the frontend draw score decomposition
        # for any country regardless of whether it's in the top-N ranked list
        "dim_scores": {
            dim: _safe(row.get(dim))
            for dim in DIMENSIONS
        },
    }


def _safe(v) -> float | None:
    try:
        f = float(v)
        return None if pd.isna(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _dimension_narrative(
    dim: str, val: float | None, pct: float, weight: float, meta: dict,
    row: "pd.Series | None" = None,
) -> str:
    """Return a single informative sentence for a pro/con card."""
    high = pct >= 0.67
    low  = pct <= 0.33
    pct_str = (
        f"top {round((1-pct)*100):.0f}% globally" if high
        else f"bottom {round(pct*100):.0f}% globally" if low
        else "mid-range globally"
    )

    def _raw(field: str) -> float | None:
        if row is None:
            return None
        v = row.get(field)
        try:
            f = float(v)
            return None if pd.isna(f) else f
        except (TypeError, ValueError):
            return None

    if dim == "need_scale":
        pin = _raw("pin")
        if pin is not None:
            m = pin / 1e6
            return (
                f"{m:.1f}M people require humanitarian assistance — {'one of the largest active crises globally' if high else 'a relatively smaller caseload compared to other crises' if low else 'a mid-sized caseload'}."
            )
        return f"Scale of need is {pct_str}."

    if dim == "funding_gap":
        cov = _raw("coverage_ratio")
        req = _raw("requirements_usd")
        if cov is not None:
            funded_pct = cov * 100
            gap_pct    = (1 - cov) * 100
            req_str    = f" against ${req/1e6:.0f}M in requirements" if req else ""
            return (
                f"Only {funded_pct:.0f}% of needs funded{req_str} — a {gap_pct:.0f}% gap." if high
                else f"Funding coverage stands at {funded_pct:.0f}% — relatively well-resourced compared to most crises." if low
                else f"Funding coverage is {funded_pct:.0f}% — partially met but with meaningful shortfalls."
            )
        return f"Funding gap is {pct_str}."

    if dim == "structural_neglect":
        yu = _raw("years_underfunded")
        ny = _raw("n_years")
        if ny is None or ny < 2:
            return "Not enough years on record to assess chronic underfunding patterns."
        if yu is None:
            return f"Chronic underfunding is {pct_str}, but the count of underfunded years is not reported."
        return (
            f"Below 30% coverage for {int(yu)} of the last {int(ny)} years on record — {'chronically neglected over time' if high else 'relatively consistent funding history' if low else 'intermittent underfunding'}."
        )

    if dim == "trend_worsening":
        ny = _raw("n_years")
        if ny is None or ny < 2:
            return "Not enough years on record to calculate a funding trend."
        slope = _raw("coverage_slope")
        if slope is not None:
            dir_str = "declining" if slope < 0 else "improving"
            rate    = abs(slope) * 100
            return (
                f"Funding coverage is {dir_str} by ~{rate:.1f}pp per year — {'the gap is widening' if high else 'the situation is stabilising or recovering' if low else 'a modest trend change'}."
            )
        return "Trend data not available."

    if dim == "targeting_gap":
        tgt = _raw("targeted")
        pin = _raw("pin")
        if pin is None or tgt is None:
            return "Targeting data not reported — it is unknown what share of people in need are reached by the response."
        if pin > 0:
            tgt_pct = min(tgt / pin, 1) * 100
            return (
                f"Only {tgt_pct:.0f}% of people in need are targeted for any humanitarian response — {'a major gap in programme reach' if high else 'relatively comprehensive targeting' if low else 'partial programme coverage'}."
            )

    if dim == "water_stress":
        if val is None:
            return "Water stress data not available for this country."
        return (
            "Faces extreme water stress — demand regularly exceeds renewable supply, compounding protection and food security risks." if high
            else "Water resources are relatively adequate — scarcity is not a primary structural driver here." if low
            else "Moderate water stress — a background vulnerability rather than a critical driver."
        )

    if dim == "food_insecurity_risk":
        if val is None:
            return "Food insecurity risk data not available for this country."
        return (
            "High structural food insecurity risk — production shortfalls, import dependency, and market fragility leave the population exposed." if high
            else "Structural food security risk is relatively lower compared to the most vulnerable countries." if low
            else "Moderate food insecurity risk — partially offset by production capacity or market access."
        )

    if dim == "displacement_risk":
        if val is None:
            return "Displacement risk data not available for this country."
        return (
            "High displacement and refugee risk — large populations uprooted internally or across borders, concentrating needs and straining host systems." if high
            else "Displacement levels are relatively contained compared to other active crises." if low
            else "Some displacement pressure, but not among the most acute cases globally."
        )

    if dim == "health_fragility":
        if val is None:
            return "Health system fragility data not available for this country."
        return (
            "Severely fragile health system — limited infrastructure, workforce, and supply chains cannot absorb the additional burden of a humanitarian crisis." if high
            else "Health system is comparatively more resilient — better placed to manage crisis-related disease burden." if low
            else "Health system fragility is moderate — capacity exists but is under stress."
        )

    if dim == "climate_vulnerability":
        if val is None:
            return "Climate vulnerability data not available for this country."
        return (
            "Highly exposed to climate shocks with very limited adaptive capacity — extreme weather events directly amplify humanitarian needs." if high
            else "Lower structural climate vulnerability — less exposed or better equipped to adapt to climate variability." if low
            else "Moderate climate vulnerability — exposed but with some adaptive capacity."
        )

    if dim == "governance_fragility":
        if val is None:
            return "Governance fragility data not available for this country."
        return (
            "Weak institutions and limited state capacity create barriers to humanitarian access, coordination, and durable solutions." if high
            else "Governance capacity is comparatively stronger — state systems can partially support crisis response." if low
            else "Governance is under strain but functional enough to permit humanitarian operations."
        )

    if dim == "disaster_risk":
        if val is None:
            return "Natural disaster exposure data not available for this country."
        return (
            "High physical exposure to floods, droughts, earthquakes, or cyclones — natural hazards repeatedly trigger or deepen the crisis." if high
            else "Lower exposure to acute natural hazard risk relative to other crisis-affected countries." if low
            else "Some natural hazard exposure, but not among the highest-risk countries globally."
        )

    if dim == "inform_severity":
        sev_raw = val * 10 if val is not None else None
        sev_str = f"{sev_raw:.1f}/10" if sev_raw is not None else "unknown"
        return (
            f"INFORM Severity score of {sev_str} — among the most severe active crises globally, with acute impact, dire conditions for affected people, and high crisis complexity." if high
            else f"INFORM Severity score of {sev_str} — relatively lower crisis severity compared to the most acute emergencies." if low
            else f"INFORM Severity score of {sev_str} — a significant but mid-range crisis on the global severity scale."
        )

    if dim == "mismatch_score":
        return (
            "This is a top-left-quadrant crisis: high severity combined with very low funding coverage — the exact mismatch that leaves the most severe crises most overlooked." if high
            else "Despite some severity, this crisis receives relatively adequate funding proportional to its scale — the mismatch is lower than in other contexts." if low
            else "Moderate severity-funding mismatch — partially funded relative to its severity, but the gap remains significant."
        )

    # Fallback
    if val is None:
        return f"{meta['label']}: data not available for this country."
    return f"{meta['label']} is {pct_str} (score {val:.2f}, weight {weight:.0%})."


def _llm_summary(
    iso3: str,
    row: pd.Series,
    weights: dict[str, float],
    user_prompt: str,
    pros: list,
    cons: list,
) -> dict[str, str]:
    top_pros = ", ".join(c["label"] for c in pros[:3]) or "none"
    top_cons = ", ".join(c["label"] for c in cons[:3]) or "none"
    pin_m    = f"{row.get('pin', 0) / 1e6:.1f}M" if row.get("pin") else "unknown"
    cov      = f"{row.get('coverage_ratio', 0) * 100:.0f}%" if row.get("coverage_ratio") is not None else "unknown"
    req      = f"${row.get('requirements_usd', 0) / 1e6:.0f}M" if row.get("requirements_usd") else "unknown"
    rank     = int(row['mcda_rank'])
    n        = row.get('n_countries', '?')

    user_msg = f"""User mandate: {user_prompt or 'balanced across all dimensions'}

Country: {row.get('country_name', iso3)} ({iso3})
MCDA rank: {rank} of {n} crisis countries
People in need: {pin_m}
Funding coverage: {cov} of {req} in requirements
Strongest pro-prioritization dimensions: {top_pros}
Dimensions where this country is weaker: {top_cons}

Return a JSON object with "why_fund" and "why_not" — each 2-3 sentences."""

    fallback_name = str(row.get('country_name', iso3))
    try:
        messages = [
            {"role": "system", "content": NARRATIVE_SYSTEM},
            {"role": "user",   "content": user_msg},
        ]
        raw = _call_llm(messages, max_tokens=350)
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip().strip("`")
        parsed = json.loads(raw)
        return {
            "why_fund": str(parsed.get("why_fund", "")),
            "why_not":  str(parsed.get("why_not",  "")),
        }
    except Exception:
        return {
            "why_fund": (
                f"{fallback_name} ranks #{rank} of {n} crisis countries on your weighted criteria. "
                f"{'Key strengths: ' + top_pros + '.' if pros else 'No strong signals on your priority dimensions.'}"
            ),
            "why_not": (
                f"{'Areas of concern: ' + top_cons + '.' if cons else 'No significant caution signals.'} "
                f"Consider whether higher-ranked alternatives better match your mandate."
            ),
        }


# ── Built-in vulnerability dataset ────────────────────────────────────────────
# Representative [0–1] scores derived from WRI Aqueduct, ND-GAIN, and INFORM Risk Index.
# 0 = low risk, 1 = extreme risk.

def _builtin_vulnerability() -> pd.DataFrame:
    # fmt: off
    DATA = [
        #  iso3   water  food   disp   health climate  gov    disaster
        ("AFG",  0.62,  0.82,  0.92,  0.85,  0.62,  0.90,  0.55),
        ("BGD",  0.42,  0.55,  0.60,  0.58,  0.82,  0.55,  0.88),
        ("BFA",  0.50,  0.78,  0.75,  0.80,  0.72,  0.72,  0.52),
        ("CAF",  0.22,  0.90,  0.88,  0.92,  0.60,  0.92,  0.38),
        ("CMR",  0.25,  0.60,  0.65,  0.70,  0.58,  0.60,  0.48),
        ("COD",  0.08,  0.80,  0.85,  0.88,  0.55,  0.88,  0.45),
        ("COG",  0.15,  0.52,  0.45,  0.62,  0.52,  0.58,  0.42),
        ("COL",  0.20,  0.42,  0.55,  0.40,  0.50,  0.42,  0.62),
        ("EGY",  0.95,  0.55,  0.38,  0.48,  0.65,  0.45,  0.30),
        ("ETH",  0.38,  0.82,  0.80,  0.80,  0.68,  0.70,  0.60),
        ("GTM",  0.22,  0.58,  0.48,  0.52,  0.62,  0.50,  0.72),
        ("HND",  0.25,  0.55,  0.45,  0.50,  0.65,  0.48,  0.68),
        ("HTI",  0.38,  0.82,  0.72,  0.85,  0.70,  0.85,  0.90),
        ("IRN",  0.78,  0.42,  0.35,  0.40,  0.62,  0.55,  0.60),
        ("IRQ",  0.80,  0.60,  0.70,  0.58,  0.68,  0.72,  0.40),
        ("JOR",  0.88,  0.60,  0.55,  0.42,  0.60,  0.38,  0.25),
        ("KEN",  0.42,  0.62,  0.68,  0.58,  0.65,  0.55,  0.58),
        ("LBN",  0.80,  0.72,  0.70,  0.48,  0.52,  0.72,  0.30),
        ("LBY",  0.88,  0.48,  0.58,  0.52,  0.58,  0.82,  0.22),
        ("MLI",  0.55,  0.82,  0.82,  0.88,  0.72,  0.80,  0.50),
        ("MMR",  0.22,  0.65,  0.80,  0.68,  0.70,  0.78,  0.75),
        ("MOZ",  0.22,  0.72,  0.65,  0.78,  0.72,  0.60,  0.80),
        ("NER",  0.72,  0.88,  0.75,  0.90,  0.78,  0.68,  0.48),
        ("NGA",  0.28,  0.70,  0.72,  0.72,  0.60,  0.70,  0.55),
        ("PAK",  0.80,  0.68,  0.68,  0.62,  0.70,  0.62,  0.72),
        ("PRK",  0.32,  0.72,  0.25,  0.70,  0.48,  0.95,  0.38),
        ("PSE",  0.90,  0.85,  0.88,  0.62,  0.55,  0.75,  0.20),
        ("SDN",  0.68,  0.82,  0.88,  0.82,  0.70,  0.82,  0.52),
        ("SOM",  0.58,  0.90,  0.85,  0.90,  0.75,  0.90,  0.55),
        ("SSD",  0.28,  0.90,  0.90,  0.92,  0.68,  0.92,  0.45),
        ("SYR",  0.82,  0.85,  0.92,  0.80,  0.58,  0.88,  0.32),
        ("TCD",  0.42,  0.85,  0.80,  0.90,  0.70,  0.80,  0.48),
        ("TUN",  0.72,  0.42,  0.30,  0.38,  0.52,  0.40,  0.28),
        ("TZA",  0.30,  0.60,  0.55,  0.65,  0.68,  0.52,  0.55),
        ("UKR",  0.32,  0.30,  0.78,  0.38,  0.32,  0.48,  0.30),
        ("VEN",  0.22,  0.72,  0.70,  0.68,  0.45,  0.72,  0.48),
        ("YEM",  0.92,  0.90,  0.90,  0.88,  0.60,  0.85,  0.30),
        ("ZWE",  0.35,  0.72,  0.55,  0.70,  0.65,  0.68,  0.58),
    ]
    # fmt: on
    cols = [
        "country_iso3", "water_stress", "food_insecurity_risk",
        "displacement_risk", "health_fragility", "climate_vulnerability",
        "governance_fragility", "disaster_risk",
    ]
    return pd.DataFrame(DATA, columns=cols)
