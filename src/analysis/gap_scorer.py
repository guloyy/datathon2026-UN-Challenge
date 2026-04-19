"""
Humanitarian gap scorer: explicit formula + ensemble rank fusion + PCA visualization.

Scoring pipeline
----------------
1. CORE GAP SCORE (theory-driven, auditable):
     gap_component = (1 - coverage_ratio) × log(1+pin) / max(log(1+pin))
   Interpretation: unmet need weighted by scale. Zero if fully funded or no PIN.

2. STRUCTURAL MULTIPLIER (chronic neglect bonus):
     structural_multiplier = 1 + α × (years_below_30pct / total_years)
   α is user-adjustable (default 0.5). A crisis underfunded every year gets 1.5×.

3. CONFIDENCE WEIGHT (data quality discount):
     high   → 1.00  (HNO or HRP confirmed)
     medium → 0.85  (CBPF/CERF signals only)
     none   → 0.70  (no crisis_plan record)

4. FINAL SCORE:
     overlooked_score = gap_component × structural_multiplier × confidence_weight

5. ENSEMBLE VALIDITY CHECK:
   Four independent rankings are computed and fused via Borda count:
     R1 — coverage_ratio (lower = more overlooked)
     R2 — gap_component  (higher = more overlooked)
     R3 — structural score: years_underfunded + decline trend
     R4 — targeting gap: (1 - targeted/pin)
   Borda score = sum of (N - rank_i) across all four methods.
   Countries in the top quartile of all four rankings are flagged as "robust".

6. PCA (visualization only — NOT used for scoring):
   Projects countries from feature space to 2D for a similarity map.
   PC1 ≈ overall crisis severity, PC2 ≈ structural vs acute gap.
   Color by continent, size by gap_score to see regional patterns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data.loader import get_fact, get_dim_country, get_dim_crisis_plan

UNDERFUNDED_THRESHOLD = 0.30
MIN_YEARS_FOR_TREND   = 2

CONFIDENCE_WEIGHTS = {"high": 1.00, "medium": 0.85}
CONFIDENCE_DEFAULT = 0.70

# Default balanced weights — each key is a scoring dimension [0, 1]
DEFAULT_WEIGHTS: dict[str, float] = {
    "scale":      0.25,   # log(1+pin) — absolute size of humanitarian need
    "gap":        0.25,   # 1 - coverage_ratio — funding shortfall
    "structural": 0.25,   # years below 30% coverage — chronic neglect
    "trend":      0.25,   # worsening coverage trajectory
}

# Persona presets — genuinely different analytical lenses
PERSONA_WEIGHTS: dict[str, dict[str, float]] = {
    "balanced": DEFAULT_WEIGHTS,
    "donor": {            # where sustained investment is missing most
        "scale":      0.10,
        "gap":        0.30,
        "structural": 0.40,
        "trend":      0.20,
    },
    "field": {            # absolute scale of need right now
        "scale":      0.50,
        "gap":        0.35,
        "structural": 0.10,
        "trend":      0.05,
    },
    "pooled": {           # gap + persistence — where pooled funds matter most
        "scale":      0.15,
        "gap":        0.35,
        "structural": 0.35,
        "trend":      0.15,
    },
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return get_fact(), get_dim_country(), get_dim_crisis_plan()


# ── Temporal features ─────────────────────────────────────────────────────────

def _temporal(fact: pd.DataFrame) -> pd.DataFrame:
    base = fact[fact["sector"] == "ALL"][["country_iso3", "year", "coverage_ratio"]].copy()
    rows = []
    for iso3, grp in base.groupby("country_iso3"):
        cov = grp.set_index("year")["coverage_ratio"].dropna().sort_index()
        # years_underfunded kept for display/explanation; structural_score uses mean gap
        yu       = int((cov < UNDERFUNDED_THRESHOLD).sum())
        mean_gap = float(1 - cov.mean()) if len(cov) >= 2 else np.nan
        slope    = float(np.polyfit(cov.index.astype(float), cov.values, 1)[0]) \
                   if len(cov) >= MIN_YEARS_FOR_TREND else np.nan
        rows.append({
            "country_iso3":      iso3,
            "years_underfunded": yu,
            "mean_funding_gap":  mean_gap,
            "coverage_slope":    slope,
            "n_coverage_years":  len(cov),
        })
    return pd.DataFrame(rows)


# ── Main scoring ──────────────────────────────────────────────────────────────

def score_crises(
    year: int = 2024,
    weights: dict[str, float] | None = None,
    top_n: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Score and rank countries for the given year using a weighted sum of four
    normalised dimensions, multiplied by a data-quality confidence weight.

    Parameters
    ----------
    year    : Reference year (2019–2025).
    weights : Dict with keys scale | gap | structural | trend.
              Values are non-negative floats; they are normalised internally.
              Pass None to use DEFAULT_WEIGHTS (balanced).
    top_n   : Return only the top-N countries if set.

    Returns
    -------
    df   : DataFrame sorted by overlooked_score descending.
    meta : Dict with methodology details and PCA info.
    """
    fact, dim_c, dim_p = _load()
    temporal = _temporal(fact)

    # ── Point-in-time slice ───────────────────────────────────────────────────
    base = (
        fact[(fact["sector"] == "ALL") & (fact["year"] == year)]
        .merge(dim_c[["country_iso3", "country_name", "continent", "region", "region_name"]],
               on="country_iso3", how="left")
        .merge(dim_p[["country_iso3", "year", "crisis_confidence"]]
               .query("year == @year"),
               on=["country_iso3", "year"], how="left")
        .merge(temporal, on="country_iso3", how="left")
    )

    n = len(base)
    total_years = fact["year"].nunique()

    # ── Resolve weights ───────────────────────────────────────────────────────
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        for k, v in weights.items():
            if k in w:
                w[k] = float(v)
    total_w = sum(w.values()) or 1.0

    # ── 1. Normalised scoring dimensions (all in [0, 1]) ─────────────────────
    log_pin = np.log1p(base["pin"].clip(lower=0).fillna(0))
    pin_max = log_pin.max() or 1.0
    base["need_scale"] = log_pin / pin_max                           # scale

    # funding_gap: no funding received = 0 is a fact → gap = 1.0 when untracked.
    base["funding_gap"] = (1 - base["coverage_ratio"].clip(0, 1)).fillna(1.0)

    base["structural_score"] = base["mean_funding_gap"]              # NaN if <2 years data

    decline = (-base["coverage_slope"]).clip(lower=0)                # NaN if <2 years data
    max_decline = decline.max()
    base["trend_score"] = (decline / max_decline) if (max_decline and max_decline > 0) else np.nan

    # ── 2. Weighted sum — skip NaN dimensions per country ────────────────────
    base["gap_component"] = base["funding_gap"] * base["need_scale"]  # kept for Borda

    dim_cols = {
        "scale": "need_scale", "gap": "funding_gap",
        "structural": "structural_score", "trend": "trend_score",
    }
    def _weighted_score(row: pd.Series) -> float:
        num = denom = 0.0
        for key, col in dim_cols.items():
            wt = w[key]
            v  = row[col]
            if wt > 0 and pd.notna(v):
                num   += wt * float(v)
                denom += wt
        return num / denom if denom > 0 else 0.0

    base["overlooked_raw"] = base.apply(_weighted_score, axis=1)

    # ── 3. Confidence weight (data quality discount) ──────────────────────────
    base["confidence_weight"] = (
        base["crisis_confidence"]
        .map(CONFIDENCE_WEIGHTS)
        .fillna(CONFIDENCE_DEFAULT)
    )

    # ── 4. Final score ────────────────────────────────────────────────────────
    base["overlooked_score"] = base["overlooked_raw"] * base["confidence_weight"]

    # ── 5. Ensemble rankings (Borda count) ────────────────────────────────────
    # R1: coverage ratio — lower is worse
    base["r1"] = base["coverage_ratio"].fillna(0).rank(ascending=True,  method="min")
    # R2: gap component — higher is worse
    base["r2"] = base["gap_component"].rank(ascending=False, method="min")
    # R3: structural — more years underfunded + worsening trend
    trend_neg = (-base["coverage_slope"]).fillna(0)   # flip: positive = worsening
    structural = base["years_underfunded"].fillna(0) + trend_neg.clip(lower=0) * total_years
    base["r3"] = structural.rank(ascending=False, method="min")
    # R4: targeting gap — lower targeted/pin is worse
    with np.errstate(invalid="ignore", divide="ignore"):
        tgt_gap = 1 - (base["targeted"] / base["pin"]).clip(0, 1)
    base["r4"] = tgt_gap.rank(ascending=False, method="min", na_option="bottom")

    # Borda: higher = more overlooked
    base["borda_score"] = (
        (n - base["r1"]) + (n - base["r2"]) + (n - base["r3"]) + (n - base["r4"])
    )
    base["borda_rank"] = base["borda_score"].rank(ascending=False, method="min").astype(int)

    # Robust flag: top quartile in ALL four rankings
    q = n // 4
    base["robust"] = (
        (base["r1"] <= q) & (base["r2"] <= q) &
        (base["r3"] <= q) & (base["r4"] <= q)
    )

    # ── 6. PCA (visualization only) ───────────────────────────────────────────
    def _safe_median(s: pd.Series, fallback: float = 0.0) -> float:
        m = s.median()
        return fallback if pd.isna(m) else float(m)

    pca_features = pd.DataFrame({
        "scale":             base["need_scale"].fillna(0),
        "gap":               base["funding_gap"].fillna(_safe_median(base["funding_gap"])),
        "targeting_gap":     tgt_gap.fillna(_safe_median(tgt_gap, 0.5)),
        "years_underfunded": base["years_underfunded"].fillna(0) / max(total_years, 1),
        "trend_gap":         (-base["coverage_slope"]).fillna(0).clip(lower=-1, upper=1),
    }).fillna(0)  # final guard — PCA cannot tolerate any NaN

    explained_var: dict = {}
    loadings:      dict = {}
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(pca_features.values)

        n_components = min(3, n, pca_features.shape[1])
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(X_scaled)

        base["pc1"] = coords[:, 0]
        base["pc2"] = coords[:, 1] if n_components > 1 else 0.0

        explained_var = dict(zip(
            [f"PC{i+1}" for i in range(n_components)],
            pca.explained_variance_ratio_.round(4).tolist(),
        ))
        loadings = {
            f"PC{i+1}": dict(zip(pca_features.columns, pca.components_[i].round(3).tolist()))
            for i in range(n_components)
        }
    except Exception as pca_err:
        print(f"  ⚠ PCA skipped: {pca_err}")
        base["pc1"] = 0.0
        base["pc2"] = 0.0

    # ── 7. Explanation ────────────────────────────────────────────────────────
    base["explanation"] = base.apply(
        lambda r: _explain(r, total_years), axis=1
    )

    # ── 8. Data completeness flag ─────────────────────────────────────────────
    base["data_complete"] = (
        base["pin"].notna() & base["coverage_ratio"].notna() & base["targeted"].notna()
    )

    # ── Sort and trim ─────────────────────────────────────────────────────────
    base = base.sort_values("overlooked_score", ascending=False).reset_index(drop=True)
    base["rank"] = range(1, len(base) + 1)

    if top_n is not None:
        base = base.head(top_n)

    robust_count = int(base["robust"].sum())
    meta = {
        "year":                year,
        "weights":             {k: round(v / total_w, 3) for k, v in w.items()},
        "total_years_in_data": total_years,
        "n_countries":         len(base),
        "robust_count":        robust_count,
        "pca_explained_var":   explained_var,
        "pca_loadings":        loadings,
        "methodology": {
            "dimensions":        "scale | gap | structural | trend  (each normalised to [0,1])",
            "overlooked_score":  "weighted_sum(dimensions) × confidence_weight",
            "confidence_weight": "high=1.0, medium=0.85, unknown=0.70",
            "ensemble":          "Borda count over 4 rankings: coverage_ratio, gap_component, structural, targeting_gap",
        },
    }

    return base, meta


# ── Explanation generator ─────────────────────────────────────────────────────

def _explain(row: pd.Series, total_years: int) -> str:
    parts = []

    if pd.notna(row.get("pin")) and row["pin"] > 0:
        parts.append(f"{row['pin']/1e6:.1f}M people in need")

    if pd.notna(row.get("coverage_ratio")):
        parts.append(f"{row['coverage_ratio']*100:.0f}% funded")
    else:
        parts.append("funding coverage unknown")

    n_cov = int(row.get("n_coverage_years") or 0)
    yu = int(row.get("years_underfunded") or 0)
    if n_cov < 2:
        parts.append("trend & structural data: not enough years on record")
    elif yu > 0:
        label = "every year on record" if yu == total_years else f"{yu}/{total_years} years on record"
        parts.append(f"chronically underfunded — below 30% for {label}")

    slope = row.get("coverage_slope", np.nan)
    if n_cov < 2:
        pass  # already noted above
    elif pd.notna(slope):
        if slope < -0.03:
            parts.append("coverage declining year-over-year")
        elif slope > 0.03:
            parts.append("coverage improving year-over-year")

    pin = row.get("pin", np.nan)
    tgt = row.get("targeted", np.nan)
    if pd.notna(pin) and pd.notna(tgt) and pin > 0:
        pct = tgt / pin * 100
        if pct < 60:
            parts.append(f"only {pct:.0f}% of people in need targeted for response")
    elif pd.notna(pin) and pd.isna(tgt):
        parts.append("targeting data: not reported")

    conf = row.get("crisis_confidence", None)
    if conf == "medium":
        parts.append("data confidence: medium (CBPF/CERF signals only, no HNO)")

    if row.get("robust"):
        parts.append("flagged as robustly overlooked across all ranking methods")

    return "; ".join(parts) if parts else "Insufficient data"
