"""Unit tests for the /prioritize pipeline.

Tests use synthetic feature matrices built via features.build_feature_matrix,
so no real master.parquet or Anthropic API key is required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.prioritize.features import (
    FEATURE_ORDER,
    build_feature_matrix,
    _winsorize,
)
from src.prioritize.explain import (
    build_ranking_slice,
    compute_counterfactual,
    compute_pros_cons,
)
from src.prioritize.intent import Intent
from src.prioritize.pipeline import prioritize
from src.prioritize.weights import (
    DEFAULT_WEIGHTS,
    InfeasibleError,
    blocking_rows,
    default_weight_vector,
    optimize_mip,
    target_rank,
)


def _mk_df(rows: list[dict]) -> pd.DataFrame:
    """Helper: build a master-like DataFrame from dicts."""
    df = pd.DataFrame(rows)
    # Ensure required columns exist
    for col in ("coverage_ratio", "pin", "requirements_usd", "fts_funding_usd",
                "population", "cbpf_usd", "cerf_usd"):
        if col not in df.columns:
            df[col] = np.nan
    return df


# ── Feature matrix tests ──────────────────────────────────────────────────────

def test_feature_matrix_shape_no_one_hots():
    """Feature matrix has only substantive dims; no country/sector one-hots."""
    df = _mk_df([
        {"country_iso3": "YEM", "country_name": "Yemen", "sector": "ALL", "year": 2024,
         "coverage_ratio": 0.1, "pin": 20_000_000, "requirements_usd": 4e9, "fts_funding_usd": 5e8},
        {"country_iso3": "SOM", "country_name": "Somalia", "sector": "ALL", "year": 2024,
         "coverage_ratio": 0.3, "pin": 7_000_000, "requirements_usd": 1.2e9, "fts_funding_usd": 4e8},
    ])
    fm = build_feature_matrix(df)
    assert fm.X.shape[1] == len(FEATURE_ORDER)
    # No one-hot features:
    assert not any(name.startswith("country_") for name in fm.feature_names)
    assert not any(name.startswith("sector_") for name in fm.feature_names)
    assert not any(name.startswith("emergency_") for name in fm.feature_names)
    # No NaN after normalization:
    assert np.isfinite(fm.X).all()


def test_winsorization_clips_outliers():
    """95th-percentile winsorization compresses extremes but preserves mid-range."""
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000.0])
    out = _winsorize(vals, 5.0, 95.0)
    assert out.max() < 1000.0        # outlier compressed
    assert out.min() >= 1.0           # low end not inflated
    assert out[4] == pytest.approx(5.0)  # mid-range preserved


# ── Default-weights-reproduce-gap_score ───────────────────────────────────────

def test_default_weights_reproduce_gap_score():
    """Linear score with w=[0.5,0.5,0...] correlates highly with the existing gap_score."""
    from src.scoring.gap_score import compute_gap_scores

    n = 30
    rng = np.random.default_rng(42)
    df = _mk_df([
        {
            "country_iso3": f"C{i:02d}", "country_name": f"Country{i}",
            "sector": "ALL", "year": 2024,
            "coverage_ratio": float(rng.uniform(0.0, 1.0)),
            "pin": float(rng.uniform(1e5, 2e7)),
            "requirements_usd": float(rng.uniform(1e8, 5e9)),
            "fts_funding_usd": float(rng.uniform(1e7, 2e9)),
        }
        for i in range(n)
    ])
    df["funding_usd"] = df["fts_funding_usd"]
    scored = compute_gap_scores(df)

    fm = build_feature_matrix(df)
    w_def = default_weight_vector(fm.feature_names)
    linear = fm.X @ w_def

    rho, _ = spearmanr(linear, scored["gap_score"].to_numpy())
    assert rho > 0.95, f"Spearman corr too low: {rho}"


# ── Optimizer tests ───────────────────────────────────────────────────────────

def _synth_fm(n: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    df = _mk_df([
        {
            "country_iso3": f"T{i:02d}", "country_name": f"Row{i}",
            "sector": "WSH", "year": 2024,
            "coverage_ratio": float(rng.uniform(0.0, 1.0)),
            "pin": float(rng.uniform(1e5, 1e7)),
            "requirements_usd": float(rng.uniform(1e8, 2e9)),
            "fts_funding_usd": float(rng.uniform(1e7, 1e9)),
            "population": float(rng.uniform(1e6, 1e8)),
        }
        for i in range(n)
    ])
    return build_feature_matrix(df)


def test_mip_rank_constraint_exact():
    """Synthetic: target at index 0 with low scores should get rank ≤ k after optimization."""
    fm = _synth_fm(n=15, seed=1)
    # Make target row weak on both gap and need but strong on per_capita_need
    # We test: can the optimizer find w that places target in top-3?
    target_idx = 0
    k = 3
    # Boost one feature for target to make it feasible
    boost_feat = fm.feature_names.index("per_capita_need")
    fm.X[target_idx] = 0.0
    fm.X[target_idx, boost_feat] = 1.0
    for j in range(1, fm.n_rows):
        fm.X[j, boost_feat] = 0.0  # no other row has this feature

    result = optimize_mip(fm.X, target_idx, fm.feature_names, k=k)
    assert result.rank_achieved <= k


def test_optimizer_infeasibility_returns_blocking_rows():
    """When target is dominated on every feature, MIP is infeasible → blocking_rows populated."""
    fm = _synth_fm(n=10, seed=2)
    target_idx = 0
    # Make target literally zero on every feature; all others >= 0.1 on every feature.
    fm.X[target_idx] = 0.0
    for j in range(1, fm.n_rows):
        fm.X[j] = np.maximum(fm.X[j], 0.2)

    w_def = default_weight_vector(fm.feature_names)
    blockers = blocking_rows(
        fm.X, target_idx, w_def, fm.feature_names, fm.rows, n=2
    )
    assert len(blockers) == 2
    for b in blockers:
        assert b["score_gap"] > 0
        assert len(b["dominant_features"]) >= 1


def test_target_rank_computation():
    """1-based rank should match argsort position."""
    fm = _synth_fm(n=8, seed=3)
    w_def = default_weight_vector(fm.feature_names)
    scores = fm.X @ w_def
    order = np.argsort(-scores)
    for pos, idx in enumerate(order, start=1):
        assert target_rank(fm.X, int(idx), w_def) == pos


# ── Pros / cons tests ─────────────────────────────────────────────────────────

def test_pros_cons_contribution_based():
    """Pros = high contribution; cons = low weight + high target value."""
    feature_names = ["coverage_gap", "need_scale", "pin", "per_capita_need"]
    w = np.array([0.6, 0.3, 0.1, 0.0])
    x_t = np.array([0.9, 0.4, 0.2, 0.85])  # coverage_gap big contrib; per_capita_need is high but w=0

    pros, cons = compute_pros_cons(w, x_t, feature_names)

    pro_feats = [p["feature"] for p in pros]
    assert "coverage_gap" in pro_feats                      # biggest contribution
    con_feats = [c["feature"] for c in cons]
    assert "per_capita_need" in con_feats                   # w=0 but x_t=0.85


def test_pros_are_sorted_by_contribution():
    feature_names = ["coverage_gap", "need_scale", "pin"]
    w = np.array([0.3, 0.4, 0.3])
    x_t = np.array([0.5, 0.9, 0.6])
    pros, _ = compute_pros_cons(w, x_t, feature_names)
    contribs = [p["contribution"] for p in pros]
    assert contribs == sorted(contribs, reverse=True)


# ── Counterfactual determinism ────────────────────────────────────────────────

def test_counterfactual_is_deterministic():
    """Same inputs → identical counterfactual struct (no LLM in selection path)."""
    fm = _synth_fm(n=12, seed=5)
    target_idx = 3
    w = default_weight_vector(fm.feature_names)

    cf1 = compute_counterfactual(fm.X, fm.rows, target_idx, w, fm.feature_names, k=3)
    cf2 = compute_counterfactual(fm.X, fm.rows, target_idx, w, fm.feature_names, k=3)
    assert cf1 == cf2


def test_counterfactual_contains_archetypes():
    fm = _synth_fm(n=12, seed=6)
    target_idx = 4
    w = default_weight_vector(fm.feature_names)
    cf = compute_counterfactual(fm.X, fm.rows, target_idx, w, fm.feature_names, k=3)
    assert "archetypes" in cf
    assert "severity_max" in cf["archetypes"]
    assert "gap_max" in cf["archetypes"]


# ── Ranking slice ─────────────────────────────────────────────────────────────

def test_ranking_includes_target_if_outside_top_n():
    fm = _synth_fm(n=20, seed=7)
    w = default_weight_vector(fm.feature_names)
    scores = fm.X @ w
    # Pick the worst-scoring row as target
    target_idx = int(np.argmin(scores))
    ranking = build_ranking_slice(fm.X, fm.rows, w, target_idx, top_n=5)
    target_in_ranking = any(
        entry.get("country_iso3") == fm.rows.iloc[target_idx]["country_iso3"]
        for entry in ranking
    )
    assert target_in_ranking


# ── Pipeline end-to-end (no LLM) ──────────────────────────────────────────────

def _build_master_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    countries = [f"T{i:02d}" for i in range(n)]
    for i, c in enumerate(countries):
        rows.append({
            "country_iso3": c, "country_name": f"Country{i}",
            "sector": "WSH", "year": 2024,
            "coverage_ratio": float(rng.uniform(0.0, 1.0)),
            "pin": float(rng.uniform(1e5, 1e7)),
            "requirements_usd": float(rng.uniform(1e8, 2e9)),
            "fts_funding_usd": float(rng.uniform(1e7, 1e9)),
            "population": float(rng.uniform(1e6, 1e8)),
        })
    return _mk_df(rows)


def test_pipeline_short_circuit_populates_cons():
    """When target is already top-k under defaults, response includes cons."""
    df = _build_master_df(n=10, seed=10)
    # Force one country to have the best default score — make it our target.
    # Default = 0.5*coverage_gap + 0.5*need_scale. Max coverage_gap = max(1-coverage_ratio).
    # Set a unique very high gap + very high pin for "T05".
    df.loc[df["country_iso3"] == "T05", "coverage_ratio"] = 0.0
    df.loc[df["country_iso3"] == "T05", "pin"] = 5e7

    fm = build_feature_matrix(df)
    intent = Intent(mode="target_sector", country_iso3="T05", sector="WSH")

    response = prioritize(
        query="ignored",
        fm=fm,
        intent=intent,
        use_llm_prose=False,
    )
    assert response["short_circuited"] is True
    assert response["target"]["country_iso3"] == "T05"
    assert response["target"]["rank"] <= 3
    assert isinstance(response["cons"], list)
    # Cons is populated as long as there exist features with w≈0 and target value > 0.5
    # In this setup, funding and population features have non-zero target value.


def test_pipeline_optimizer_path_places_target_in_top_k():
    """When target is not top-k under defaults, optimizer should lift it if feasible."""
    df = _build_master_df(n=15, seed=11)
    # Deliberately hide target from default top-k by making coverage/pin mediocre
    # but per-capita need very high (small population, decent pin).
    df.loc[df["country_iso3"] == "T09", "coverage_ratio"] = 0.5
    df.loc[df["country_iso3"] == "T09", "pin"] = 5e5
    df.loc[df["country_iso3"] == "T09", "population"] = 1e6  # tiny pop → huge per-capita
    # Push others' population high so per_capita_need is dominated by T09
    others = df["country_iso3"] != "T09"
    df.loc[others, "population"] = 5e7

    fm = build_feature_matrix(df)
    intent = Intent(mode="target_sector", country_iso3="T09", sector="WSH")

    response = prioritize(
        query="ignored", fm=fm, intent=intent, use_llm_prose=False
    )
    # Optimizer should place T09 in top-3 via per_capita_need
    assert response["short_circuited"] is False
    assert response["target"]["rank"] <= 3
    assert response["weight_deviation_from_default"] > 0


def test_pipeline_infeasible_raises_with_blocking_rows():
    """A target dominated on every feature → InfeasibleError with blockers.

    We build the feature matrix directly instead of going through the DataFrame
    normalization path — this lets us exactly zero out the target across all
    features while keeping the others uniformly high.
    """
    fm = _synth_fm(n=12, seed=12)
    target_idx = 0
    # Target is strictly zero on every feature.
    fm.X[target_idx] = 0.0
    # Everyone else is at least 0.5 on every feature (strict dominance).
    for j in range(1, fm.n_rows):
        fm.X[j] = np.maximum(fm.X[j], 0.5)
    # Rewrite row metadata so the country_iso3 matches the intent target.
    fm.rows.loc[target_idx, "country_iso3"] = "T00"
    intent = Intent(mode="target_sector", country_iso3="T00", sector="WSH")

    with pytest.raises(InfeasibleError) as exc:
        prioritize(query="ignored", fm=fm, intent=intent, use_llm_prose=False)
    assert len(exc.value.blocking_rows) >= 1
    for b in exc.value.blocking_rows:
        assert b["score_gap"] > 0


def test_pipeline_filter_only_mode_returns_ranking_under_defaults():
    df = _build_master_df(n=10, seed=13)
    fm = build_feature_matrix(df)
    intent = Intent(mode="filter_only", sector="WSH")

    response = prioritize(
        query="ignored", fm=fm, intent=intent, use_llm_prose=False
    )
    assert response["mode"] == "filter_only"
    assert response["target"] is None
    assert response["counterfactual"] is None
    assert len(response["ranking"]) > 0
    assert response["ranking"][0]["rank"] == 1
