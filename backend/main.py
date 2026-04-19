"""
Geo-Insight FastAPI backend.
Run from repo root with: uvicorn backend.main:app --reload
Docs at:  http://localhost:8000/docs
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _san(v: Any) -> Any:
    """Recursively replace NaN/inf with None so JSON serialisation never fails."""
    if isinstance(v, float):
        return None if not math.isfinite(v) else v
    if isinstance(v, dict):
        return {k: _san(w) for k, w in v.items()}
    if isinstance(v, list):
        return [_san(w) for w in v]
    return v

app = FastAPI(
    title="Geo-Insight API",
    description="Natural language → SQL → humanitarian crisis rankings via Databricks.",
    version="0.2.0",
)


@app.on_event("startup")
async def _preload_data():
    """Eagerly load data tables on startup so the first request isn't slow."""
    from src.data.loader import get_tables
    _, source = get_tables()
    print(f"  ✓ Data preloaded from: {source}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    year_from: int = 2019
    year_to: int = 2025
    limit: int = 50


class QueryResponse(BaseModel):
    query: str
    sql: str
    columns: List[str]
    rows: List[Dict[str, Any]]
    row_count: int


class ScoreRequest(BaseModel):
    year: int = 2024
    top_n: int = 58
    weights: Optional[Dict[str, float]] = None


class ScoreResponse(BaseModel):
    year: int
    rows: List[Dict[str, Any]]
    row_count: int
    meta: Dict[str, Any]


class AnalyzeRequest(BaseModel):
    prompt: str = ""
    year: int = 2024
    sector: str = "ALL"
    country_iso3: Optional[str] = None
    top_n: int = 20
    previous_scores: Optional[Dict[str, int]] = None
    force_scores: Optional[Dict[str, int]] = None


class AnalyzeResponse(BaseModel):
    prompt: str
    year: int
    sector: str
    weights: Dict[str, float]
    importance_scores: Dict[str, int]
    interpretation: str
    ranked: List[Dict[str, Any]]
    pro_con: Optional[Dict[str, Any]] = None
    dimension_leaders: List[Dict[str, Any]]
    full_rank_map: Dict[str, int]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    from src.data.loader import data_source
    return {"status": "ok", "data_source": data_source()}


@app.post("/refresh")
def refresh_data():
    """Force-reload all tables from Databricks (or parquets if unavailable)."""
    from src.data.loader import get_tables
    _, source = get_tables(refresh=True)
    return {"status": "refreshed", "data_source": source}


@app.post("/query", response_model=QueryResponse)
def run_query(req: QueryRequest):
    """
    Convert a natural-language query to SQL using Databricks Foundation Model APIs,
    execute it against the Databricks warehouse, and return results.
    """
    from src.query.databricks_client import query

    try:
        sql, df = query(req.query, year_filter=(req.year_from, req.year_to))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Query failed: {e}")

    df = df.where(pd.notna(df), None)

    return QueryResponse(
        query=req.query,
        sql=sql,
        columns=list(df.columns),
        rows=df.to_dict(orient="records"),
        row_count=len(df),
    )


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    """
    Weighted gap scoring across four dimensions: scale | gap | structural | trend.
    Returns countries ranked by overlooked_score with explanations and Borda validation.
    """
    from src.analysis.gap_scorer import score_crises, DEFAULT_WEIGHTS

    if req.weights:
        unknown = set(req.weights) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise HTTPException(status_code=422,
                detail=f"Unknown weight keys: {unknown}. Valid: {list(DEFAULT_WEIGHTS)}")

    try:
        df, meta = score_crises(year=req.year, weights=req.weights, top_n=req.top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    keep = [
        "rank", "country_iso3", "country_name", "continent", "region_name",
        "overlooked_score", "gap_component", "confidence_weight",
        "need_scale", "funding_gap", "structural_score", "trend_score",
        "borda_rank", "robust", "explanation", "data_complete",
        "pin", "coverage_ratio", "targeted", "requirements_usd", "fts_funding_usd",
        "years_underfunded", "coverage_slope", "n_coverage_years",
        "pc1", "pc2",
    ]
    cols = [c for c in keep if c in df.columns]
    df_out = df[cols].where(pd.notna(df[cols]), None)

    return ScoreResponse(
        year=req.year,
        rows=df_out.to_dict(orient="records"),
        row_count=len(df_out),
        meta=meta,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    Natural-language priority statement → weight vector → MCDA scoring →
    ranked country list + optional pro/con analysis for a specific country.
    """
    from src.analysis.mcda_analyzer import extract_weights, weights_from_scores, score_all, build_pro_con, DIMENSIONS

    if req.force_scores:
        # Direct re-score — skip LLM entirely
        importance_scores = {k: max(1, min(10, v)) for k, v in req.force_scores.items()}
        weights = weights_from_scores(importance_scores)
        interpretation = "Scores adjusted manually."
    else:
        if not req.prompt.strip():
            raise HTTPException(status_code=422, detail="prompt must not be empty")
        try:
            weights, importance_scores, interpretation = extract_weights(
                req.prompt, previous_scores=req.previous_scores
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Weight extraction failed: {e}")

    compare_year = req.year - 1 if req.year > 2019 else None
    try:
        scored = score_all(
            weights=weights, year=req.year,
            sector=req.sector, compare_year=compare_year,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")

    # Full rank lookup for every country (enables inverse query on frontend)
    full_rank_map = {
        str(r["country_iso3"]): int(r["mcda_rank"])
        for r in scored[["country_iso3", "mcda_rank"]].to_dict(orient="records")
    }

    # Columns to expose in ranked list
    ranked_cols = [
        "mcda_rank", "country_iso3", "country_name", "continent", "region_name",
        "mcda_score", "rank_delta", "has_vulnerability",
        "need_scale", "funding_gap", "structural_neglect", "trend_worsening",
        "targeting_gap", "water_stress", "food_insecurity_risk", "displacement_risk",
        "health_fragility", "climate_vulnerability", "governance_fragility", "disaster_risk",
        "pin", "coverage_ratio", "requirements_usd", "years_underfunded",
    ]
    cols = [c for c in ranked_cols if c in scored.columns]
    top  = scored.head(req.top_n)[cols].where(pd.notna(scored.head(req.top_n)[cols]), None)

    # Pro/con: use requested country; when none specified leave unset (frontend shows leaders)
    target_iso3 = req.country_iso3.upper() if req.country_iso3 else None
    pro_con = None
    if target_iso3:
        try:
            pro_con = build_pro_con(
                country_iso3=target_iso3,
                scored=scored,
                weights=weights,
                user_prompt=req.prompt,
            )
        except Exception as e:
            pro_con = {"error": str(e)}

    # Dimension leaders: top country on each individual metric
    dimension_leaders = []
    for dim, meta in DIMENSIONS.items():
        if dim not in scored.columns:
            continue
        col = scored[dim].dropna()
        if col.empty:
            continue
        leader_row = scored.loc[col.idxmax()]
        v = float(leader_row[dim])
        dimension_leaders.append(_san({
            "dimension":    dim,
            "label":        meta["label"],
            "group":        meta["group"],
            "weight":       round(weights.get(dim, 0), 3),
            "country_iso3": leader_row["country_iso3"],
            "country_name": leader_row["country_name"],
            "value":        round(v, 3) if math.isfinite(v) else None,
            "percentile":   1.0,
            "mcda_rank":    int(leader_row["mcda_rank"]),
        }))

    return AnalyzeResponse(
        prompt=req.prompt,
        year=req.year,
        sector=req.sector,
        weights=weights,
        importance_scores=importance_scores,
        interpretation=interpretation,
        ranked=_san(top.to_dict(orient="records")),
        pro_con=_san(pro_con),
        dimension_leaders=dimension_leaders,
        full_rank_map=full_rank_map,
    )
