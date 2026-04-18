# [SCAFFOLD] Claude-generated — not yet run or tested by the team.
# Run with: uvicorn src.api.main:app --reload
# Docs at:  http://localhost:8000/docs
# Depends on: master.parquet (run build_master_dataset first),
#             ANTHROPIC_API_KEY in .env

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.query.llm_parser import parse_query, QueryFilter
from src.ranking.ranker import run_pipeline

app = FastAPI(
    title="Geo-Insight: Overlooked Crises API",
    description="Ranks humanitarian crises by the gap between documented need and funding coverage.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MASTER_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "master.parquet"
_master_df: pd.DataFrame | None = None


def get_master() -> pd.DataFrame:
    global _master_df
    if _master_df is None:
        if not MASTER_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Master dataset not found. Run build_master_dataset() first.",
            )
        _master_df = pd.read_parquet(MASTER_PATH)
    return _master_df


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_n: int = 20
    include_temporal: bool = True


class CrisisResult(BaseModel):
    rank: int
    country_iso3: Optional[str] = None
    country_name: Optional[str] = None
    year: Optional[int] = None
    pin: Optional[float] = None
    requirements_usd: Optional[float] = None
    funding_usd: Optional[float] = None
    coverage_ratio: Optional[float] = None
    gap_score: float


class RankingResponse(BaseModel):
    query: str
    filters_applied: dict
    results: list[CrisisResult]
    total_in_scope: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rank", response_model=RankingResponse)
def rank_crises(req: QueryRequest):
    """Accept a natural-language query and return ranked overlooked crises."""
    df = get_master()
    filters: QueryFilter = parse_query(req.query)
    ranked = run_pipeline(df, filters, top_n=req.top_n, include_temporal=req.include_temporal)

    results = []
    for _, row in ranked.iterrows():
        results.append(CrisisResult(
            rank=int(row.get("rank", 0)),
            country_iso3=row.get("country_iso3"),
            country_name=row.get("country_name"),
            year=int(row["year"]) if pd.notna(row.get("year")) else None,
            pin=float(row["pin"]) if pd.notna(row.get("pin")) else None,
            requirements_usd=float(row["requirements_usd"]) if pd.notna(row.get("requirements_usd")) else None,
            funding_usd=float(row["funding_usd"]) if pd.notna(row.get("funding_usd")) else None,
            coverage_ratio=float(row["coverage_ratio"]) if pd.notna(row.get("coverage_ratio")) else None,
            gap_score=float(row["gap_score"]),
        ))

    return RankingResponse(
        query=req.query,
        filters_applied=filters.__dict__,
        results=results,
        total_in_scope=len(ranked),
    )


@app.get("/rank/simple", response_model=RankingResponse)
def rank_simple(
    q: str = Query(..., description="Natural language query"),
    top_n: int = Query(20, ge=1, le=100),
):
    """GET version for quick browser/curl testing."""
    return rank_crises(QueryRequest(query=q, top_n=top_n))

# [SCAFFOLD] generate_explanation is commented out in llm_parser.py.
# Re-enable include_explanations once that function is tested:
#
# class QueryRequest(BaseModel):
#     ...
#     include_explanations: bool = False
#
# Inside rank_crises():
#     if req.include_explanations:
#         explanation = generate_explanation(row.to_dict(), req.query)
