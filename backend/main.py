"""
Geo-Insight FastAPI backend.
Run from repo root with: uvicorn backend.main:app --reload
Docs at:  http://localhost:8000/docs
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

app = FastAPI(
    title="Geo-Insight API",
    description="Natural language → SQL → humanitarian crisis rankings via Databricks.",
    version="0.2.0",
)

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
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int


class PrioritizeIntentOverride(BaseModel):
    mode: str
    country_iso3: str | None = None
    sector: str | None = None
    region: str | None = None
    emergency_group: str | None = None
    year: int | None = None
    hints: list[str] = []


class PrioritizeRequest(BaseModel):
    query: str
    k: int = 3
    use_llm_prose: bool = True
    intent_override: PrioritizeIntentOverride | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


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


@app.post("/prioritize")
def run_prioritize(req: PrioritizeRequest):
    """
    Turn an HC-style intent ("prioritize Brazil for water supply") into a
    defensible weight vector that places the target in the top-k, or explain
    why it cannot.

    If `intent_override` is supplied, the NL parser is bypassed (useful when
    ANTHROPIC_API_KEY is not configured).
    """
    from src.prioritize.intent import Intent
    from src.prioritize.pipeline import prioritize
    from src.prioritize.weights import InfeasibleError

    intent = None
    if req.intent_override is not None:
        intent = Intent(
            mode=req.intent_override.mode,
            country_iso3=req.intent_override.country_iso3,
            sector=req.intent_override.sector,
            region=req.intent_override.region,
            emergency_group=req.intent_override.emergency_group,
            year=req.intent_override.year,
            hints=req.intent_override.hints,
            raw_query=req.query,
        )

    try:
        response = prioritize(
            req.query,
            k=req.k,
            intent=intent,
            use_llm_prose=req.use_llm_prose,
        )
    except InfeasibleError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "infeasible",
                "reason": e.reason,
                "blocking_rows": e.blocking_rows,
            },
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prioritize failed: {e}")

    return response
