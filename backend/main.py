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
    year_from: int = 2024
    year_to: int = 2026
    limit: int = 50


class QueryResponse(BaseModel):
    query: str
    sql: str
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int


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
    from src.query.sql_generator import generate_sql
    from src.query.databricks_client import run_sql

    try:
        sql = generate_sql(
            req.query,
            year_filter=(req.year_from, req.year_to),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"SQL generation failed: {e}")

    try:
        df = run_sql(sql)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Databricks query failed: {e}")

    # Convert NaN to None so JSON serialises cleanly
    df = df.where(pd.notna(df), None)

    return QueryResponse(
        query=req.query,
        sql=sql,
        columns=list(df.columns),
        rows=df.to_dict(orient="records"),
        row_count=len(df),
    )
