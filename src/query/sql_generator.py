"""
Natural-language → SQL generator using Databricks Foundation Model APIs.

Uses the Databricks AI toolkit (OpenAI-compatible endpoint) so no Anthropic key
is needed — only your DATABRICKS_HOST and DATABRICKS_TOKEN.

Available models (free tier):
    databricks-meta-llama-3-3-70b-instruct  ← default, best for SQL
    databricks-dbrx-instruct
    databricks-mixtral-8x7b-instruct

Usage:
    from src.query.sql_generator import generate_sql

    sql = generate_sql("which African countries have less than 20% funding coverage in 2025?")
    print(sql)

The schema is read dynamically from data/processed/master.parquet so it stays
in sync automatically when the unified dataset is updated.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

DEFAULT_MODEL = "databricks-meta-llama-3-3-70b-instruct"


def _get_client():
    """Return an OpenAI client pointed at the Databricks Foundation Model endpoint."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package not installed.\n"
            "Run: pip install openai  (or add it to environment.yml)"
        )
    from dotenv import load_dotenv
    load_dotenv()

    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    if not host or not token:
        raise EnvironmentError(
            "DATABRICKS_HOST and DATABRICKS_TOKEN must be set in your .env file."
        )

    return OpenAI(
        api_key=token,
        base_url=f"{host}/serving-endpoints",
    )


def get_schema(parquet_path: Path | None = None) -> dict[str, str]:
    """
    Read column names and dtypes from the master parquet file.
    Returns a dict of {column_name: dtype_string}.
    """
    path = parquet_path or (PROCESSED_DIR / "master.parquet")
    df = pd.read_parquet(path, engine="pyarrow")
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def _schema_description(schema: dict[str, str]) -> str:
    """Format schema dict as a readable column list for the prompt."""
    lines = []
    for col, dtype in schema.items():
        if "int" in dtype:
            sql_type = "INTEGER"
        elif "float" in dtype:
            sql_type = "DOUBLE"
        elif "object" in dtype or "string" in dtype:
            sql_type = "STRING"
        elif "bool" in dtype:
            sql_type = "BOOLEAN"
        else:
            sql_type = dtype.upper()
        lines.append(f"  {col}  ({sql_type})")
    return "\n".join(lines)


def generate_sql(
    user_query: str,
    table_name: str = "geo_insight.master",
    parquet_path: Path | None = None,
    year_filter: tuple[int, int] = (2024, 2026),
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Convert a natural-language query into a Databricks SQL SELECT statement
    using the Databricks Foundation Model APIs.

    Args:
        user_query:   Free-text question from the user.
        table_name:   Fully-qualified Databricks table (schema.table or catalog.schema.table).
        parquet_path: Override path to the parquet file for schema detection.
        year_filter:  Default year range applied when the user doesn't specify one.
        model:        Databricks Foundation Model endpoint name.

    Returns:
        A SQL string ready to execute against Databricks.
    """
    schema = get_schema(parquet_path)
    schema_block = _schema_description(schema)

    system_prompt = f"""You are a SQL expert for humanitarian data analysis.

The dataset is a Databricks Delta table named `{table_name}` with these columns:

{schema_block}

Key domain facts:
- `pin` = people in need (from HNO). Null for rows with no HNO counterpart.
- `requirements_usd` = total funding requested (from FTS). Null where no formal plan exists.
- `funding_usd` = total funding received (from FTS).
- `coverage_ratio` = funding_usd / requirements_usd, clamped to [0, 1]. Null where requirements_usd is null.
- `sector` = humanitarian cluster code: ALL (overall caseload), FSC (food), HEA (health),
  EDU (education), NUT (nutrition), PRO (protection), SHL (shelter), WSH (water & sanitation).
- `country_iso3` = ISO 3-letter country code (e.g. AFG, YEM, SSD).
- HNO data covers years 2024–2026. FTS spans 1999–2031.
  Always add WHERE year BETWEEN {year_filter[0]} AND {year_filter[1]} unless the user requests otherwise.

Rules:
- Return ONLY the SQL query — no explanation, no markdown fences, no commentary.
- Use Databricks SQL (Spark SQL dialect).
- Always filter out rows where both pin IS NULL AND requirements_usd IS NULL, unless the user explicitly asks for all rows.
- For "most overlooked" / "worst gap" queries use:
  gap_score = (1 - coverage_ratio) * (LN(1 + pin) / MAX(LN(1 + pin)) OVER ())
- Default LIMIT 50 unless the user specifies otherwise."""

    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        max_tokens=1024,
        temperature=0,
    )
    sql = response.choices[0].message.content.strip()

    # Strip markdown fences if the model adds them anyway
    if sql.startswith("```"):
        sql = "\n".join(
            line for line in sql.splitlines()
            if not line.startswith("```")
        ).strip()

    return sql
