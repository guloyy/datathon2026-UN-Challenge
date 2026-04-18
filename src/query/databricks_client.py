"""
Databricks SQL execution client with automatic self-healing retry.

If the generated SQL fails, the error is fed back to the LLM with the original
question and failed SQL so it can correct the query. Retries up to MAX_RETRIES times.
"""

from __future__ import annotations

import os

import pandas as pd

_REQUIRED_ENV = ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_HTTP_PATH")
MAX_RETRIES = 3  # number of fix attempts after the initial try (4 total)


def _check_env() -> None:
    missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required env vars: {missing}\n"
            "Copy .env.example → .env and fill in your Databricks credentials."
        )


def _get_connection():
    try:
        from databricks import sql as dbsql
    except ImportError:
        raise ImportError("Run: pip install databricks-sql-connector")

    host = os.environ["DATABRICKS_HOST"].rstrip("/").replace("https://", "")
    return dbsql.connect(
        server_hostname=host,
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
        access_token=os.environ["DATABRICKS_TOKEN"],
        catalog=os.environ.get("DATABRICKS_CATALOG", "workspace"),
        schema=os.environ.get("DATABRICKS_SCHEMA", "geo_insight"),
    )


def run_sql(sql: str) -> pd.DataFrame:
    """Execute SQL and return results as a DataFrame. Raises on error."""
    from dotenv import load_dotenv
    load_dotenv()
    _check_env()

    with _get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

    return pd.DataFrame(rows, columns=columns)


def query(user_prompt: str, year_filter: tuple[int, int] = (2019, 2025)) -> tuple[str, pd.DataFrame]:
    """
    End-to-end: natural language → SQL → DataFrame with self-healing retry.

    Flow:
        1. generate_sql() asks Llama 4 Maverick (falls back to Llama 3.3 70B if unavailable)
        2. run_sql() executes on Databricks
        3. On SQL error: fix_sql() sends the error back to the LLM for correction
        4. Retries up to MAX_RETRIES times before raising RuntimeError

    Returns:
        (final_sql, DataFrame) tuple.
    """
    from dotenv import load_dotenv
    from src.query.sql_generator import generate_sql, fix_sql

    load_dotenv()
    _check_env()

    sql = generate_sql(user_prompt, year_filter=year_filter)
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 2):  # 1 = first try, 2..MAX_RETRIES+1 = retries
        try:
            df = run_sql(sql)
            if attempt > 1:
                print(f"  ✓ Succeeded on attempt {attempt}/{MAX_RETRIES + 1}")
            return sql, df

        except Exception as exc:
            last_error = exc
            if attempt > MAX_RETRIES:
                break

            error_msg = str(exc)
            print(f"  ✗ Attempt {attempt}/{MAX_RETRIES + 1} failed: {error_msg[:120]}")
            print(f"  ↻ Sending error to LLM for correction…")
            sql = fix_sql(user_prompt, sql, error_msg, year_filter=year_filter)

    raise RuntimeError(
        f"Query failed after {MAX_RETRIES + 1} attempts.\nLast error: {last_error}"
    ) from last_error
