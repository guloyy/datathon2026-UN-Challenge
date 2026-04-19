"""
Unified data loader: Databricks-first with local parquet fallback.

On first call, attempts to pull all four core tables from the live Databricks
warehouse. If successful the DataFrames are cached in memory for the process
lifetime.  If Databricks is unavailable (missing env vars, network error, etc.)
it silently falls back to the local parquet snapshots in data/processed/.

Public API
----------
get_tables(refresh=False) -> (dict[str, DataFrame], source)
get_fact()               -> DataFrame   (fact_crisis_metrics)
get_dim_country()        -> DataFrame   (dim_country)
get_dim_crisis_plan()    -> DataFrame   (dim_crisis_plan)
data_source()            -> "databricks" | "parquet" | "unknown"
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

_TABLES: dict[str, pd.DataFrame] = {}
_SOURCE: str = "unknown"

_DB_TABLES = (
    "fact_crisis_metrics",
    "dim_country",
    "dim_crisis_plan",
    "dim_sector",
)


def _try_databricks() -> dict[str, pd.DataFrame] | None:
    required = ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_HTTP_PATH")
    if any(not os.environ.get(k) for k in required):
        print("  ℹ Data loader: Databricks env vars not set — using parquets")
        return None

    try:
        from databricks import sql as dbsql  # type: ignore[import]

        host = os.environ["DATABRICKS_HOST"].rstrip("/").replace("https://", "")
        conn_kwargs = dict(
            server_hostname=host,
            http_path=os.environ["DATABRICKS_HTTP_PATH"],
            access_token=os.environ["DATABRICKS_TOKEN"],
            catalog=os.environ.get("DATABRICKS_CATALOG", "workspace"),
            schema=os.environ.get("DATABRICKS_SCHEMA", "geo_insight"),
        )

        tables: dict[str, pd.DataFrame] = {}
        with dbsql.connect(**conn_kwargs) as conn:
            for table in _DB_TABLES:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT * FROM workspace.geo_insight.{table}"
                    )
                    rows = cur.fetchall()
                    cols = [d[0] for d in cur.description]
                    tables[table] = pd.DataFrame(rows, columns=cols)

        total = sum(len(v) for v in tables.values())
        print(f"  ✓ Data loader: pulled {total} rows from Databricks")
        return tables

    except Exception as exc:
        print(f"  ⚠ Data loader: Databricks failed ({exc}) — using parquets")
        return None


def _load_parquets() -> dict[str, pd.DataFrame]:
    return {
        t: pd.read_parquet(PROCESSED_DIR / f"{t}.parquet") for t in _DB_TABLES
    }


def get_tables(refresh: bool = False) -> tuple[dict[str, pd.DataFrame], str]:
    """
    Return (tables, source).  Cached after first successful load.
    Pass refresh=True to force a re-fetch from Databricks.
    """
    global _TABLES, _SOURCE

    if _TABLES and not refresh:
        return _TABLES, _SOURCE

    from dotenv import load_dotenv
    load_dotenv()

    db = _try_databricks()
    if db is not None:
        _TABLES, _SOURCE = db, "databricks"
    else:
        _TABLES, _SOURCE = _load_parquets(), "parquet"

    return _TABLES, _SOURCE


def get_fact() -> pd.DataFrame:
    tables, _ = get_tables()
    return tables["fact_crisis_metrics"].copy()


def get_dim_country() -> pd.DataFrame:
    tables, _ = get_tables()
    return tables["dim_country"].copy()


def get_dim_crisis_plan() -> pd.DataFrame:
    tables, _ = get_tables()
    return tables["dim_crisis_plan"].copy()


def data_source() -> str:
    return _SOURCE
