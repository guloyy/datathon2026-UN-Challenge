"""
Upload star schema parquet tables to Databricks as Delta tables via direct SQL.
Bypasses DBFS (which is disabled) by creating tables with DDL and batch-inserting rows.

Usage:
    python scripts/upload_to_databricks.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

TABLES = ["dim_country", "dim_sector", "dim_crisis_plan", "fact_crisis_metrics"]

# DDL for each table — explicit types so Delta schema is clean
DDL = {
    "dim_country": """
        CREATE OR REPLACE TABLE {fqn} (
            country_iso3  STRING  NOT NULL,
            country_name  STRING  NOT NULL,
            region        STRING  NOT NULL,
            region_name   STRING  NOT NULL,
            continent     STRING  NOT NULL,
            population    DOUBLE
        ) USING DELTA
    """,
    "dim_sector": """
        CREATE OR REPLACE TABLE {fqn} (
            sector       STRING  NOT NULL,
            sector_name  STRING  NOT NULL,
            description  STRING  NOT NULL
        ) USING DELTA
    """,
    "dim_crisis_plan": """
        CREATE OR REPLACE TABLE {fqn} (
            country_iso3      STRING   NOT NULL,
            year              BIGINT   NOT NULL,
            crisis_name       STRING,
            emergency_group   STRING   NOT NULL,
            emergency_type    STRING   NOT NULL,
            is_crisis         BOOLEAN  NOT NULL,
            crisis_confidence STRING   NOT NULL,
            crisis_signals    STRING   NOT NULL
        ) USING DELTA
    """,
    "fact_crisis_metrics": """
        CREATE OR REPLACE TABLE {fqn} (
            country_iso3     STRING   NOT NULL,
            year             BIGINT   NOT NULL,
            sector           STRING   NOT NULL,
            pin              DOUBLE,
            targeted         DOUBLE,
            requirements_usd DOUBLE,
            fts_funding_usd  DOUBLE,
            cbpf_usd         DOUBLE,
            cerf_usd         DOUBLE,
            coverage_ratio   DOUBLE,
            cbpf_targeted    DOUBLE,
            cbpf_reached     DOUBLE
        ) USING DELTA
    """,
}

BATCH_SIZE = 200


def _to_sql_val(v) -> str:
    """Convert a Python value to a SQL literal."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return str(v)
    return "'" + str(v).replace("'", "''") + "'"


def _insert_df(cur, fqn: str, df: pd.DataFrame, batch_size: int = BATCH_SIZE) -> None:
    cols = ", ".join(df.columns)
    rows = df.values.tolist()
    total = len(rows)
    inserted = 0
    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        values_clause = ", ".join(
            "(" + ", ".join(_to_sql_val(v) for v in row) + ")"
            for row in batch
        )
        cur.execute(f"INSERT INTO {fqn} ({cols}) VALUES {values_clause}")
        inserted += len(batch)
        print(f"    {inserted}/{total} rows inserted", end="\r")
    print()


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    host     = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    token    = os.environ.get("DATABRICKS_TOKEN", "")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH", "")
    catalog  = os.environ.get("DATABRICKS_CATALOG", "hive_metastore")
    schema   = os.environ.get("DATABRICKS_SCHEMA", "geo_insight")

    if not host or not token or not http_path:
        print("ERROR: Set DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH in .env")
        sys.exit(1)

    missing = [t for t in TABLES if not (PROCESSED / f"{t}.parquet").exists()]
    if missing:
        print(f"ERROR: Missing parquet files: {missing}")
        print("Run: python -c \"from src.ingestion.build_star_schema import build_star; build_star()\"")
        sys.exit(1)

    try:
        from databricks import sql as dbsql
    except ImportError:
        print("ERROR: pip install databricks-sql-connector")
        sys.exit(1)

    # Load all dataframes upfront
    dfs = {t: pd.read_parquet(PROCESSED / f"{t}.parquet") for t in TABLES}

    server_hostname = host.replace("https://", "")
    print(f"Connecting to {host} …\n")

    with dbsql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        access_token=token,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
            print(f"Schema {catalog}.{schema} ready.\n")

            for table in TABLES:
                fqn = f"{catalog}.{schema}.{table}"
                df  = dfs[table]

                print(f"Creating {fqn} …")
                cur.execute(DDL[table].format(fqn=fqn))

                print(f"  Inserting {len(df):,} rows …")
                _insert_df(cur, fqn, df)

                cur.execute(f"SELECT COUNT(*) FROM {fqn}")
                count = cur.fetchone()[0]
                print(f"  ✓  {fqn}  →  {count:,} rows\n")

    print("All 4 tables uploaded successfully.")
    print("Next: python scripts/verify_databricks.py")


if __name__ == "__main__":
    main()
