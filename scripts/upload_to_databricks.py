"""
Upload master.parquet to Databricks and register it as a Delta table.

Usage:
    python scripts/upload_to_databricks.py

Prerequisites:
    1. Fill in DATABRICKS_HOST, DATABRICKS_TOKEN in your .env
    2. Start your cluster in the Databricks Community Edition UI
    3. Copy the cluster's JDBC/ODBC HTTP Path into DATABRICKS_HTTP_PATH in .env
       (Cluster → Advanced Options → JDBC/ODBC tab)

What this script does:
    1. Reads data/processed/master.parquet
    2. Uploads it to DBFS at dbfs:/geo_insight/master.parquet
    3. Creates schema geo_insight in hive_metastore (if it doesn't exist)
    4. Registers a Delta table geo_insight.master pointing at that file
    5. Prints the table schema so you can verify column names
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "processed" / "master.parquet"
DBFS_DIR = "dbfs:/geo_insight"
DBFS_PATH = f"{DBFS_DIR}/master.parquet"


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH", "")
    catalog = os.environ.get("DATABRICKS_CATALOG", "hive_metastore")
    schema = os.environ.get("DATABRICKS_SCHEMA", "geo_insight")

    if not host or not token:
        print("ERROR: Set DATABRICKS_HOST and DATABRICKS_TOKEN in your .env file.")
        sys.exit(1)

    if not http_path:
        print(
            "ERROR: DATABRICKS_HTTP_PATH not set.\n"
            "Find it in: Databricks UI → Clusters → your cluster → "
            "Advanced Options → JDBC/ODBC → HTTP Path\n"
            "It looks like:  /sql/1.0/warehouses/abc123  or  /sql/protocolv1/o/0/abc123"
        )
        sys.exit(1)

    if not PARQUET.exists():
        print(f"ERROR: {PARQUET} not found. Run build_master_dataset() first.")
        sys.exit(1)

    # ── 1. Upload parquet to DBFS via Databricks SDK ──────────────────────────
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError:
        print("ERROR: databricks-sdk not installed. Run: pip install databricks-sdk")
        sys.exit(1)

    print(f"Connecting to {host} …")
    ws = WorkspaceClient(host=host, token=token)

    print(f"Uploading {PARQUET.name} ({PARQUET.stat().st_size / 1024:.0f} KB) → {DBFS_PATH}")
    with open(PARQUET, "rb") as f:
        ws.dbfs.upload(DBFS_PATH, f.read(), overwrite=True)
    print("  Upload complete.")

    # ── 2. Create Delta table via SQL connector ────────────────────────────────
    try:
        from databricks import sql as dbsql
    except ImportError:
        print("ERROR: databricks-sql-connector not installed. Run: pip install databricks-sql-connector")
        sys.exit(1)

    server_hostname = host.replace("https://", "")
    print(f"\nConnecting via SQL connector …")

    with dbsql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        access_token=token,
    ) as conn:
        with conn.cursor() as cur:
            # Create schema
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
            print(f"  Schema {catalog}.{schema} ready.")

            # Create or replace Delta table from the uploaded parquet
            cur.execute(f"""
                CREATE OR REPLACE TABLE {catalog}.{schema}.master
                USING DELTA
                AS SELECT * FROM parquet.`{DBFS_PATH}`
            """)
            print(f"  Table {catalog}.{schema}.master created.")

            # Print schema for verification
            cur.execute(f"DESCRIBE TABLE {catalog}.{schema}.master")
            rows = cur.fetchall()

    print(f"\nTable schema ({catalog}.{schema}.master):")
    print(f"  {'Column':<30} {'Type':<20}")
    print(f"  {'-'*30} {'-'*20}")
    for row in rows:
        print(f"  {str(row[0]):<30} {str(row[1]):<20}")

    print(f"\nDone. Update DATABRICKS_CATALOG and DATABRICKS_SCHEMA in .env if needed.")
    print(f"Next: from src.query.databricks_client import query")
    print(f"      df = query('which countries have the highest gap score in 2025?')")


if __name__ == "__main__":
    main()
