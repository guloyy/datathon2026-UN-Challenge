"""
Verify that all star schema tables exist in Databricks and run a quick sanity query.

Usage:
    python scripts/verify_databricks.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TABLES = ["dim_country", "dim_sector", "dim_crisis_plan", "fact_crisis_metrics"]


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH", "")
    catalog = os.environ.get("DATABRICKS_CATALOG", "hive_metastore")
    schema = os.environ.get("DATABRICKS_SCHEMA", "geo_insight")

    try:
        from databricks import sql as dbsql
    except ImportError:
        print("ERROR: pip install databricks-sql-connector")
        sys.exit(1)

    server_hostname = host.replace("https://", "")
    print(f"Connecting to {host} …\n")

    with dbsql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        access_token=token,
    ) as conn:
        with conn.cursor() as cur:
            # Row counts per table
            for table in TABLES:
                cur.execute(f"SELECT COUNT(*) FROM {catalog}.{schema}.{table}")
                count = cur.fetchone()[0]
                print(f"  {catalog}.{schema}.{table:<30}  {count:>6,} rows")

            print()

            # Sanity: top 5 countries by gap score in 2025
            cur.execute(f"""
                SELECT
                    c.country_name,
                    c.region_name,
                    f.year,
                    f.pin,
                    f.coverage_ratio,
                    ROUND((1 - f.coverage_ratio) * (LN(1 + f.pin) / MAX(LN(1 + f.pin)) OVER ()), 3) AS gap_score
                FROM {catalog}.{schema}.fact_crisis_metrics f
                JOIN {catalog}.{schema}.dim_country c ON f.country_iso3 = c.country_iso3
                WHERE f.sector = 'ALL'
                  AND f.year = 2025
                  AND f.coverage_ratio IS NOT NULL
                  AND f.pin IS NOT NULL
                ORDER BY gap_score DESC
                LIMIT 5
            """)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]

    print("Top 5 most underfunded crises (2025, sector=ALL):")
    print(f"  {'Country':<30} {'Region':<30} {'Year'} {'PIN':>12} {'Coverage':>10} {'Gap':>8}")
    print(f"  {'-'*30} {'-'*30} {'-'*4} {'-'*12} {'-'*10} {'-'*8}")
    for row in rows:
        d = dict(zip(cols, row))
        print(f"  {str(d.get('country_name','')):<30} {str(d.get('region_name','')):<30} "
              f"{d.get('year','')} {d.get('pin', 0):>12,.0f} "
              f"{d.get('coverage_ratio', 0):>10.2%} {d.get('gap_score', 0):>8.3f}")

    print("\nVerification complete.")


if __name__ == "__main__":
    main()
