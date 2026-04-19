"""
Natural-language → SQL generator using Databricks Foundation Model APIs.

Uses the Databricks AI toolkit (OpenAI-compatible endpoint) so no Anthropic key
is needed — only your DATABRICKS_HOST and DATABRICKS_TOKEN.

Schema: star schema with 4 Delta tables in geo_insight schema:
    dim_country, dim_sector, dim_crisis_plan, fact_crisis_metrics

Usage:
    from src.query.sql_generator import generate_sql
    sql = generate_sql("which African countries have less than 20% funding coverage in 2025?")
"""

from __future__ import annotations

import os
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

PRIMARY_MODEL = "databricks-llama-4-maverick"
FALLBACK_MODEL = "databricks-meta-llama-3-3-70b-instruct"

STAR_SCHEMA_DESCRIPTION = """
Star schema — 4 Delta tables in the `geo_insight` schema. Years covered: 2019–2025.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TABLE: workspace.geo_insight.fact_crisis_metrics
Grain: one row per country × year × sector  (1,712 rows total)
Primary key: (country_iso3, year, sector)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  country_iso3     STRING   NOT NULL  FK → dim_country.country_iso3
  year             BIGINT   NOT NULL  range: 2019–2025
  sector           STRING   NOT NULL  FK → dim_sector.sector
  pin              DOUBLE   NULL(75%) People In Need from OCHA HNO assessments; NULL for countries not covered by HNO
  targeted         DOUBLE   NULL(75%) people targeted for humanitarian response (from HNO)
  requirements_usd DOUBLE   NULL(14%) funding requested in USD (from FTS); NULL when no formal plan exists
  fts_funding_usd  DOUBLE   NULL(13%) funding received in USD (from FTS)
  cbpf_usd         DOUBLE   NULL(67%) Country-Based Pooled Fund allocation in USD
  cerf_usd         DOUBLE   NULL(56%) Central Emergency Response Fund allocation in USD
  coverage_ratio   DOUBLE   NULL(14%) fts_funding_usd / requirements_usd clamped to [0,1]; NULL iff requirements_usd IS NULL
  cbpf_targeted    DOUBLE   NULL(67%) people targeted by CBPF-funded projects
  cbpf_reached     DOUBLE   NULL(67%) people reached by CBPF-funded projects

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TABLE: workspace.geo_insight.dim_country
Grain: one row per crisis country  (50 rows total)
Primary key: country_iso3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  country_iso3  STRING   NOT NULL  ISO 3-letter code e.g. YEM, SDN, COD, AFG
  country_name  STRING   NOT NULL  full English name e.g. "Yemen", "Sudan"
  region        STRING   NOT NULL  one of: SSA | MENA | APAC | LAC | EECA
  region_name   STRING   NOT NULL  one of: "Sub-Saharan Africa" | "Middle East & North Africa" | "Asia-Pacific" | "Latin America & Caribbean" | "Eastern Europe & Central Asia"
  continent     STRING   NOT NULL  one of: "Africa" | "Middle East" | "Asia" | "Latin America" | "Europe & Central Asia"
                                   Use this for broad geographic filters: "Africa" covers both Sub-Saharan and North African countries (EGY, LBY, TUN)
  population    DOUBLE   NULL(18%) total population from UN COD; NULL for CAF COG JOR LBN LBY MMR SYR UKR YEM

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TABLE: workspace.geo_insight.dim_sector
Grain: one row per humanitarian cluster  (15 rows total)
Primary key: sector
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  sector       STRING  NOT NULL  cluster code — one of: ALL FSC HEA WSH SHL PRO EDU NUT MPC LOG ETC CCCM ERY COOR OTHER
  sector_name  STRING  NOT NULL  human-readable name e.g. "Food Security", "Health", "WASH"
  description  STRING  NOT NULL  brief description of what the cluster covers

  Key sector values:
    ALL  = overall plan caseload (use this for country-level totals, not sector breakdowns)
    FSC  = Food Security    HEA = Health          WSH  = WASH
    SHL  = Shelter          PRO = Protection       EDU  = Education
    NUT  = Nutrition        MPC = Multipurpose Cash

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TABLE: workspace.geo_insight.dim_crisis_plan
Grain: one row per country × year  (239 rows total)
Primary key: (country_iso3, year)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  country_iso3      STRING   NOT NULL  FK → dim_country.country_iso3
  year              BIGINT   NOT NULL  range: 2019–2025
  crisis_name       STRING   NULL(0%)  official English plan name e.g. "Yemen Humanitarian Response Plan"
  emergency_group   STRING   NOT NULL  exactly one of: "Conflict-related" | "Natural Disaster" | "Other"
  emergency_type    STRING   NOT NULL  specific sub-type e.g. "Natural Disaster - Flood", "Disease Outbreak - Ebola", "Conflict-related - Displacement"
  is_crisis         BOOLEAN  NOT NULL  always TRUE — non-crisis rows excluded at build time
  crisis_confidence STRING   NOT NULL  "high" = confirmed by HNO or HRP plan | "medium" = CBPF/CERF only
  crisis_signals    STRING   NOT NULL  comma-separated signals that triggered classification; any subset of: HNO, HRP, CBPF, CERF

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HUMANITARIAN TERMINOLOGY — column aliases and common phrasings
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When the user uses any of these terms, map them to the correct column:

  "people in need" / "PIN" / "people who are in need" / "those in need"
  / "affected population" / "people needing assistance"
      → f.pin

  "people targeted" / "targeted population" / "response target"
  / "people planned to be reached"
      → f.targeted

  "funding requirements" / "funds requested" / "appeal amount"
  / "humanitarian ask" / "budget needed"
      → f.requirements_usd

  "funding received" / "funds raised" / "donations" / "actual funding"
  / "money received" / "contributions"
      → f.fts_funding_usd

  "funding gap" / "coverage" / "funding coverage" / "coverage ratio"
  / "percent funded" / "how funded" / "funding rate"
      → f.coverage_ratio  (value between 0 and 1; multiply by 100 for percentage)

  "CBPF" / "country-based pooled fund" / "pooled fund"
      → f.cbpf_usd

  "CERF" / "central emergency response fund"
      → f.cerf_usd

  "overlooked" / "neglected" / "forgotten" / "underfunded" / "worst gap"
      → use the gap_score formula: (1 - f.coverage_ratio) * (LN(1 + f.pin) / MAX(LN(1 + f.pin)) OVER ())
        (requires both coverage_ratio IS NOT NULL and pin IS NOT NULL)

  "crisis type" / "type of emergency" / "conflict" / "natural disaster"
      → p.emergency_group or p.emergency_type (join dim_crisis_plan)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUERY RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Always alias: fact_crisis_metrics AS f, dim_country AS c, dim_sector AS s, dim_crisis_plan AS p
- Always include f.country_iso3 and c.country_name in SELECT so the map and table can render correctly
- Qualify every column with its table alias to avoid ambiguity
- sector='ALL' rows contain pre-aggregated totals across all humanitarian clusters for a country×year. Use f.sector = 'ALL' ONLY when the question asks about overall/aggregate figures (e.g. "total need", "overall coverage", "all sectors combined"). If the question names or implies a specific sector (food, health, WASH, protection, etc.), use that sector code instead (FSC, HEA, WSH, PRO…). You can have country-level results with any sector — sector and geographic grain are independent. NEVER mix sector='ALL' rows with other sector rows in the same query.
- coverage_ratio is NULL when requirements_usd is NULL — guard arithmetic with IS NOT NULL
- "Most overlooked / worst gap" formula (only where both fields are non-null):
    gap_score = (1 - f.coverage_ratio) * (LN(1 + f.pin) / MAX(LN(1 + f.pin)) OVER ())
- Typical join pattern:
    FROM workspace.geo_insight.fact_crisis_metrics f
    JOIN workspace.geo_insight.dim_country c ON f.country_iso3 = c.country_iso3
    JOIN workspace.geo_insight.dim_crisis_plan p ON f.country_iso3 = p.country_iso3 AND f.year = p.year
- Default LIMIT 50 unless user specifies otherwise
- ALWAYS include ORDER BY before LIMIT. Choose the most meaningful sort: if the question implies severity/priority, order by coverage_ratio ASC (worst funded first) or pin DESC (most people in need first). Never use LIMIT without ORDER BY.
- When the user asks about a problem (water, food, health, etc.) filter to that sector AND order by the most relevant metric so LIMIT returns the most severe cases, not arbitrary rows.
"""


def _get_client():
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")
    from dotenv import load_dotenv
    load_dotenv()

    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    if not host or not token:
        raise EnvironmentError("DATABRICKS_HOST and DATABRICKS_TOKEN must be set in .env")

    return OpenAI(api_key=token, base_url=f"{host}/serving-endpoints")


def _call_llm(messages: list[dict], model: str = PRIMARY_MODEL, max_tokens: int = 1024) -> str:
    """Call the LLM; fall back to FALLBACK_MODEL if the primary is unavailable."""
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        if model == PRIMARY_MODEL:
            print(f"  ⚠ {PRIMARY_MODEL} unavailable ({e}), falling back to {FALLBACK_MODEL}")
            response = client.chat.completions.create(
                model=FALLBACK_MODEL, messages=messages, max_tokens=max_tokens, temperature=0
            )
            return response.choices[0].message.content.strip()
        raise


def generate_sql(
    user_query: str,
    year_filter: tuple[int, int] = (2019, 2025),
    model: str = PRIMARY_MODEL,
) -> str:
    """
    Convert a natural-language query into Databricks SQL using Foundation Model APIs.

    Args:
        user_query:  Free-text question from the user.
        year_filter: Default year range when the user doesn't specify one.
        model:       Databricks Foundation Model endpoint name.

    Returns:
        A SQL string ready to execute against Databricks.
    """
    system_prompt = f"""You are a SQL expert for humanitarian crisis data analysis.

{STAR_SCHEMA_DESCRIPTION}

Additional rules:
- Return ONLY the SQL query — no explanation, no markdown fences, no commentary.
- Use Databricks SQL (Spark SQL dialect).
- Default year filter: WHERE f.year BETWEEN {year_filter[0]} AND {year_filter[1]} — apply unless the user specifies otherwise.
- Skip rows with no useful data: AND (f.requirements_usd IS NOT NULL OR f.pin IS NOT NULL) — always wrap this OR in parentheses to avoid operator precedence bugs."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    sql = _call_llm(messages, model=model)

    if sql.startswith("```"):
        sql = "\n".join(
            line for line in sql.splitlines()
            if not line.startswith("```")
        ).strip()

    return sql


def fix_sql(
    user_query: str,
    failed_sql: str,
    error_message: str,
    year_filter: tuple[int, int] = (2019, 2025),
    model: str = PRIMARY_MODEL,
) -> str:
    """
    Ask the LLM to correct a SQL query that failed execution.

    Sends the original question, the failed SQL, and the Databricks error back
    to the LLM and requests a corrected query.
    """
    system_prompt = f"""You are a SQL expert for humanitarian crisis data analysis.

{STAR_SCHEMA_DESCRIPTION}

Additional rules:
- Return ONLY the corrected SQL query — no explanation, no markdown fences, no commentary.
- Use Databricks SQL (Spark SQL dialect).
- Default year filter: WHERE f.year BETWEEN {year_filter[0]} AND {year_filter[1]} — apply unless the user specifies otherwise.
- Skip rows with no useful data: AND (f.requirements_usd IS NOT NULL OR f.pin IS NOT NULL) — always wrap this OR in parentheses to avoid operator precedence bugs.
- ALWAYS include ORDER BY before LIMIT. Never use LIMIT without ORDER BY."""

    user_message = f"""The following SQL query failed with an error. Please fix it.

Original question:
{user_query}

Failed SQL:
{failed_sql}

Error from Databricks:
{error_message}

Return only the corrected SQL query."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    sql = _call_llm(messages, model=model)

    if sql.startswith("```"):
        sql = "\n".join(
            line for line in sql.splitlines()
            if not line.startswith("```")
        ).strip()

    return sql
