"""
Build a star schema from the unified humanitarian crisis dataset.

Tables:
    dim_country          — one row per country (52 crisis countries)
    dim_sector           — one row per humanitarian sector (15 sectors)
    dim_crisis_plan      — one row per country × year (crisis identity + classification)
    fact_crisis_metrics  — one row per country × year × sector (all numeric metrics)

All tables saved to data/processed/ as parquet and optionally uploaded to Databricks
as ACID-compliant Delta tables.

Run:
    python -c "from src.ingestion.build_star_schema import build_star; build_star()"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


# ── Static lookups ────────────────────────────────────────────────────────────

SECTOR_INFO: dict[str, dict] = {
    "ALL":  {"sector_name": "All Sectors",          "description": "Overall humanitarian plan caseload"},
    "FSC":  {"sector_name": "Food Security",         "description": "Food assistance and agricultural support"},
    "HEA":  {"sector_name": "Health",                "description": "Healthcare services and medical response"},
    "WSH":  {"sector_name": "WASH",                  "description": "Water, sanitation and hygiene"},
    "SHL":  {"sector_name": "Shelter",               "description": "Emergency shelter and non-food items"},
    "PRO":  {"sector_name": "Protection",            "description": "Protection from violence, abuse and exploitation"},
    "EDU":  {"sector_name": "Education",             "description": "Emergency education and learning"},
    "NUT":  {"sector_name": "Nutrition",             "description": "Nutrition support and treatment of malnutrition"},
    "MPC":  {"sector_name": "Multipurpose Cash",     "description": "Multi-purpose cash transfers"},
    "LOG":  {"sector_name": "Logistics",             "description": "Humanitarian supply chain and logistics"},
    "ETC":  {"sector_name": "Emergency Telecoms",    "description": "Emergency telecommunications"},
    "CCCM": {"sector_name": "Camp Management",       "description": "Camp coordination and camp management"},
    "ERY":  {"sector_name": "Early Recovery",        "description": "Livelihoods and early recovery"},
    "COOR": {"sector_name": "Coordination",          "description": "Humanitarian coordination and common services"},
    "OTHER":{"sector_name": "Other",                 "description": "Other humanitarian activities"},
}

# UN region lookup by ISO3
COUNTRY_REGIONS: dict[str, str] = {
    # Sub-Saharan Africa
    "AGO": "SSA", "BDI": "SSA", "BEN": "SSA", "BFA": "SSA", "BWA": "SSA",
    "CAF": "SSA", "CMR": "SSA", "COD": "SSA", "COG": "SSA", "COM": "SSA",
    "DJI": "SSA", "ERI": "SSA", "ETH": "SSA", "GAB": "SSA", "GHA": "SSA",
    "GIN": "SSA", "GMB": "SSA", "GNB": "SSA", "GNQ": "SSA", "KEN": "SSA",
    "LBR": "SSA", "LSO": "SSA", "MDG": "SSA", "MLI": "SSA", "MOZ": "SSA",
    "MRT": "SSA", "MWI": "SSA", "NAM": "SSA", "NER": "SSA", "NGA": "SSA",
    "RWA": "SSA", "SDN": "SSA", "SEN": "SSA", "SLE": "SSA", "SOM": "SSA",
    "SSD": "SSA", "SWZ": "SSA", "TCD": "SSA", "TGO": "SSA", "TZA": "SSA",
    "UGA": "SSA", "ZAF": "SSA", "ZMB": "SSA", "ZWE": "SSA",
    # Middle East & North Africa
    "IRQ": "MENA", "JOR": "MENA", "LBN": "MENA", "LBY": "MENA",
    "PSE": "MENA", "SYR": "MENA", "YEM": "MENA",
    # Asia-Pacific
    "AFG": "APAC", "BGD": "APAC", "IDN": "APAC", "KHM": "APAC",
    "LAO": "APAC", "LKA": "APAC", "MMR": "APAC", "MNG": "APAC",
    "NPL": "APAC", "PAK": "APAC", "PHL": "APAC", "TJK": "APAC",
    "TLS": "APAC", "PRK": "APAC", "VNM": "APAC",
    # Latin America & Caribbean
    "COL": "LAC", "CUB": "LAC", "ECU": "LAC", "GTM": "LAC", "GRD": "LAC",
    "HND": "LAC", "HTI": "LAC", "PER": "LAC", "SLV": "LAC",
    "TTO": "LAC", "VCT": "LAC", "VEN": "LAC",
    # Eastern Europe & Central Asia
    "TKM": "EECA", "TUR": "EECA", "UKR": "EECA",
    # North Africa (within MENA)
    "EGY": "MENA", "TUN": "MENA",
    "IRN": "MENA",
}

COUNTRY_CONTINENTS: dict[str, str] = {
    # Africa — Sub-Saharan
    "AGO": "Africa", "BDI": "Africa", "BEN": "Africa", "BFA": "Africa", "BWA": "Africa",
    "CAF": "Africa", "CMR": "Africa", "COD": "Africa", "COG": "Africa", "COM": "Africa",
    "DJI": "Africa", "ERI": "Africa", "ETH": "Africa", "GAB": "Africa", "GHA": "Africa",
    "GIN": "Africa", "GMB": "Africa", "GNB": "Africa", "GNQ": "Africa", "KEN": "Africa",
    "LBR": "Africa", "LSO": "Africa", "MDG": "Africa", "MLI": "Africa", "MOZ": "Africa",
    "MRT": "Africa", "MWI": "Africa", "NAM": "Africa", "NER": "Africa", "NGA": "Africa",
    "RWA": "Africa", "SDN": "Africa", "SEN": "Africa", "SLE": "Africa", "SOM": "Africa",
    "SSD": "Africa", "SWZ": "Africa", "TCD": "Africa", "TGO": "Africa", "TZA": "Africa",
    "UGA": "Africa", "ZAF": "Africa", "ZMB": "Africa", "ZWE": "Africa",
    # Africa — North Africa (MENA region but geographically Africa)
    "EGY": "Africa", "LBY": "Africa", "TUN": "Africa",
    # Middle East
    "IRN": "Middle East", "IRQ": "Middle East", "JOR": "Middle East", "LBN": "Middle East",
    "PSE": "Middle East", "SYR": "Middle East", "YEM": "Middle East",
    # Asia
    "AFG": "Asia", "BGD": "Asia", "IDN": "Asia", "KHM": "Asia", "LAO": "Asia",
    "LKA": "Asia", "MMR": "Asia", "MNG": "Asia", "NPL": "Asia", "PAK": "Asia",
    "PHL": "Asia", "PRK": "Asia", "TJK": "Asia", "TLS": "Asia", "VNM": "Asia",
    # Latin America & Caribbean
    "COL": "Latin America", "CUB": "Latin America", "ECU": "Latin America",
    "GRD": "Latin America", "GTM": "Latin America", "HND": "Latin America",
    "HTI": "Latin America", "PER": "Latin America", "SLV": "Latin America",
    "TTO": "Latin America", "VCT": "Latin America", "VEN": "Latin America",
    # Europe & Central Asia
    "TKM": "Europe & Central Asia", "TUR": "Europe & Central Asia", "UKR": "Europe & Central Asia",
}

REGION_NAMES: dict[str, str] = {
    "SSA":  "Sub-Saharan Africa",
    "MENA": "Middle East & North Africa",
    "APAC": "Asia-Pacific",
    "LAC":  "Latin America & Caribbean",
    "EECA": "Eastern Europe & Central Asia",
}

COUNTRY_NAMES: dict[str, str] = {
    "AFG": "Afghanistan",          "AGO": "Angola",
    "BDI": "Burundi",              "BEN": "Benin",
    "BFA": "Burkina Faso",         "BGD": "Bangladesh",
    "BWA": "Botswana",             "CAF": "Central African Republic",
    "CMR": "Cameroon",             "COD": "DR Congo",
    "COG": "Congo",                "COL": "Colombia",
    "CUB": "Cuba",
    "COM": "Comoros",              "DJI": "Djibouti",
    "ECU": "Ecuador",              "EGY": "Egypt",
    "ERI": "Eritrea",              "ETH": "Ethiopia",
    "GAB": "Gabon",                "GHA": "Ghana",
    "GIN": "Guinea",               "GMB": "Gambia",
    "GNB": "Guinea-Bissau",        "GNQ": "Equatorial Guinea",
    "GRD": "Grenada",              "GTM": "Guatemala",
    "HND": "Honduras",             "HTI": "Haiti",
    "IDN": "Indonesia",            "IRN": "Iran",
    "IRQ": "Iraq",                 "JOR": "Jordan",
    "KEN": "Kenya",                "KHM": "Cambodia",
    "LAO": "Laos",                 "LBN": "Lebanon",
    "LBR": "Liberia",              "LBY": "Libya",
    "LKA": "Sri Lanka",            "LSO": "Lesotho",
    "MDG": "Madagascar",           "MLI": "Mali",
    "MNG": "Mongolia",
    "MMR": "Myanmar",              "MOZ": "Mozambique",
    "MRT": "Mauritania",           "MWI": "Malawi",
    "NAM": "Namibia",              "NER": "Niger",
    "NGA": "Nigeria",              "NPL": "Nepal",
    "PAK": "Pakistan",             "PER": "Peru",
    "PHL": "Philippines",          "PRK": "North Korea",
    "PSE": "Palestine",            "RWA": "Rwanda",
    "SDN": "Sudan",                "SEN": "Senegal",
    "SLE": "Sierra Leone",         "SLV": "El Salvador",
    "SOM": "Somalia",              "SSD": "South Sudan",
    "SWZ": "Eswatini",             "SYR": "Syria",
    "TCD": "Chad",                 "TGO": "Togo",
    "TJK": "Tajikistan",           "TKM": "Turkmenistan",
    "TLS": "Timor-Leste",          "TTO": "Trinidad & Tobago",
    "TUN": "Tunisia",              "TUR": "Turkey",
    "TZA": "Tanzania",             "UGA": "Uganda",
    "UKR": "Ukraine",              "VCT": "St. Vincent & Grenadines",
    "VEN": "Venezuela",            "VNM": "Vietnam",
    "YEM": "Yemen",                "ZAF": "South Africa",
    "ZMB": "Zambia",               "ZWE": "Zimbabwe",
}


# ── Builders ──────────────────────────────────────────────────────────────────

def _build_dim_country(master: pd.DataFrame) -> pd.DataFrame:
    """One row per country with name, region, and population."""
    pop = (
        master[["country_iso3", "population"]]
        .dropna(subset=["population"])
        .groupby("country_iso3")["population"]
        .max()
        .reset_index()
    )

    countries = pd.DataFrame({"country_iso3": sorted(master["country_iso3"].dropna().unique())})
    countries = countries.merge(pop, on="country_iso3", how="left")
    countries["country_name"] = countries["country_iso3"].map(COUNTRY_NAMES)
    countries["region"] = countries["country_iso3"].map(COUNTRY_REGIONS)
    countries["region_name"] = countries["region"].map(REGION_NAMES)
    countries["continent"] = countries["country_iso3"].map(COUNTRY_CONTINENTS)

    return countries[["country_iso3", "country_name", "region", "region_name", "continent", "population"]]


def _build_dim_sector() -> pd.DataFrame:
    """One row per humanitarian sector — fully static."""
    rows = [
        {"sector": code, "sector_name": info["sector_name"], "description": info["description"]}
        for code, info in SECTOR_INFO.items()
    ]
    return pd.DataFrame(rows)


def _build_dim_crisis_plan(master: pd.DataFrame) -> pd.DataFrame:
    """One row per country × year — crisis identity and classification."""
    cols = [
        "country_iso3", "year",
        "crisis_name", "emergency_group", "emergency_type",
        "is_crisis", "crisis_confidence", "crisis_signals",
    ]
    available = [c for c in cols if c in master.columns]

    plan = (
        master[available]
        .drop_duplicates(subset=["country_iso3", "year"])
        .sort_values(["country_iso3", "year"])
        .reset_index(drop=True)
    )
    return plan


def _build_fact_crisis_metrics(master: pd.DataFrame) -> pd.DataFrame:
    """
    One row per country × year × sector.
    Only keeps rows where at least one numeric metric is present.
    """
    metric_cols = [
        "pin", "targeted",
        "requirements_usd", "fts_funding_usd",
        "cbpf_usd", "cerf_usd",
        "coverage_ratio",
        "cbpf_targeted", "cbpf_reached",
    ]

    # A row must have at least one *financial* metric — targeted-only rows carry no analysis value
    financial_cols = ["pin", "requirements_usd", "fts_funding_usd", "cbpf_usd", "cerf_usd"]

    key_cols = ["country_iso3", "year", "sector"]
    available_metrics = [c for c in metric_cols if c in master.columns]
    available_financial = [c for c in financial_cols if c in master.columns]

    fact = master[key_cols + available_metrics].copy()

    # Drop rows where all financial metrics are null — nothing to analyse
    fact = fact.dropna(subset=available_financial, how="all").reset_index(drop=True)

    return fact.sort_values(["country_iso3", "year", "sector"]).reset_index(drop=True)


def _check_referential_integrity(
    fact: pd.DataFrame,
    dim_country: pd.DataFrame,
    dim_sector: pd.DataFrame,
    dim_crisis_plan: pd.DataFrame,
) -> None:
    """Raise if any FK in the fact table has no matching PK in its dimension."""
    issues: list[str] = []

    unknown_countries = set(fact["country_iso3"]) - set(dim_country["country_iso3"])
    if unknown_countries:
        issues.append(f"fact → dim_country: unknown country_iso3 {unknown_countries}")

    unknown_sectors = set(fact["sector"]) - set(dim_sector["sector"])
    if unknown_sectors:
        issues.append(f"fact → dim_sector: unknown sector codes {unknown_sectors}")

    plan_keys = set(zip(dim_crisis_plan["country_iso3"], dim_crisis_plan["year"]))
    fact_keys = set(zip(fact["country_iso3"], fact["year"]))
    orphan_keys = fact_keys - plan_keys
    if orphan_keys:
        issues.append(f"fact → dim_crisis_plan: {len(orphan_keys)} orphan (country, year) combinations")

    if issues:
        raise ValueError("Referential integrity violations:\n" + "\n".join(f"  • {i}" for i in issues))
    print("  Referential integrity: OK")


def build_star(save: bool = True) -> dict[str, pd.DataFrame]:
    """
    Load the master parquet, split into star schema tables,
    validate referential integrity, and save each table.
    """
    master_path = PROCESSED / "master.parquet"
    if not master_path.exists():
        raise FileNotFoundError(
            f"{master_path} not found.\n"
            "Run: python -c \"from src.ingestion.build_dataset import build; build()\""
        )

    print("Loading master dataset…")
    master = pd.read_parquet(master_path)
    print(f"  {len(master):,} rows × {len(master.columns)} columns")

    print("\nBuilding dim_country…")
    dim_country = _build_dim_country(master)
    print(f"  {len(dim_country)} countries")

    print("Building dim_sector…")
    dim_sector = _build_dim_sector()
    print(f"  {len(dim_sector)} sectors")

    print("Building dim_crisis_plan…")
    dim_crisis_plan = _build_dim_crisis_plan(master)
    print(f"  {len(dim_crisis_plan)} country×year crisis plans")

    print("Building fact_crisis_metrics…")
    fact = _build_fact_crisis_metrics(master)
    print(f"  {len(fact):,} rows")

    print("\nChecking referential integrity…")
    _check_referential_integrity(fact, dim_country, dim_sector, dim_crisis_plan)

    print("\nNull rates in fact table:")
    null_pct = (fact.isnull().mean() * 100).round(1)
    for col, pct in null_pct.items():
        bar = "█" * int(pct / 5)
        print(f"  {col:<25} {pct:5.1f}%  {bar}")

    tables = {
        "dim_country":         dim_country,
        "dim_sector":          dim_sector,
        "dim_crisis_plan":     dim_crisis_plan,
        "fact_crisis_metrics": fact,
    }

    if save:
        for name, df in tables.items():
            path = PROCESSED / f"{name}.parquet"
            df.to_parquet(path, index=False)
            print(f"\nSaved {name} → {path.name}  ({len(df):,} rows × {len(df.columns)} cols)")

    return tables


if __name__ == "__main__":
    build_star()
