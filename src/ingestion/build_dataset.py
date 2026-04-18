"""
Build the unified humanitarian crisis master dataset.

Sources:
    HNO   — people in need by country/year/sector  (data/raw/hpc_hno_*.csv)
    FTS   — requirements + funding by country/year/cluster  (data/raw/fts_requirements_funding_cluster_global.csv)
    CBPF  — pooled fund allocations by country/year/cluster  (data/cbpf/SectorsOverview_*.csv)
    CERF  — central emergency fund by country/year/cluster  (data/cerf/allocations__*.csv)
    POP   — country population baseline  (data/raw/cod_population_admin0.csv)

Output:
    data/processed/master.parquet  (~one row per country × year × sector)

Run:
    python -c "from src.ingestion.build_dataset import build; build()"
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CBPF_DIR = ROOT / "data" / "cbpf"
CERF_DIR = ROOT / "data" / "cerf"
OUT = ROOT / "data" / "processed" / "master.parquet"

# ── Cluster name → standard code ──────────────────────────────────────────────
# Maps messy names from CBPF/CERF/HNO to a consistent set of sector codes.

CLUSTER_MAP: dict[str, str] = {
    # Food
    "food security":                       "FSC",
    "food security and agriculture":       "FSC",
    "fsc":                                 "FSC",
    "agriculture":                         "FSC",
    "food":                                "FSC",
    # Health
    "health":                              "HEA",
    "hea":                                 "HEA",
    # WASH
    "water, sanitation and hygiene":       "WSH",
    "wash":                                "WSH",
    "wsh":                                 "WSH",
    "water sanitation hygiene":            "WSH",
    # Shelter
    "emergency shelter":                   "SHL",
    "shelter":                             "SHL",
    "shelter and nfi":                     "SHL",
    "emergency shelter and nfi":           "SHL",
    "shl":                                 "SHL",
    # Protection
    "protection":                          "PRO",
    "pro":                                 "PRO",
    "pro-gbv":                             "PRO",
    "pro-cpn":                             "PRO",
    "pro-hlp":                             "PRO",
    "pro-min":                             "PRO",
    "gender-based violence":               "PRO",
    "child protection":                    "PRO",
    "mine action":                         "PRO",
    # Education
    "education":                           "EDU",
    "edu":                                 "EDU",
    # Nutrition
    "nutrition":                           "NUT",
    "nut":                                 "NUT",
    # Logistics / coordination / other
    "logistics":                           "LOG",
    "emergency telecommunications":        "ETC",
    "camp coordination and camp management": "CCCM",
    "cccm":                                "CCCM",
    "early recovery":                      "ERY",
    "multi-sector":                        "MPC",
    "multipurpose cash":                   "MPC",
    "mpc":                                 "MPC",
    "coordination":                        "COOR",
    "ccm":                                 "COOR",
    # Overall plan caseload
    "all":                                 "ALL",
    "plan caseload":                       "ALL",
    "final hrp caseload":                  "ALL",
    "hrp caseload":                        "ALL",
}

def _norm_cluster(raw: str | float) -> str:
    if pd.isna(raw):
        return "OTHER"
    key = str(raw).strip().lower()
    return CLUSTER_MAP.get(key, "OTHER")


# ── Country name → ISO3 ───────────────────────────────────────────────────────
# CBPF and CERF use country/fund names instead of ISO3 codes.

NAME_TO_ISO3: dict[str, str] = {
    "afghanistan":                  "AFG",
    "burkina faso":                 "BFA",
    "burundi":                      "BDI",
    "cameroon":                     "CMR",
    "central african republic":     "CAF",
    "chad":                         "TCD",
    "colombia":                     "COL",
    "democratic republic of the congo": "COD",
    "drc":                          "COD",
    "dr congo":                     "COD",
    "ethiopia":                     "ETH",
    "haiti":                        "HTI",
    "iraq":                         "IRQ",
    "jordan":                       "JOR",
    "kenya":                        "KEN",
    "lebanon":                      "LBN",
    "libya":                        "LBY",
    "mali":                         "MLI",
    "mozambique":                   "MOZ",
    "myanmar":                      "MMR",
    "niger":                        "NER",
    "nigeria":                      "NGA",
    "occupied palestinian territory": "OPT",
    "state of palestine":           "PSE",
    "pakistan":                     "PAK",
    "philippines":                  "PHL",
    "somalia":                      "SOM",
    "south sudan":                  "SSD",
    "sudan":                        "SDN",
    "syria":                        "SYR",
    "syrian arab republic":         "SYR",
    "ukraine":                      "UKR",
    "venezuela":                    "VEN",
    "yemen":                        "YEM",
    "zimbabwe":                     "ZWE",
    "bangladesh":                   "BGD",
    "cambodia":                     "KHM",
    "ecuador":                      "ECU",
    "el salvador":                  "SLV",
    "eritrea":                      "ERI",
    "guatemala":                    "GTM",
    "guinea":                       "GIN",
    "honduras":                     "HND",
    "indonesia":                    "IDN",
    "iran":                         "IRN",
    "laos":                         "LAO",
    "lao pdr":                      "LAO",
    "madagascar":                   "MDG",
    "malawi":                       "MWI",
    "mauritania":                   "MRT",
    "namibia":                      "NAM",
    "nepal":                        "NPL",
    "north korea":                  "PRK",
    "dprk":                         "PRK",
    "democratic people's republic of korea": "PRK",
    "occupied palestinian territories": "PSE",
    "peru":                         "PER",
    "republic of the congo":        "COG",
    "congo":                        "COG",
    "senegal":                      "SEN",
    "sierra leone":                 "SLE",
    "sri lanka":                    "LKA",
    "tajikistan":                   "TJK",
    "tanzania":                     "TZA",
    "timor-leste":                  "TLS",
    "togo":                         "TGO",
    "trinidad and tobago":          "TTO",
    "tunisia":                      "TUN",
    "turkiye":                      "TUR",
    "turkey":                       "TUR",
    "turkmenistan":                 "TKM",
    "uganda":                       "UGA",
    "zambia":                       "ZMB",
    "ghana":                        "GHA",
    "guinea-bissau":                "GNB",
    "liberia":                      "LBR",
    "angola":                       "AGO",
    "benin":                        "BEN",
    "cabo verde":                   "CPV",
    "gabon":                        "GAB",
    "gambia":                       "GMB",
    "guinea bissau":                "GNB",
    "rwanda":                       "RWA",
    "south africa":                 "ZAF",
    "swaziland":                    "SWZ",
    "eswatini":                     "SWZ",
    "lesotho":                      "LSO",
    "botswana":                     "BWA",
    "comoros":                      "COM",
    "djibouti":                     "DJI",
    "egypt":                        "EGY",
    "equatorial guinea":            "GNQ",
    "ghana":                        "GHA",
    "somalia":                      "SOM",
    "south africa":                 "ZAF",
    "ukraine - kyiv":               "UKR",
    "ukraine - kharkiv":            "UKR",
    "occupied palestinian territory (opt)": "PSE",
    "global":                       None,
    "regional":                     None,
}

def _name_to_iso3(name: str | float) -> str | None:
    if pd.isna(name):
        return None
    key = str(name).strip().lower()
    # Exact match first
    if key in NAME_TO_ISO3:
        return NAME_TO_ISO3[key]
    # Try stripping "humanitarian fund" suffix
    cleaned = re.sub(r"\s*(humanitarian fund|hf|cbpf)\s*$", "", key).strip()
    if cleaned in NAME_TO_ISO3:
        return NAME_TO_ISO3[cleaned]
    return None


# ── Individual loaders ────────────────────────────────────────────────────────

def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def _load_hno() -> pd.DataFrame:
    """
    HNO people-in-need, national-level totals per country/year/sector.
    Returns: country_iso3, year, sector, pin, targeted, reached
    """
    frames = []
    for year in (2024, 2025, 2026):
        for stem in (f"hpc_hno_{year}", f"hno_{year}"):
            p = RAW / f"{stem}.csv"
            if p.exists():
                df = _read_csv(p)
                break
        else:
            continue

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

        rename = {
            "country_iso3": "country_iso3",
            "in_need":      "pin",
            "cluster":      "sector_raw",
            "category":     "category",
            "targeted":     "targeted",
            "reached":      "reached",
        }
        df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

        df["year"] = year

        # Filter to national-level rows (no admin breakdown) + total/null category
        if "admin_1_pcode" in df.columns:
            df = df[df["admin_1_pcode"].isna()].copy()
        if "category" in df.columns:
            cat = df["category"].astype(str).str.strip().str.lower()
            df = df[cat.isin(["total", "nan", ""])].copy()

        df["pin"] = pd.to_numeric(
            df["pin"].astype(str).str.replace(",", ""), errors="coerce"
        )
        df["targeted"] = pd.to_numeric(
            df.get("targeted", pd.Series(dtype=float)).astype(str).str.replace(",", ""),
            errors="coerce",
        )
        df["reached"] = pd.to_numeric(
            df.get("reached", pd.Series(dtype=float)).astype(str).str.replace(",", ""),
            errors="coerce",
        )

        df["sector"] = df["sector_raw"].apply(_norm_cluster)

        keep = [c for c in ("country_iso3", "year", "sector", "pin", "targeted", "reached") if c in df.columns]
        agg = (
            df[keep]
            .groupby(["country_iso3", "year", "sector"], dropna=False)
            .agg({"pin": "max", "targeted": "max", "reached": "max"})
            .reset_index()
        )
        frames.append(agg)

    if not frames:
        print("  WARNING: No HNO files found in data/raw/")
        return pd.DataFrame(columns=["country_iso3", "year", "sector", "pin", "targeted", "reached"])

    return pd.concat(frames, ignore_index=True)


def _load_fts_cluster() -> pd.DataFrame:
    """
    FTS cluster-level requirements and funding.
    Returns: country_iso3, year, sector, requirements_usd, fts_funding_usd
    """
    path = RAW / "fts_requirements_funding_cluster_global.csv"
    if not path.exists():
        print(f"  WARNING: {path.name} not found")
        return pd.DataFrame(columns=["country_iso3", "year", "sector", "requirements_usd", "fts_funding_usd"])

    df = _read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

    rename = {
        "countrycode":   "country_iso3",
        "cluster":       "sector_raw",
        "requirements":  "requirements_usd",
        "funding":       "fts_funding_usd",
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    df["year"] = pd.to_numeric(df.get("year", pd.Series(dtype=float)), errors="coerce")
    df["requirements_usd"] = pd.to_numeric(df.get("requirements_usd", pd.Series(dtype=float)), errors="coerce")
    df["fts_funding_usd"] = pd.to_numeric(df.get("fts_funding_usd", pd.Series(dtype=float)), errors="coerce")

    df["sector"] = df["sector_raw"].apply(_norm_cluster)

    # Filter to analysis window and valid rows
    df = df[df["year"].between(2018, 2026)].copy()

    agg = (
        df.groupby(["country_iso3", "year", "sector"], dropna=False)
        .agg({"requirements_usd": "sum", "fts_funding_usd": "sum"})
        .reset_index()
    )
    return agg


def _load_cbpf() -> pd.DataFrame:
    """
    CBPF SectorsOverview: pooled fund allocations by country/year/cluster.
    Returns: country_iso3, year, sector, cbpf_usd, cbpf_targeted, cbpf_reached
    """
    candidates = sorted(CBPF_DIR.glob("SectorsOverview*.csv"))
    if not candidates:
        print("  WARNING: No CBPF SectorsOverview file found")
        return pd.DataFrame(columns=["country_iso3", "year", "sector", "cbpf_usd", "cbpf_targeted", "cbpf_reached"])

    df = _read_csv(candidates[-1])  # most recent
    df.columns = df.columns.str.strip()

    rename = {
        "Year":               "year",
        "CBPF Name":          "fund_name",
        "Cluster":            "sector_raw",
        "Total Allocations":  "cbpf_usd",
        "Targeted People":    "cbpf_targeted",
        "Reached People":     "cbpf_reached",
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    df["country_iso3"] = df["fund_name"].apply(_name_to_iso3)
    df = df[df["country_iso3"].notna()].copy()

    df["cbpf_usd"] = pd.to_numeric(df["cbpf_usd"], errors="coerce")
    df["cbpf_targeted"] = pd.to_numeric(df["cbpf_targeted"], errors="coerce")
    df["cbpf_reached"] = pd.to_numeric(df["cbpf_reached"], errors="coerce")
    df["sector"] = df["sector_raw"].apply(_norm_cluster)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    agg = (
        df.groupby(["country_iso3", "year", "sector"], dropna=False)
        .agg({"cbpf_usd": "sum", "cbpf_targeted": "sum", "cbpf_reached": "sum"})
        .reset_index()
    )
    return agg


def _load_cerf() -> pd.DataFrame:
    """
    CERF allocations by country/year/cluster.
    Returns: country_iso3, year, sector, cerf_usd
    """
    candidates = sorted(CERF_DIR.glob("allocations__*.csv"))
    # Prefer the file with Fund/country column (7 columns)
    df = None
    for p in candidates:
        tmp = _read_csv(p)
        if "Fund" in tmp.columns or "fund" in tmp.columns.str.lower().tolist():
            df = tmp
            break

    if df is None:
        print("  WARNING: No suitable CERF allocations file found")
        return pd.DataFrame(columns=["country_iso3", "year", "sector", "cerf_usd"])

    df.columns = df.columns.str.strip()
    rename = {
        "Year":    "year",
        "Fund":    "fund_name",
        "Cluster": "sector_raw",
        "Budget":  "cerf_usd",
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    df["country_iso3"] = df["fund_name"].apply(_name_to_iso3)
    df = df[df["country_iso3"].notna()].copy()

    df["cerf_usd"] = pd.to_numeric(df["cerf_usd"], errors="coerce")
    df["sector"] = df["sector_raw"].apply(_norm_cluster)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    df = df[df["year"].between(2018, 2026)].copy()

    agg = (
        df.groupby(["country_iso3", "year", "sector"], dropna=False)
        .agg({"cerf_usd": "sum"})
        .reset_index()
    )
    return agg


def _load_crisis_names() -> dict[tuple[str, int], str]:
    """
    Return {(country_iso3, year): crisis_name} from HRP planVersion.
    When a country has multiple plans in a year, prefer HRP/HNRP over Flash Appeal.
    """
    path = RAW / "humanitarian-response-plans.csv"
    if not path.exists():
        return {}

    df = _read_csv(path)
    df.columns = df.columns.str.strip()
    if df.iloc[0].astype(str).str.startswith("#").any():
        df = df.iloc[1:].reset_index(drop=True)

    # Priority: full HRP/HNRP > Flash Appeal > other
    def _plan_priority(cat: str) -> int:
        low = str(cat).lower()
        if "humanitarian response plan" in low or "humanitarian needs and response plan" in low:
            return 3
        if "flash appeal" in low:
            return 2
        if "consolidated" in low or "regional" in low:
            return 1
        return 0

    result: dict[tuple[str, int], str] = {}
    for _, row in df.iterrows():
        locs = str(row.get("locations", "")).split("|")
        years_raw = re.findall(r"20\d{2}", str(row.get("years", "")))
        plan_name = str(row.get("planVersion", "")).strip()
        priority = _plan_priority(str(row.get("categories", "")))

        for iso3 in locs:
            iso3 = iso3.strip().upper()
            if len(iso3) != 3:
                continue
            for yr in years_raw:
                key = (iso3, int(yr))
                # Keep highest-priority plan name
                existing = result.get(key, ("", -1))
                if priority > existing[1]:
                    result[key] = (plan_name, priority)

    return {k: v[0] for k, v in result.items()}


def _load_emergency_types() -> dict[tuple[str, int], tuple[str, str]]:
    """
    Return {(country_iso3, year): (emergency_group, emergency_type)} from CERF timeline.
    When multiple emergencies exist in a year, keep the one with the largest budget.
    """
    candidates = sorted(CERF_DIR.glob("alloctimeline*.csv"))
    if not candidates:
        return {}

    df = _read_csv(candidates[-1])
    df.columns = df.columns.str.strip()

    rename = {
        "Year": "year", "Fund": "fund_name",
        "Emergency Group": "emergency_group",
        "Emergency Type": "emergency_type",
        "Budget": "budget",
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    df["country_iso3"] = df["fund_name"].apply(_name_to_iso3)
    df = df[df["country_iso3"].notna()].copy()
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce").fillna(0)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Per country+year, keep the emergency type with the highest budget
    idx = df.groupby(["country_iso3", "year"])["budget"].idxmax()
    top = df.loc[idx].copy()

    result: dict[tuple[str, int], tuple[str, str]] = {}
    for _, row in top.iterrows():
        key = (str(row["country_iso3"]), int(row["year"]))
        result[key] = (
            str(row.get("emergency_group", "")),
            str(row.get("emergency_type", "")),
        )
    return result


def _load_hrp_countries() -> set[tuple[str, int]]:
    """
    Return a set of (country_iso3, year) tuples that have a formal
    humanitarian response plan (HRP, Flash Appeal, HNRP, or CAP).
    These are the gold-standard active-crisis markers from OCHA.
    """
    path = RAW / "humanitarian-response-plans.csv"
    if not path.exists():
        print("  WARNING: humanitarian-response-plans.csv not found")
        return set()

    df = _read_csv(path)
    df.columns = df.columns.str.strip()

    # Skip HXL hashtag row
    if df.iloc[0].astype(str).str.startswith("#").any():
        df = df.iloc[1:].reset_index(drop=True)

    # Crisis plan categories (exclude "Other" and development plans)
    crisis_keywords = (
        "humanitarian response plan", "flash appeal",
        "humanitarian needs and response plan", "hnrp",
        "consolidated", "regional response plan",
    )

    def _is_crisis_plan(cat: str) -> bool:
        if pd.isna(cat):
            return False
        low = str(cat).lower()
        return any(kw in low for kw in crisis_keywords)

    df = df[df["categories"].apply(_is_crisis_plan)].copy()

    pairs: set[tuple[str, int]] = set()
    for _, row in df.iterrows():
        locs = str(row.get("locations", "")).split("|")
        years_raw = str(row.get("years", ""))
        years = re.findall(r"20\d{2}", years_raw)
        for iso3 in locs:
            iso3 = iso3.strip().upper()
            if len(iso3) == 3:
                for yr in years:
                    pairs.add((iso3, int(yr)))
    return pairs


def _load_population() -> pd.DataFrame:
    """
    National-level total population from COD admin0.
    Returns: country_iso3, population
    """
    path = RAW / "cod_population_admin0.csv"
    if not path.exists():
        print("  WARNING: cod_population_admin0.csv not found")
        return pd.DataFrame(columns=["country_iso3", "population"])

    df = _read_csv(path)
    df.columns = df.columns.str.strip()

    # Keep total population rows (Gender=all, Age_range=all)
    mask = pd.Series(True, index=df.index)
    if "Gender" in df.columns:
        mask &= df["Gender"].astype(str).str.lower().str.strip() == "all"
    if "Age_range" in df.columns:
        mask &= df["Age_range"].astype(str).str.lower().str.strip() == "all"

    df = df[mask].copy()

    iso_col = next((c for c in df.columns if c in ("ISO3", "ISO_A3", "country_iso3")), None)
    pop_col = next((c for c in df.columns if c.lower() == "population"), None)

    if not iso_col or not pop_col:
        print("  WARNING: Could not find ISO3 or Population column in COD admin0")
        return pd.DataFrame(columns=["country_iso3", "population"])

    df = df[[iso_col, pop_col]].rename(columns={iso_col: "country_iso3", pop_col: "population"})
    df["population"] = pd.to_numeric(df["population"], errors="coerce")

    # Keep most recent estimate per country (sum across groups if multiple rows)
    pop = df.groupby("country_iso3")["population"].max().reset_index()
    return pop


# ── Crisis name normalization ─────────────────────────────────────────────────

# French/Spanish → English phrase translations
_PHRASE_MAP = [
    # French HRP titles
    (r"Besoins Humanitaires et Plan de R[eé]ponse",  "Humanitarian Needs and Response Plan"),
    (r"Plan de R[eé]ponse Humanitaire",              "Humanitarian Response Plan"),
    (r"Plan de Respuesta Humanitaria",               "Humanitarian Response Plan"),
    (r"Necesidades Humanitarias y Plan de Respuesta","Humanitarian Needs and Response Plan"),
    (r"Plan de Respuesta a Prioridades Comunitarias","Community Priorities Response Plan"),
    # Country name translations (French → English)
    (r"R[eé]publique [Cc]entrafricaine",             "Central African Republic"),
    (r"R[eé]publique [Dd][eé]mocratique du Congo",   "Democratic Republic of the Congo"),
]

def _normalize_crisis_name(raw: str | None) -> str | None:
    if not raw or pd.isna(raw):
        return None
    name = raw.strip()
    # Apply phrase translations
    for pattern, replacement in _PHRASE_MAP:
        name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
    # Strip trailing year (e.g. "... 2026" or "... 2025 - 2026")
    name = re.sub(r"\s+20\d{2}(?:\s*[-–]\s*20\d{2})?$", "", name).strip()
    return name


def _infer_emergency_group(name: str | None) -> str | None:
    """Infer emergency group from crisis name keywords."""
    if not name or pd.isna(name):
        return None
    low = name.lower()
    if any(w in low for w in ["flood", "drought", "earthquake", "cyclone", "typhoon",
                               "hurricane", "storm", "volcano", "tsunami", "landslide"]):
        return "Natural Disaster"
    if any(w in low for w in ["ebola", "cholera", "disease", "outbreak", "epidemic", "covid"]):
        return "Disease Outbreak"
    if any(w in low for w in ["refugee", "rohingya", "displacement", "migrant", "rmrp",
                               "conflict", "hostilities", "violence"]):
        return "Conflict-related"
    # Most HRPs without a specific keyword are conflict-driven crises
    if any(w in low for w in ["humanitarian needs and response plan", "humanitarian response plan",
                               "flash appeal", "humanitarian response priorities"]):
        return "Conflict-related"
    return "Others"


def _infer_emergency_type(name: str | None) -> str | None:
    """Infer emergency type from crisis name keywords."""
    if not name or pd.isna(name):
        return None
    low = name.lower()
    if "flood" in low:           return "Natural Disaster - Flood"
    if "drought" in low:         return "Natural Disaster - Drought"
    if "earthquake" in low:      return "Natural Disaster - Earthquake"
    if any(w in low for w in ["cyclone", "typhoon", "hurricane", "storm"]):
        return "Natural Disaster - Storm"
    if "volcano" in low:         return "Natural Disaster - Volcanic Eruption"
    if "ebola" in low:           return "Disease Outbreak - Ebola"
    if "cholera" in low:         return "Disease Outbreak - Cholera"
    if any(w in low for w in ["disease", "outbreak", "epidemic"]):
        return "Disease Outbreak"
    if any(w in low for w in ["rohingya", "refugee", "migrant", "rmrp"]):
        return "Conflict-related - Displacement"
    if any(w in low for w in ["hostilities", "violence", "clashes"]):
        return "Conflict-related - Violence/Clashes"
    if "conflict" in low:        return "Conflict-related"
    return "Conflict-related"  # default for generic HRPs


# ── Master dataset builder ────────────────────────────────────────────────────

def build(save: bool = True) -> pd.DataFrame:
    """
    Build and return the unified master dataset.
    Joins HNO + FTS + CBPF + CERF + Population on country_iso3 × year × sector.
    """
    print("Loading HNO (people in need)…")
    hno = _load_hno()
    print(f"  → {len(hno):,} rows, {hno['country_iso3'].nunique()} countries")

    print("Loading FTS cluster requirements & funding…")
    fts = _load_fts_cluster()
    print(f"  → {len(fts):,} rows, {fts['country_iso3'].nunique()} countries")

    print("Loading CBPF sector allocations…")
    cbpf = _load_cbpf()
    print(f"  → {len(cbpf):,} rows, {cbpf['country_iso3'].nunique()} countries")

    print("Loading CERF allocations…")
    cerf = _load_cerf()
    print(f"  → {len(cerf):,} rows, {cerf['country_iso3'].nunique()} countries")

    print("Loading population…")
    pop = _load_population()
    print(f"  → {len(pop):,} countries")

    print("Loading HRP crisis plans…")
    hrp_pairs = _load_hrp_countries()
    print(f"  → {len(hrp_pairs):,} country×year crisis plan entries")

    print("Loading crisis names from HRP…")
    crisis_names = _load_crisis_names()
    print(f"  → {len(crisis_names):,} named crises")

    print("Loading emergency types from CERF…")
    emergency_types = _load_emergency_types()
    print(f"  → {len(emergency_types):,} country×year emergency classifications")

    # ── Merge ─────────────────────────────────────────────────────────────────
    print("Merging…")

    # Start with FTS as the spine (widest country × year × sector coverage)
    master = pd.merge(fts, hno, on=["country_iso3", "year", "sector"], how="outer")
    master = pd.merge(master, cbpf, on=["country_iso3", "year", "sector"], how="left")
    master = pd.merge(master, cerf, on=["country_iso3", "year", "sector"], how="left")
    master = pd.merge(master, pop, on="country_iso3", how="left")

    # ── Derived columns ───────────────────────────────────────────────────────
    master["coverage_ratio"] = (
        master["fts_funding_usd"] / master["requirements_usd"].replace(0, np.nan)
    ).clip(0, 1)

    # ── Crisis classification ─────────────────────────────────────────────────
    # is_crisis = True if any signal confirms an active humanitarian emergency.
    # crisis_confidence = "high" if HNO or HRP (formal OCHA assessment),
    #                     "medium" if only CBPF/CERF (funded but no formal plan record)

    hrp_key = master.apply(
        lambda r: (str(r["country_iso3"]), int(r["year"])) if pd.notna(r["year"]) else ("", 0),
        axis=1,
    )

    sig_hno  = master["pin"].notna()
    sig_hrp  = hrp_key.apply(lambda k: k in hrp_pairs)
    sig_cbpf = master["cbpf_usd"].notna()
    sig_cerf = master["cerf_usd"].notna()

    master["is_crisis"] = sig_hno | sig_hrp | sig_cbpf | sig_cerf

    def _signals(row: pd.Series) -> str:
        s = []
        if row["pin"] == row["pin"]:   s.append("HNO")   # notna check
        if sig_hrp[row.name]:          s.append("HRP")
        if row["cbpf_usd"] == row["cbpf_usd"]: s.append("CBPF")
        if row["cerf_usd"] == row["cerf_usd"]:  s.append("CERF")
        return ",".join(s) if s else "none"

    master["crisis_signals"] = master.apply(_signals, axis=1)

    master["crisis_confidence"] = "none"
    master.loc[sig_cbpf | sig_cerf, "crisis_confidence"] = "medium"
    master.loc[sig_hno | sig_hrp,   "crisis_confidence"] = "high"

    # ── Crisis name + emergency type ──────────────────────────────────────────
    raw_names = hrp_key.apply(lambda k: crisis_names.get(k, None))
    master["crisis_name"] = raw_names.apply(_normalize_crisis_name)

    master["emergency_group"] = hrp_key.apply(
        lambda k: emergency_types.get(k, (None, None))[0]
    )
    master["emergency_type"] = hrp_key.apply(
        lambda k: emergency_types.get(k, (None, None))[1]
    )

    # Fill missing emergency_group/type by inferring from crisis_name keywords
    no_group = master["emergency_group"].isna()
    master.loc[no_group, "emergency_group"] = master.loc[no_group, "crisis_name"].apply(_infer_emergency_group)
    master.loc[no_group, "emergency_type"] = master.loc[no_group, "crisis_name"].apply(_infer_emergency_type)

    # Sector display name
    code_to_name = {
        "FSC": "Food Security",   "HEA": "Health",       "WSH": "WASH",
        "SHL": "Shelter",         "PRO": "Protection",   "EDU": "Education",
        "NUT": "Nutrition",       "MPC": "Multipurpose Cash",
        "LOG": "Logistics",       "ETC": "Emergency Telecoms",
        "CCCM": "Camp Management","ERY": "Early Recovery",
        "COOR": "Coordination",   "ALL": "All Sectors",  "OTHER": "Other",
    }
    master["sector_name"] = master["sector"].map(code_to_name).fillna(master["sector"])

    # ── Filter to crisis rows only ────────────────────────────────────────────
    # Drop non-crisis rows — these are FTS entries with no humanitarian signal
    before = len(master)
    master = master[master["is_crisis"]].copy()
    master = master[master["year"].between(2018, 2026)].copy()
    print(f"  Dropped {before - len(master):,} non-crisis rows → {len(master):,} rows remaining")

    # Drop columns that are still >85% null after crisis filter
    null_rates = master.isnull().mean()
    drop_cols = null_rates[null_rates > 0.85].index.tolist()
    if drop_cols:
        print(f"  Dropping high-null columns (>85%): {drop_cols}")
        master.drop(columns=drop_cols, inplace=True)

    # Column order
    cols = [
        "country_iso3", "year", "sector", "sector_name",
        "is_crisis", "crisis_confidence", "crisis_signals",
        "crisis_name", "emergency_group", "emergency_type",
        "pin", "targeted", "reached", "population",
        "requirements_usd", "fts_funding_usd", "cbpf_usd", "cerf_usd",
        "coverage_ratio",
        "cbpf_targeted", "cbpf_reached",
    ]
    master = master[[c for c in cols if c in master.columns]]

    print(f"\nMaster dataset: {len(master):,} rows × {len(master.columns)} columns")
    print(f"Countries: {master['country_iso3'].nunique()}")
    print(f"Years: {int(master['year'].min())}–{int(master['year'].max())}")
    print(f"Sectors: {sorted(master['sector'].dropna().unique())}")
    print("\nNull rates:")
    null_pct = (master.isnull().mean() * 100).round(1)
    for col, pct in null_pct.items():
        print(f"  {col:<25} {pct}%")

    if save:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        master.to_parquet(OUT, index=False)
        print(f"\nSaved → {OUT}")

    return master


if __name__ == "__main__":
    build()
