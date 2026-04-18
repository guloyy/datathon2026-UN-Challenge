"""
Data loaders for the five primary HDX datasets.

Each loader reads from data/raw/ (populated by scripts/download_data.py)
and returns a cleaned, normalised DataFrame.

Column aliases below reflect common HDX schema variations — update them
after inspecting your actual downloaded files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Expected filenames written by scripts/download_data.py
_RAW_FILES = {
    "hno":        ["hno.csv", "hno.xlsx"],
    "hrp":        ["hrp.csv", "hrp.xlsx"],
    "funding":    ["funding.csv", "funding.xlsx"],
    "population": ["population.csv", "population.xlsx"],
    "cbpf":       ["cbpf.json"],
}


def _find_raw(alias: str) -> Path:
    for name in _RAW_FILES[alias]:
        p = RAW_DIR / name
        if p.exists():
            return p
    candidates = ", ".join(_RAW_FILES[alias])
    raise FileNotFoundError(
        f"No raw file found for '{alias}' in {RAW_DIR}.\n"
        f"Expected one of: {candidates}\n"
        f"Run:  python scripts/download_data.py --only {alias}"
    )


def _read_tabular(path: Path) -> pd.DataFrame:
    """Read CSV or Excel, normalise column names."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    return df


# ── Individual loaders ────────────────────────────────────────────────────────

def load_hno(path: str | Path | None = None) -> pd.DataFrame:
    """
    Humanitarian Needs Overview — people in need by country/sector.

    Verified columns: country_iso3, year, pin, sector
    """
    if path is None:
        path = _find_raw("hno")

    df = _read_tabular(Path(path))

    # Verified against hno_2026.csv: Country ISO3, Cluster, In Need, Description
    col_aliases = {
        "country_iso3":   "country_iso3",
        "in_need":        "pin",
        "cluster":        "sector",
        "iso3":           "country_iso3",
        "adm0_pcode":     "country_iso3",
        "country_code":   "country_iso3",
        "people_in_need": "pin",
        "pin_total":      "pin",
        "total_pin":      "pin",
        "country":        "country_name",
        "description":    "country_name",
    }
    df.rename(columns={k: v for k, v in col_aliases.items() if k in df.columns}, inplace=True)

    if "pin" in df.columns:
        df["pin"] = pd.to_numeric(df["pin"].astype(str).str.replace(",", ""), errors="coerce")

    # Derive year from filename (e.g. hno_2026.csv → 2026)
    if "year" not in df.columns:
        import re
        years = re.findall(r"(20\d{2})", str(path))
        if years:
            df["year"] = int(years[-1])

    return df


# [SCAFFOLD] load_hrp is defined but not used in build_master_dataset.
# HRP plan metadata is complex (pipe-separated locations, HXL tags).
# FTS funding data (load_funding) is used instead for requirements/funding figures.
# Uncomment and wire up if you need HRP plan codes or revised requirements.
#
# def load_hrp(path: str | Path | None = None) -> pd.DataFrame:
#     """
#     Humanitarian Response Plan metadata (plan names, locations, years).
#     Note: row 0 is HXL hashtags — skipped automatically.
#     Actual schema: code, internalId, startDate, endDate, planVersion,
#                    categories, locations, years, origRequirements, revisedRequirements
#     """
#     if path is None:
#         path = _find_raw("hrp")
#     df = _read_tabular(Path(path))
#     if df.iloc[0].astype(str).str.startswith("#").any():
#         df = df.iloc[1:].reset_index(drop=True)
#     col_aliases = {
#         "code":                "hrp_code",
#         "planversion":         "country_name",
#         "origrequirements":    "requirements_usd",
#         "revisedrequirements": "requirements_revised_usd",
#         "locations":           "country_iso3_raw",
#         "years":               "year_raw",
#     }
#     df.rename(columns={k: v for k, v in col_aliases.items() if k in df.columns}, inplace=True)
#     for col in ("requirements_usd", "requirements_revised_usd"):
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
#     return df


def load_funding(path: str | Path | None = None) -> pd.DataFrame:
    """
    Global requirements and funding — FTS data (fts_requirements_funding_global.csv).

    Verified columns: country_iso3, country_name, year,
                      requirements_usd, funding_usd, coverage_ratio
    """
    if path is None:
        path = _find_raw("funding")

    df = _read_tabular(Path(path))

    # Verified against funding.csv: countryCode, name, year, requirements, funding, percentFunded
    col_aliases = {
        "countrycode":   "country_iso3",
        "name":          "country_name",
        "requirements":  "requirements_usd",
        "funding":       "funding_usd",
        "percentfunded": "coverage_ratio",
    }
    df.rename(columns={k: v for k, v in col_aliases.items() if k in df.columns}, inplace=True)

    for col in ("requirements_usd", "funding_usd"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    if "coverage_ratio" not in df.columns and {"requirements_usd", "funding_usd"}.issubset(df.columns):
        df["coverage_ratio"] = df["funding_usd"] / df["requirements_usd"].replace(0, float("nan"))
    elif "coverage_ratio" in df.columns:
        cr = pd.to_numeric(df["coverage_ratio"], errors="coerce")
        df["coverage_ratio"] = cr.where(cr <= 1, cr / 100)

    return df


def load_cbpf(path: str | Path | None = None) -> pd.DataFrame:
    """
    CBPF pooled fund allocations (JSON from AllocationBudgetTotalsByYearAndFund endpoint).

    Verified columns: countrycode (country name, not ISO3), year, allocations_usd
    NOTE: uses country names not ISO3 — join to master requires a name→ISO3 mapping.
    """
    if path is None:
        path = _find_raw("cbpf")

    p = Path(path)
    if p.suffix.lower() == ".json":
        with open(p) as f:
            data = json.load(f)
        records = data.get("value", data) if isinstance(data, dict) else data
        df = pd.DataFrame(records)
    else:
        df = _read_tabular(p)

    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

    # Verified against cbpf.json: PooledFundName (country name), AllocationYear, ApprovedBudget
    col_aliases = {
        "pooledfundname":  "countrycode",
        "allocationyear":  "year",
        "approvedbudget":  "allocations_usd",
        # Legacy field names kept for safety
        "pfbicountrycode": "countrycode",
        "countryiso3code": "countrycode",
        "allocatedamount": "allocations_usd",
    }
    df.rename(columns={k: v for k, v in col_aliases.items() if k in df.columns}, inplace=True)

    if "allocations_usd" in df.columns:
        df["allocations_usd"] = pd.to_numeric(df["allocations_usd"], errors="coerce")

    return df


# ── Master dataset builder ─────────────────────────────────────────────────────

def build_master_dataset(
    hno_path: str | Path | None = None,
    hrp_path: str | Path | None = None,
    funding_path: str | Path | None = None,
    cbpf_path: str | Path | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Merge HNO + FTS funding into a single analysis-ready DataFrame.
    Join key: country_iso3 + year (outer join).

    Output columns:
        country_iso3, year, pin, sector, country_name,
        requirements_usd, funding_usd, coverage_ratio
    """
    hno = load_hno(hno_path)
    fts = load_funding(funding_path)

    keep_fts = [c for c in ("country_iso3", "country_name", "year",
                             "requirements_usd", "funding_usd", "coverage_ratio")
                if c in fts.columns]
    keep_hno = [c for c in ("country_iso3", "year", "pin", "sector", "severity_phase")
                if c in hno.columns]

    join_on = [c for c in ("country_iso3", "year")
               if c in hno.columns and c in fts.columns]

    master = pd.merge(hno[keep_hno], fts[keep_fts], on=join_on, how="outer")

    # [SCAFFOLD] CBPF join is broken: CBPF uses country names ("Somalia"), not ISO3 codes.
    # A name→ISO3 lookup table is needed before this join will produce non-null values.
    # Uncomment once the mapping is implemented.
    #
    # cbpf_file = cbpf_path or (RAW_DIR / "cbpf.json" if (RAW_DIR / "cbpf.json").exists() else None)
    # if cbpf_file:
    #     cbpf = load_cbpf(cbpf_file)
    #     group_cols = [c for c in ("countrycode", "year") if c in cbpf.columns]
    #     if "countrycode" in cbpf.columns and "allocations_usd" in cbpf.columns:
    #         cbpf_agg = (cbpf.groupby(group_cols)["allocations_usd"]
    #                         .sum().reset_index()
    #                         .rename(columns={"countrycode": "country_iso3",
    #                                          "allocations_usd": "cbpf_allocations_usd"}))
    #         cbpf_join = [c for c in ("country_iso3", "year") if c in cbpf_agg.columns]
    #         master = master.merge(cbpf_agg, on=cbpf_join, how="left")

    if save:
        out = PROCESSED_DIR / "master.parquet"
        master.to_parquet(out, index=False)
        print(f"Saved master dataset → {out}  ({len(master):,} rows)")

    return master
