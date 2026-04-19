"""
INFORM Global Crisis Severity Index loader.

Downloads ALL monthly snapshots from HDX (2019–2025), parses each one,
and computes per-country yearly averages. Averaging across months removes
point-in-time noise from individual snapshots.

Scale note: files up to Dec 2025 use a 1–5 scale.
Normalisation: divide by 5 → [0, 1] before averaging.
"""

from __future__ import annotations

import io
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PARQUET_PATH  = PROCESSED_DIR / "dim_inform_severity.parquet"

HDX_PACKAGE_API = (
    "https://data.humdata.org/api/3/action/package_show"
    "?id=inform-global-crisis-severity-index"
)

# Sheet name candidates across file versions
_SHEET_CANDIDATES = [
    "INFORM Severity - all crises",
    "INFORM Severity - country",
    "GCSI",
]

# Severity column name candidates across versions
_SEV_CANDIDATES = [
    "INFORM Severity Index",
    "Crisis Severity",
    "GCSI",
    "Severity",
]

# Files using 1–10 scale (INFORM changed scale in early 2026)
_SCALE_10_FROM_YEAR = 2026


def _fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _get_all_monthly_urls() -> list[tuple[int, int, str]]:
    """Return [(year, month, url), ...] for all xlsx files in 2019–2025."""
    data = _fetch_bytes(HDX_PACKAGE_API)
    resources = __import__("json").loads(data)["result"]["resources"]

    entries: list[tuple[int, int, str]] = []
    for r in resources:
        url = r.get("url", "")
        if ".xlsx" not in url.lower():
            continue
        m = re.search(r"(20\d{2})(\d{2})", r.get("name", "") + url)
        if not m:
            continue
        yr, mo = int(m.group(1)), int(m.group(2))
        if 2019 <= yr <= 2025 and 1 <= mo <= 12:
            entries.append((yr, mo, url))

    return sorted(entries)


def _parse_file(data: bytes, year: int) -> pd.DataFrame | None:
    """
    Parse one xlsx snapshot. Returns DataFrame with columns:
        country_iso3, severity_raw
    Returns None if the file can't be parsed.
    """
    try:
        xl = pd.ExcelFile(io.BytesIO(data))
    except Exception:
        return None

    sheet = next(
        (s for s in _SHEET_CANDIDATES if s in xl.sheet_names), None
    )
    if sheet is None:
        return None

    try:
        df = xl.parse(sheet, header=1)
    except Exception:
        return None

    iso3_col = next(
        (c for c in df.columns if str(c).strip().upper() == "ISO3"), None
    )
    sev_col = next(
        (c for c in df.columns if str(c).strip() in _SEV_CANDIDATES), None
    )
    if iso3_col is None or sev_col is None:
        return None

    df = df[[iso3_col, sev_col]].copy()
    df.columns = ["country_iso3", "severity_raw"]

    # Drop header/legend rows — valid ISO3s are exactly 3 uppercase letters
    df = df[df["country_iso3"].str.match(r"^[A-Z]{3}$", na=False)].copy()
    df["severity_raw"] = pd.to_numeric(df["severity_raw"], errors="coerce")
    df = df.dropna(subset=["severity_raw"])

    return df if not df.empty else None


def build_inform_parquet(verbose: bool = True) -> pd.DataFrame:
    """
    Download all monthly INFORM snapshots, parse, and average per (country, year).

    Returns DataFrame with columns:
        country_iso3 | year | inform_severity | n_months
    where inform_severity is normalised to [0, 1] and n_months is how many
    monthly snapshots contributed to that country-year average.
    """
    if verbose:
        print("  Fetching INFORM file index from HDX…")
    entries = _get_all_monthly_urls()

    by_year: dict[int, list[tuple[int, str]]] = {}
    for yr, mo, url in entries:
        by_year.setdefault(yr, []).append((mo, url))

    all_rows: list[pd.DataFrame] = []

    for yr in sorted(by_year):
        monthly = by_year[yr]
        scale   = 10 if yr >= _SCALE_10_FROM_YEAR else 5

        if verbose:
            print(f"  {yr}: downloading {len(monthly)} snapshots…", end=" ", flush=True)

        year_rows: list[pd.DataFrame] = []
        for mo, url in monthly:
            try:
                raw = _fetch_bytes(url)
                df  = _parse_file(raw, yr)
                if df is not None:
                    df["month"] = mo
                    year_rows.append(df)
                time.sleep(0.15)   # polite crawl rate
            except Exception:
                pass   # skip failed downloads silently

        if not year_rows:
            if verbose:
                print("no data")
            continue

        combined = pd.concat(year_rows, ignore_index=True)

        # Multiple crises per country per snapshot → take max (worst crisis)
        snapshot_max = (
            combined.groupby(["country_iso3", "month"])["severity_raw"]
            .max()
            .reset_index()
        )

        # Average across all months for the year
        yearly_avg = (
            snapshot_max.groupby("country_iso3")["severity_raw"]
            .agg(severity_raw_mean="mean", n_months="count")
            .reset_index()
        )
        yearly_avg["year"]             = yr
        yearly_avg["inform_severity"]  = (yearly_avg["severity_raw_mean"] / scale).clip(0, 1)
        all_rows.append(yearly_avg)

        if verbose:
            n = len(yearly_avg)
            avg_mo = yearly_avg["n_months"].mean()
            print(f"{n} countries, avg {avg_mo:.1f} months/country")

    if not all_rows:
        raise RuntimeError("No INFORM data could be downloaded.")

    result = pd.concat(all_rows, ignore_index=True)[
        ["country_iso3", "year", "inform_severity", "n_months"]
    ]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(PARQUET_PATH, index=False)
    if verbose:
        print(f"  ✓ Saved {len(result)} country-year rows → {PARQUET_PATH}")

    return result


def get_inform_severity(rebuild: bool = False) -> pd.DataFrame:
    """
    Return INFORM severity table. Uses cached parquet; rebuilds if missing or rebuild=True.

    Columns: country_iso3 | year | inform_severity | n_months
    """
    if not rebuild and PARQUET_PATH.exists():
        return pd.read_parquet(PARQUET_PATH)
    return build_inform_parquet()
