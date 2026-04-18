"""
Download all required datasets from HDX (CKAN API) and CBPF.

Usage:
    python scripts/download_data.py              # download everything
    python scripts/download_data.py --only hno hrp
    python scripts/download_data.py --list       # show available resources without downloading
    python scripts/download_data.py --force      # re-download even if file exists
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HDX_API = "https://data.humdata.org/api/3/action/package_show"

# Map short alias → HDX dataset slug
HDX_DATASETS: dict[str, str] = {
    "hno":     "global-hpc-hno",
    "hrp":     "humanitarian-response-plans",
    "funding": "global-requirements-and-funding-data",
    "population": "cod-ps-global",
}

# CBPF is a live API endpoint — not on HDX
# AllocationBudgetTotalsByYearAndFund gives approved budget by country/year/org-type
CBPF_URL = "https://cbpfapi.unocha.org/vo2/odata/AllocationBudgetTotalsByYearAndFund?$format=json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hdx_resources(dataset_slug: str) -> list[dict]:
    """Return all resource records for a HDX dataset."""
    resp = requests.get(HDX_API, params={"id": dataset_slug}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"HDX API error for {dataset_slug}: {data.get('error')}")
    return data["result"]["resources"]


def _pick_resource(resources: list[dict]) -> dict:
    """
    Prefer a global CSV/XLSX file, breaking ties by most recent year in the name.
    Priority: global > CSV > requirements/funding keyword > recency.
    """
    import re

    def score(r: dict) -> int:
        name = (r.get("name") or "").lower()
        fmt  = (r.get("format") or "").lower()
        s = 0
        if "global" in name:                               s += 10
        if fmt == "csv":                                   s += 5
        if fmt in ("xlsx", "xls"):                         s += 3
        if "requirement" in name or "funding" in name:     s += 2
        # Prefer most recent year to break ties (e.g. 2026 > 2025 > 2024)
        years = re.findall(r"20(\d{2})", name)
        if years:
            s += int(max(years))
        return s

    return max(resources, key=score)


def _all_csv_resources(resources: list[dict]) -> list[dict]:
    """Return all CSV resources, sorted most-recent-year first."""
    import re

    def year_key(r: dict) -> int:
        years = re.findall(r"2\d{3}", (r.get("name") or ""))
        return int(max(years)) if years else 0

    csvs = [r for r in resources if (r.get("format") or "").lower() == "csv"]
    return sorted(csvs, key=year_key, reverse=True)


def _download_file(url: str, dest: Path, force: bool = False) -> bool:
    """Download url → dest. Returns True if downloaded, False if skipped."""
    if dest.exists() and not force:
        print(f"  [skip] {dest.name} already exists (use --force to re-download)")
        return False

    print(f"  ↓ {dest.name}")
    headers = {"User-Agent": "Mozilla/5.0 humanitarian-datathon-downloader"}
    with requests.get(url, stream=True, timeout=120, headers=headers) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                          desc=dest.name, leave=False) as bar:
            for chunk in resp.iter_content(chunk_size=16_384):
                f.write(chunk)
                bar.update(len(chunk))
    return True


# ── Per-source downloaders ────────────────────────────────────────────────────

def download_hdx(alias: str, slug: str, force: bool = False, list_only: bool = False) -> list[Path]:
    print(f"\n[{alias.upper()}] dataset: {slug}")
    try:
        resources = _hdx_resources(slug)
    except Exception as e:
        print(f"  ERROR fetching resource list: {e}")
        return []

    if list_only:
        for r in resources:
            print(f"  • {r.get('name')!r:50s}  format={r.get('format')}  url={r.get('download_url') or r.get('url')}")
        return []

    # HNO: download all years for temporal analysis
    if alias == "hno":
        return _download_all_hno(resources, force=force)

    chosen = _pick_resource(resources)
    url = chosen.get("download_url") or chosen.get("url")
    if not url:
        print(f"  ERROR: no download URL found in chosen resource")
        return []

    fmt = (chosen.get("format") or Path(url).suffix.lstrip(".") or "csv").lower()
    dest = RAW_DIR / f"{alias}.{fmt}"
    _download_file(url, dest, force=force)
    return [dest]


def _download_all_hno(resources: list[dict], force: bool = False) -> list[Path]:
    """Download every available HNO year as hno_<year>.csv, plus symlink latest as hno.csv."""
    import re
    paths = []
    for r in _all_csv_resources(resources):
        url = r.get("download_url") or r.get("url") or ""
        years = re.findall(r"2\d{3}", r.get("name") or url)
        year = max(years) if years else "unknown"
        dest = RAW_DIR / f"hno_{year}.csv"
        _download_file(url, dest, force=force)
        paths.append(dest)

    # Make hno.csv point to the most recent year so loaders.py finds it
    if paths:
        latest = paths[0]  # already sorted most-recent first
        symlink = RAW_DIR / "hno.csv"
        if symlink.exists() or symlink.is_symlink():
            symlink.unlink()
        symlink.symlink_to(latest.name)
        print(f"  → hno.csv → {latest.name}")

    return paths


def download_cbpf(force: bool = False, list_only: bool = False) -> Path | None:
    alias = "cbpf"
    print(f"\n[{alias.upper()}] live API: {CBPF_URL}")
    if list_only:
        print(f"  • cbpf.json  (CBPF allocations by country)")
        return None

    dest = RAW_DIR / "cbpf.json"
    if dest.exists() and not force:
        print(f"  [skip] cbpf.json already exists (use --force to re-download)")
        return dest

    print(f"  ↓ cbpf.json")
    try:
        resp = requests.get(CBPF_URL, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"  WARNING: CBPF API unavailable ({e}). Skipping — pipeline will run without pooled fund data.")
        return None
    dest.write_bytes(resp.content)
    print(f"  Saved {dest.stat().st_size / 1024:.1f} KB")
    return dest


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download humanitarian datasets for Geo-Insight")
    parser.add_argument("--only", nargs="+", choices=list(HDX_DATASETS) + ["cbpf"],
                        metavar="SOURCE", help="Download only these sources")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    parser.add_argument("--list", action="store_true", dest="list_only",
                        help="List available resources without downloading")
    args = parser.parse_args()

    targets = args.only or list(HDX_DATASETS) + ["cbpf"]

    print(f"Output directory: {RAW_DIR}\n")

    for alias in targets:
        if alias == "cbpf":
            download_cbpf(force=args.force, list_only=args.list_only)
        elif alias in HDX_DATASETS:
            download_hdx(alias, HDX_DATASETS[alias], force=args.force, list_only=args.list_only)

    if not args.list_only:
        _download_naturalearth(force=args.force)
        print(f"\nDone. Files saved to {RAW_DIR}")
        print("Next step: python -c \"from src.ingestion.loaders import build_master_dataset; build_master_dataset()\"")


def _download_naturalearth(force: bool = False) -> None:
    """Download Natural Earth 110m country boundaries for geopandas maps."""
    dest = RAW_DIR / "ne_world.gpkg"
    if dest.exists() and not force:
        print(f"\n[NATURALEARTH] [skip] ne_world.gpkg already exists")
        return
    print("\n[NATURALEARTH] Downloading Natural Earth 110m boundaries…")
    try:
        import geopandas as gpd
        url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(url)
        world.to_file(dest, driver="GPKG")
        print(f"  Saved {dest.stat().st_size / 1024:.1f} KB → {dest}")
    except Exception as e:
        print(f"  WARNING: Natural Earth download failed ({e}). Maps will fall back to Plotly choropleth.")


if __name__ == "__main__":
    main()
