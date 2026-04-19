"""Feature matrix for the prioritization optimizer.

Builds a numeric feature matrix over (country_iso3, year, sector) rows from
master.parquet. Only substantive dimensions are features; country/sector/
emergency labels are filters/stratifiers, not features.

Normalization: log1p (for skewed) → winsorize [5,95]-pct → min-max to [0,1].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = ROOT / "data" / "processed" / "master.parquet"

LOG_FEATURES: tuple[str, ...] = (
    "pin",
    "targeted",
    "reached",
    "population",
    "per_capita_need",
    "requirements_usd",
    "fts_funding_usd",
    "cbpf_usd",
    "cerf_usd",
    "funding_shortfall",
)

DERIVED_RATIOS: tuple[str, ...] = ("cbpf_share", "cerf_share", "reached_ratio")

CORE_GAP_FEATURES: tuple[str, ...] = ("coverage_gap", "need_scale")

FEATURE_ORDER: tuple[str, ...] = (
    CORE_GAP_FEATURES
    + (
        "pin", "targeted", "reached", "population", "per_capita_need",
        "requirements_usd", "fts_funding_usd", "cbpf_usd", "cerf_usd",
        "funding_shortfall",
    )
    + DERIVED_RATIOS
)

FEATURE_LABELS: dict[str, str] = {
    "coverage_gap":      "Low funding coverage",
    "need_scale":        "Scale of people in need",
    "pin":               "People in need (absolute)",
    "targeted":          "People targeted for response",
    "reached":           "People reached",
    "population":        "Country population",
    "per_capita_need":   "Per-capita need",
    "requirements_usd":  "Absolute funding requirement",
    "fts_funding_usd":   "Absolute funding received",
    "cbpf_usd":          "CBPF pooled-fund allocation",
    "cerf_usd":          "CERF emergency allocation",
    "funding_shortfall": "Absolute funding shortfall",
    "cbpf_share":        "CBPF share of total funding",
    "cerf_share":        "CERF share of total funding",
    "reached_ratio":     "Reached / targeted effectiveness",
}


@dataclass
class FeatureMatrix:
    """Cached feature matrix + row metadata."""
    X: np.ndarray                       # (n_rows, n_features)
    feature_names: list[str]
    rows: pd.DataFrame                  # metadata (country_iso3, year, sector, country_name, ...)

    @property
    def n_rows(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


def _winsorize(values: np.ndarray, low_pct: float = 5.0, high_pct: float = 95.0) -> np.ndarray:
    """Clip to [5th, 95th] percentile of non-NaN values."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    lo = np.percentile(finite, low_pct)
    hi = np.percentile(finite, high_pct)
    if lo == hi:
        return values
    return np.clip(values, lo, hi)


def _min_max(values: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]. NaN → 0."""
    out = np.array(values, dtype=float).copy()
    finite = out[np.isfinite(out)]
    if finite.size == 0:
        return np.zeros_like(out)
    lo, hi = finite.min(), finite.max()
    if hi - lo < 1e-12:
        out = np.where(np.isfinite(out), 0.0, 0.0)
        return out
    out = (out - lo) / (hi - lo)
    out = np.where(np.isfinite(out), out, 0.0)
    return out


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "coverage_gap" not in df.columns:
        if "coverage_ratio" in df.columns:
            df["coverage_gap"] = (1.0 - df["coverage_ratio"].clip(0, 1)).fillna(1.0)
        else:
            df["coverage_gap"] = 1.0

    if "need_scale" not in df.columns:
        pin = pd.to_numeric(df.get("pin", pd.Series(0, index=df.index)), errors="coerce").fillna(0).clip(lower=0)
        log_pin = np.log1p(pin)
        pin_max = float(log_pin.max() or 1.0)
        df["need_scale"] = log_pin / pin_max

    pin = pd.to_numeric(df.get("pin", pd.Series(np.nan, index=df.index)), errors="coerce")
    pop = pd.to_numeric(df.get("population", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["per_capita_need"] = pin / pop.replace(0, np.nan)

    req = pd.to_numeric(df.get("requirements_usd", pd.Series(np.nan, index=df.index)), errors="coerce")
    fts = pd.to_numeric(df.get("fts_funding_usd", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["funding_shortfall"] = (req - fts).clip(lower=0)

    cbpf = pd.to_numeric(df.get("cbpf_usd", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    cerf = pd.to_numeric(df.get("cerf_usd", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    total_funding = fts.fillna(0) + cbpf + cerf
    denom = total_funding.replace(0, np.nan)
    df["cbpf_share"] = (cbpf / denom).fillna(0)
    df["cerf_share"] = (cerf / denom).fillna(0)

    cbpf_t = pd.to_numeric(df.get("cbpf_targeted", pd.Series(np.nan, index=df.index)), errors="coerce")
    cbpf_r = pd.to_numeric(df.get("cbpf_reached", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["reached_ratio"] = (cbpf_r / cbpf_t.replace(0, np.nan)).clip(0, 1).fillna(0)

    return df


def build_feature_matrix(df: pd.DataFrame) -> FeatureMatrix:
    """Build the normalized feature matrix from a master-like DataFrame."""
    df = df.reset_index(drop=True).copy()
    df = _add_derived_columns(df)

    columns: list[np.ndarray] = []
    for name in FEATURE_ORDER:
        col = pd.to_numeric(df.get(name, pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy(dtype=float)

        if name in LOG_FEATURES:
            col = np.where(col < 0, 0.0, col)
            col = np.log1p(col)

        if name not in CORE_GAP_FEATURES:
            col = _winsorize(col)
            col = _min_max(col)
        else:
            col = np.where(np.isfinite(col), np.clip(col, 0.0, 1.0), 0.0)

        columns.append(col)

    X = np.column_stack(columns)

    meta_cols = [
        c for c in (
            "country_iso3", "country_name", "year", "sector", "sector_name",
            "coverage_ratio", "pin", "requirements_usd", "fts_funding_usd",
            "emergency_group", "emergency_type", "crisis_confidence",
        ) if c in df.columns
    ]
    rows = df[meta_cols].copy()

    return FeatureMatrix(X=X, feature_names=list(FEATURE_ORDER), rows=rows)


_cached: Optional[FeatureMatrix] = None


def load_master(path: Path = MASTER_PATH) -> pd.DataFrame:
    """Load the master dataset from the processed parquet."""
    if not path.exists():
        raise FileNotFoundError(
            f"master.parquet not found at {path}. "
            "Build it with: "
            "`python -c \"from src.ingestion.build_dataset import build; build()\"`"
        )
    return pd.read_parquet(path)


def get_feature_matrix(force_reload: bool = False) -> FeatureMatrix:
    """Lazy-load and cache the feature matrix at module level."""
    global _cached
    if _cached is None or force_reload:
        _cached = build_feature_matrix(load_master())
    return _cached


def filter_rows(
    fm: FeatureMatrix,
    country_iso3: Optional[str] = None,
    sector: Optional[str] = None,
    region: Optional[str] = None,
    emergency_group: Optional[str] = None,
    year: Optional[int] = None,
    region_to_iso3: Optional[dict[str, Iterable[str]]] = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (X_filtered, rows_filtered) with the given filter applied.

    Country/sector/region/emergency_group/year are filter dimensions, never features.
    """
    mask = pd.Series(True, index=fm.rows.index)

    if country_iso3 is not None and "country_iso3" in fm.rows.columns:
        mask &= fm.rows["country_iso3"].astype(str).str.upper() == country_iso3.upper()

    if sector is not None and "sector" in fm.rows.columns:
        mask &= fm.rows["sector"].astype(str).str.upper() == sector.upper()

    if region is not None and region_to_iso3 is not None and "country_iso3" in fm.rows.columns:
        iso3s = {c.upper() for c in region_to_iso3.get(region.upper(), [])}
        if iso3s:
            mask &= fm.rows["country_iso3"].astype(str).str.upper().isin(iso3s)

    if emergency_group is not None and "emergency_group" in fm.rows.columns:
        mask &= fm.rows["emergency_group"].astype(str).str.lower() == emergency_group.lower()

    if year is not None and "year" in fm.rows.columns:
        mask &= pd.to_numeric(fm.rows["year"], errors="coerce") == int(year)

    idx = fm.rows.index[mask]
    return fm.X[idx.to_numpy()], fm.rows.loc[idx].reset_index(drop=True)


def get_target_row_index(
    rows: pd.DataFrame,
    country_iso3: str,
    sector: Optional[str] = None,
    year: Optional[int] = None,
) -> Optional[int]:
    """Find the row index of the target in a filtered rows DataFrame.

    If year is None, picks the most recent year for that (iso3, sector).
    Returns positional index in `rows`, or None if not found.
    """
    mask = rows["country_iso3"].astype(str).str.upper() == country_iso3.upper()
    if sector is not None and "sector" in rows.columns:
        mask &= rows["sector"].astype(str).str.upper() == sector.upper()
    if year is not None and "year" in rows.columns:
        mask &= pd.to_numeric(rows["year"], errors="coerce") == int(year)

    candidates = rows[mask]
    if candidates.empty:
        return None

    if "year" in candidates.columns:
        candidates = candidates.sort_values("year", ascending=False)

    return int(candidates.index[0])
