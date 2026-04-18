"""
Gap scoring engine.

Computes a composite "overlooked" score for each crisis/country row.

Score components
----------------
1. coverage_gap  = 1 - (funding_usd / requirements_usd)  ∈ [0, 1]
2. need_scale    = log1p(pin) normalised to [0, 1] across all rows
3. gap_score     = coverage_gap * need_scale  (higher = more overlooked)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


MIN_PIN = 50_000          # filter out trivially small crises
MIN_REQUIREMENTS = 1e6   # at least $1M requested to be in scope


def compute_gap_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add gap scoring columns to the master dataset.
    Returns df with added columns: coverage_gap, need_scale, gap_score, in_scope.
    Verified: tests pass against synthetic data in tests/test_scoring.py.
    """
    df = df.copy()

    for col in ("funding_usd", "requirements_usd", "pin"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "coverage_ratio" not in df.columns:
        df["coverage_ratio"] = df["funding_usd"] / df["requirements_usd"].replace(0, np.nan)
    df["coverage_gap"] = (1 - df["coverage_ratio"].clip(0, 1)).fillna(1.0)

    pin = df["pin"].clip(lower=0).fillna(0)
    log_pin = np.log1p(pin)
    pin_max = log_pin.max() or 1.0
    df["need_scale"] = log_pin / pin_max

    df["gap_score"] = df["coverage_gap"] * df["need_scale"]

    df["in_scope"] = (
        (df.get("pin", pd.Series(0)) >= MIN_PIN)
        & (df.get("requirements_usd", pd.Series(0)) >= MIN_REQUIREMENTS)
    )

    return df


# [SCAFFOLD] Temporal penalty for multi-year chronic underfunding (bonus task).
# Not yet tested against real data — the streak logic needs validation
# once we have confirmed multi-year HNO+FTS rows aligned by country+year.
#
# def apply_temporal_penalty(df: pd.DataFrame, years_column: str = "year") -> pd.DataFrame:
#     """
#     Boost gap_score for crises underfunded (coverage_ratio < 0.5) for N consecutive years.
#     Multiplier: 1 + 0.15 * N
#     """
#     if years_column not in df.columns:
#         return df
#     df = df.copy()
#     df[years_column] = pd.to_numeric(df[years_column], errors="coerce")
#     df_sorted = df.sort_values([c for c in ("country_iso3", years_column) if c in df.columns])
#
#     def _streak(group: pd.DataFrame) -> pd.Series:
#         underfunded = group["coverage_ratio"].fillna(0) < 0.5
#         streak = underfunded.groupby((~underfunded).cumsum()).cumsum()
#         return streak
#
#     if "country_iso3" in df_sorted.columns:
#         df_sorted["underfunding_streak"] = (
#             df_sorted.groupby("country_iso3", group_keys=False).apply(_streak)
#         )
#     else:
#         df_sorted["underfunding_streak"] = _streak(df_sorted)
#
#     df_sorted["temporal_multiplier"] = 1 + 0.15 * df_sorted["underfunding_streak"].fillna(0)
#     df_sorted["gap_score"] = df_sorted["gap_score"] * df_sorted["temporal_multiplier"]
#     return df_sorted


def rank_crises(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Return the top-N most overlooked in-scope crises, sorted by gap_score descending.
    Verified: tests pass in tests/test_scoring.py.
    """
    in_scope = df[df.get("in_scope", pd.Series(True, index=df.index))].copy()
    ranked = in_scope.sort_values("gap_score", ascending=False).head(top_n)
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked.reset_index(drop=True)
