# [SCAFFOLD] Claude-generated — not yet run or tested by the team.
# Entry point: run_pipeline(master_df, filters, top_n) → ranked DataFrame
# Depends on: src/query/llm_parser.py (QueryFilter), src/scoring/gap_score.py

from __future__ import annotations

import pandas as pd

from src.query.llm_parser import QueryFilter
from src.scoring.gap_score import compute_gap_scores, rank_crises

# [SCAFFOLD] apply_temporal_penalty is commented out in gap_score.py — remove this
# import once it's implemented and tested.
# from src.scoring.gap_score import apply_temporal_penalty

# Region → ISO3 mapping — extend as needed
REGION_ISO3: dict[str, list[str]] = {
    "SSA":  ["SOM", "SSD", "COD", "CAF", "NER", "MLI", "BFA", "TCD", "ETH", "MOZ", "ZWE"],
    "MENA": ["YEM", "SYR", "IRQ", "LBY", "PSE", "LBN", "SDN"],
    "LAC":  ["HTI", "VEN", "COL", "GTM", "HND"],
    "ASIA": ["AFG", "MMR", "BGD", "PRK", "PAK"],
}


def apply_filters(df: pd.DataFrame, filters: QueryFilter) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    if filters.country_iso3s and "country_iso3" in df.columns:
        iso3_upper = [c.upper() for c in filters.country_iso3s]
        mask &= df["country_iso3"].str.upper().isin(iso3_upper)

    if filters.regions and "country_iso3" in df.columns:
        region_countries: set[str] = set()
        for r in filters.regions:
            region_countries.update(REGION_ISO3.get(r.upper(), []))
        if region_countries:
            mask &= df["country_iso3"].str.upper().isin(region_countries)

    if filters.sectors and "sector" in df.columns:
        sectors_lower = [s.lower() for s in filters.sectors]
        mask &= df["sector"].str.lower().isin(sectors_lower)

    if filters.max_coverage_ratio is not None and "coverage_ratio" in df.columns:
        mask &= df["coverage_ratio"].fillna(0) <= filters.max_coverage_ratio

    if filters.min_pin is not None and "pin" in df.columns:
        mask &= df["pin"].fillna(0) >= filters.min_pin

    if filters.year_range is not None and "year" in df.columns:
        start, end = filters.year_range
        mask &= df["year"].between(start, end)

    if filters.require_hrp and "hrp_status" in df.columns:
        mask &= df["hrp_status"].notna() & (df["hrp_status"].str.lower() != "none")

    return df[mask].copy()


def run_pipeline(
    master_df: pd.DataFrame,
    filters: QueryFilter,
    top_n: int = 20,
    include_temporal: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: filter → score → rank.
    Returns a ranked DataFrame sorted by gap_score descending.

    Note: temporal penalty is commented out until apply_temporal_penalty is verified.
    """
    filtered = apply_filters(master_df, filters)
    scored = compute_gap_scores(filtered)

    # [SCAFFOLD] Re-enable once apply_temporal_penalty is uncommented and tested:
    # if include_temporal or filters.chronic_only:
    #     scored = apply_temporal_penalty(scored)
    # if filters.chronic_only and "underfunding_streak" in scored.columns:
    #     scored = scored[scored["underfunding_streak"] >= 2]

    ranked = rank_crises(scored, top_n=top_n)
    return ranked
