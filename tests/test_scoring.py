"""Basic unit tests for gap scoring logic."""

import pandas as pd
import pytest
from src.scoring.gap_score import compute_gap_scores, rank_crises


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "country_iso3": ["SOM", "YEM", "SYR", "NOR"],
        "country_name": ["Somalia", "Yemen", "Syria", "Norway"],
        "year": [2023, 2023, 2023, 2023],
        "pin": [7_000_000, 21_000_000, 15_000_000, 0],
        "requirements_usd": [1.2e9, 4.3e9, 3.1e9, 0],
        "funding_usd": [3e8, 5e8, 2.8e9, 0],
    })


def test_coverage_gap_computed(sample_df):
    scored = compute_gap_scores(sample_df)
    # Yemen: 500M / 4300M ≈ 0.116 → gap ≈ 0.884
    yemen = scored[scored["country_iso3"] == "YEM"].iloc[0]
    assert pytest.approx(yemen["coverage_gap"], abs=0.01) == 1 - (5e8 / 4.3e9)


def test_gap_score_ordering(sample_df):
    scored = compute_gap_scores(sample_df)
    in_scope = scored[scored["in_scope"]]
    ranked = rank_crises(in_scope)
    # Yemen has high PIN and low coverage → should rank near top
    assert ranked.iloc[0]["country_iso3"] in {"YEM", "SOM"}


def test_norway_out_of_scope(sample_df):
    scored = compute_gap_scores(sample_df)
    norway = scored[scored["country_iso3"] == "NOR"].iloc[0]
    assert not norway["in_scope"]
