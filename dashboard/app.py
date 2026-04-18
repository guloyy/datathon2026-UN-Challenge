# [SCAFFOLD] Claude-generated — not yet run or tested by the team.
# Run with: streamlit run dashboard/app.py
# Depends on: master.parquet (run build_master_dataset first),
#             ANTHROPIC_API_KEY in .env

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.loaders import build_master_dataset
from src.query.llm_parser import parse_query
from src.ranking.ranker import run_pipeline

st.set_page_config(
    page_title="Geo-Insight: Overlooked Crises",
    page_icon="🌍",
    layout="wide",
)

st.sidebar.title("Geo-Insight")
st.sidebar.markdown("Surface humanitarian crises where **need outpaces funding coverage**.")

top_n = st.sidebar.slider("Top N crises", min_value=5, max_value=50, value=15)
include_temporal = st.sidebar.checkbox("Apply temporal penalty (multi-year underfunding)", value=True)


@st.cache_data(show_spinner="Loading master dataset…")
def load_data() -> pd.DataFrame:
    processed = Path(__file__).parent.parent / "data" / "processed" / "master.parquet"
    if processed.exists():
        return pd.read_parquet(processed)
    st.warning("Processed dataset not found — building from raw files.")
    return build_master_dataset(save=True)


df = load_data()

st.title("🌍 Geo-Insight: Which Crises Are Most Overlooked?")

query = st.text_input(
    "Ask a humanitarian question",
    placeholder="e.g. Which crises have the highest proportion of people in need but the lowest fund allocations?",
)

if st.button("Analyse", type="primary") and query:
    with st.spinner("Parsing query…"):
        filters = parse_query(query)

    st.subheader("Filters detected")
    filter_cols = st.columns(4)
    filter_cols[0].metric("Regions", ", ".join(filters.regions) or "All")
    filter_cols[1].metric("Countries", ", ".join(filters.country_iso3s) or "All")
    filter_cols[2].metric("Sectors", ", ".join(filters.sectors) or "All")
    filter_cols[3].metric("Max coverage", f"{filters.max_coverage_ratio:.0%}" if filters.max_coverage_ratio else "Any")

    with st.spinner("Scoring and ranking…"):
        ranked = run_pipeline(df, filters, top_n=top_n, include_temporal=include_temporal)

    if ranked.empty:
        st.error("No crises matched the filters. Try broadening your query.")
        st.stop()

    st.subheader(f"Top {len(ranked)} Most Overlooked Crises")

    display_cols = [c for c in ["rank", "country_iso3", "country_name", "year", "pin",
                                 "requirements_usd", "funding_usd", "coverage_ratio", "gap_score"]
                    if c in ranked.columns]
    st.dataframe(ranked[display_cols], use_container_width=True)

    label_col = "country_name" if "country_name" in ranked.columns else "country_iso3"
    if label_col in ranked.columns:
        fig_bar = px.bar(
            ranked,
            x=label_col,
            y="gap_score",
            color="coverage_ratio",
            color_continuous_scale="RdYlGn_r",
            title="Gap Score by Country (higher = more overlooked)",
            labels={"gap_score": "Gap Score", "coverage_ratio": "Coverage Ratio"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    if "country_iso3" in ranked.columns:
        fig_map = px.choropleth(
            ranked,
            locations="country_iso3",
            color="gap_score",
            hover_name=label_col if label_col in ranked.columns else "country_iso3",
            hover_data=["pin", "coverage_ratio"],
            color_continuous_scale="Reds",
            title="Overlooked Crises Map",
        )
        st.plotly_chart(fig_map, use_container_width=True)

    csv = ranked.to_csv(index=False).encode()
    st.download_button("Download results as CSV", csv, "overlooked_crises.csv", "text/csv")

else:
    st.info("Enter a query above and click **Analyse** to get started.")
    st.markdown("""
**Example queries:**
- *Which crises have the highest proportion of people in need but the lowest fund allocations?*
- *Are there countries with active HRPs where funding is absent or negligible?*
- *Which regions are consistently underfunded relative to need across multiple years?*
- *Show me acute food insecurity hotspots that received less than 10% of requested funding.*
""")

# [SCAFFOLD] AI explanation panel — uncomment once generate_explanation is tested:
#
# show_explanations = st.sidebar.checkbox("Generate AI explanations (slower)", value=False)
# if show_explanations:
#     from src.query.llm_parser import generate_explanation
#     st.subheader("Why these crises rank highest")
#     for _, row in ranked.head(5).iterrows():
#         with st.expander(f"#{int(row.get('rank', '?'))} — {row.get('country_name', row.get('country_iso3', 'Unknown'))}"):
#             st.markdown(generate_explanation(row.to_dict(), query))
