from __future__ import annotations

import pandas as pd
import streamlit as st

from app.dashboard.components.layout import render_callout, render_section_intro
from app.dashboard.components.sections import (
    render_content_types,
    render_data_quality,
    render_explorer,
    render_genres,
    render_overview,
    render_pairwise_insight,
    render_popularity_quality,
    render_trends,
)
from app.services.analytics import key_findings


def render_analysis_visualizations(filtered_df: pd.DataFrame) -> None:
    render_section_intro(
        "Analysis",
        "Analysis & Visualizations",
        "All exploratory analysis now lives in a single curated workspace. Use the sections below to move from big-picture patterns to deeper content, audience, genre, time, and data-quality views without losing context.",
        icon="📈",
    )

    findings = key_findings(filtered_df)[:4]
    if findings:
        summary_cols = st.columns(len(findings))
        for column, finding in zip(summary_cols, findings):
            with column:
                render_callout("Insight Snapshot", finding)

    with st.expander("Overview & Rating Distribution", expanded=True):
        render_overview(filtered_df)

    with st.expander("Content Types, Popularity, and Pairwise Relationships", expanded=False):
        render_content_types(filtered_df)
        st.markdown("")
        render_popularity_quality(filtered_df)
        st.markdown("")
        render_pairwise_insight(filtered_df)

    with st.expander("Genres and Time Trends", expanded=False):
        render_genres(filtered_df)
        st.markdown("")
        render_trends(filtered_df)

    with st.expander("Explorer and Data Quality", expanded=False):
        explorer_col, quality_col = st.columns([1.2, 1], gap="large")
        with explorer_col:
            render_explorer(filtered_df)
        with quality_col:
            render_data_quality(filtered_df)
