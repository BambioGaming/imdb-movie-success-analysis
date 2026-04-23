from __future__ import annotations

import pandas as pd
import streamlit as st

from app.dashboard.components.about import render_about
from app.dashboard.components.analysis import render_analysis_visualizations
from app.dashboard.components.filters import render_sidebar_filters
from app.dashboard.components.layout import apply_theme, render_callout, render_hero, render_kpi
from app.dashboard.components.modeling import render_modeling, render_prediction_sandbox
from app.services.analytics import summary_metrics
from app.services.data_loader import apply_filters, load_dataset


@st.cache_data(show_spinner=True)
def get_dashboard_dataset():
    return load_dataset()


def run_dashboard() -> None:
    st.set_page_config(
        page_title="IMDb Movie Success Analysis",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()
    render_hero(
        "IMDb Movie Success Intelligence Hub",
        "A portfolio-ready analytics dashboard for exploring IMDb title performance, comparing machine learning models, and testing prediction scenarios with a weighted multi-model sandbox.",
    )

    df = get_dashboard_dataset()
    if "live_rows" not in st.session_state:
        st.session_state.live_rows = pd.DataFrame()
    if not st.session_state.live_rows.empty:
        df = pd.concat([df, st.session_state.live_rows], ignore_index=True)

    filters = render_sidebar_filters(df)
    filtered_df = apply_filters(
        df,
        year_range=filters["year_range"],
        min_rating=filters["min_rating"],
        min_votes=filters["min_votes"],
        title_types=filters["selected_types"],
        genres=filters["selected_genres"],
        search=filters["search_term"],
    )
    if filters["highlight_title"]:
        filtered_df["highlight_state"] = "Normal"
        filtered_df.loc[
            filtered_df["primaryTitle"].str.contains(filters["highlight_title"], case=False, na=False),
            "highlight_state",
        ] = "Highlighted"

    metrics = summary_metrics(filtered_df)
    kpi_columns = st.columns(5)
    kpi_data = [
        ("Titles", f"{metrics['total_titles']:,}", "Records in the current analytical slice"),
        ("Avg Rating", f"{metrics['average_rating']:.2f}", "Mean IMDb score after filtering"),
        ("Median Rating", f"{metrics['median_rating']:.2f}", "Central tendency across selected titles"),
        ("Votes", f"{metrics['total_votes']:,}", "Audience footprint represented here"),
        ("Median Runtime", f"{metrics['median_runtime']:.0f} min", "Typical duration in the filtered view"),
    ]
    for column, (label, value, caption) in zip(kpi_columns, kpi_data):
        with column:
            render_kpi(label, value, caption)

    summary_col1, summary_col2 = st.columns([1.2, 1])
    with summary_col1:
        render_callout(
            "Current Slice",
            f"Preset: {filters['preset_name']}. This view spans {filters['year_range'][0]} to {filters['year_range'][1]} with at least {filters['min_votes']:,} votes and a minimum rating threshold of {filters['min_rating']:.1f}.",
        )
    with summary_col2:
        render_callout(
            "Model Definition",
            f"Modeling is running in {filters['model_mode'].title()} mode, with success defined as rating >= {filters['success_rating']:.1f} and votes >= {filters['success_votes']:,}.",
        )

    tabs = st.tabs(
        [
            "Analysis & Visualizations",
            "Model Lab",
            "Prediction Sandbox",
            "About",
        ]
    )

    with tabs[0]:
        render_analysis_visualizations(filtered_df)
    with tabs[1]:
        render_modeling(filtered_df, filters)
    with tabs[2]:
        render_prediction_sandbox(filtered_df, filters)
    with tabs[3]:
        render_about()


if __name__ == "__main__":
    run_dashboard()
