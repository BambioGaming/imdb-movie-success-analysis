from __future__ import annotations

import streamlit as st

from app.dashboard.components.layout import render_callout, render_section_intro


def render_about() -> None:
    render_section_intro(
        "About",
        "Project Overview",
        "This dashboard is an end-to-end IMDb analytics and machine learning application designed for coursework, presentation, and portfolio use. It combines interactive data exploration with a reproducible modeling workflow for understanding what movie metadata patterns are associated with stronger IMDb performance.",
        icon="ℹ️",
    )

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        render_callout(
            "Objectives",
            "Explore rating, vote, genre, and temporal patterns in IMDb titles; compare several classification models for predicting title success; and present the findings through a polished analytical dashboard and API.",
        )
        render_callout(
            "Dataset",
            "The project uses the IMDb title basics and title ratings tables. After cleaning and merging, the working dataset includes title type, year, runtime, genre composition, average rating, vote counts, and derived signals such as decade, log-scaled votes, and success score.",
        )
        render_callout(
            "Technologies",
            "Python powers the full stack. Pandas and NumPy handle data shaping, scikit-learn powers preprocessing and modeling, Plotly supports interactive charts, Streamlit delivers the dashboard UI, and FastAPI exposes reusable backend endpoints.",
        )
    with col2:
        render_callout(
            "Models Used",
            "The modeling workflow compares Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, K-Nearest Neighbors, and tuned variants in full mode. The dashboard reports holdout metrics, cross-validation, confusion matrices, ROC curves where available, threshold behavior, calibration, and feature importance when the model supports it.",
        )
        render_callout(
            "Success Definition",
            "A title is labeled successful when it meets both an analyst-controlled rating threshold and a vote threshold. This keeps the target transparent while allowing the dashboard user to redefine what success means for different analytical scenarios.",
        )
        render_callout(
            "How To Use",
            "Start with the sidebar to filter titles and define success thresholds. Review the Analysis & Visualizations tab for data insights, use Model Lab to inspect any trained model and adjust ensemble weights, then move to Prediction Sandbox to test a hypothetical title across one or more models.",
        )

    st.subheader("Workflow")
    st.markdown(
        """
        1. Filter the IMDb slice you want to study from the sidebar.
        2. Explore the consolidated analysis sections to understand ratings, votes, genres, trends, and data quality.
        3. Open Model Lab to compare models, adjust weights, and inspect diagnostics for any selected model.
        4. Use Prediction Sandbox to enter movie metadata, compare outputs across models, and review the weighted ensemble probability.
        """
    )
