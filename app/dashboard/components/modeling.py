from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.dashboard.components.layout import render_callout, render_mini_stats, render_section_intro
from app.services.data_loader import load_dataset
from app.services.modeling import (
    ModelingConfig,
    compare_models,
    load_cached_comparison,
    load_persisted_model,
    predict_success,
)


def style_figure(fig, height: int, legend: bool = True):
    fig.update_layout(
        height=height,
        showlegend=legend,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)",
        font=dict(color="#e5edf7"),
        title_font=dict(size=20, color="#f8fafc"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
        zeroline=False,
        linecolor="rgba(148,163,184,0.16)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
        zeroline=False,
        linecolor="rgba(148,163,184,0.16)",
    )
    return fig


@st.cache_resource(show_spinner=False)
def cached_model_comparison(
    year_start: int,
    year_end: int,
    min_votes: int,
    success_rating: float,
    success_votes: int,
    random_seed: int,
    mode: str,
    force_retrain: bool = False,
):
    config = ModelingConfig(
        min_votes=min_votes,
        year_start=year_start,
        year_end=year_end,
        success_rating=success_rating,
        success_votes=success_votes,
        random_state=random_seed,
        mode=mode,
        cv_folds=3 if mode == "fast" else 5,
        sample_size=50000 if mode == "fast" else None,
    )
    return compare_models(load_dataset(), config=config, persist=True, force_retrain=force_retrain)


def render_modeling(filtered_df: pd.DataFrame, filters: dict) -> None:
    render_section_intro(
        "Model Lab",
        "Success Prediction Comparison",
        "This tab keeps the full modeling workflow intact while presenting the results like an experiment "
        "review board: comparable metrics, diagnostics, feature signals, and a live prediction sandbox.",
        icon="🤖",
    )
    config = ModelingConfig(
        min_votes=filters["min_votes"],
        year_start=filters["year_range"][0],
        year_end=filters["year_range"][1],
        success_rating=filters["success_rating"],
        success_votes=filters["success_votes"],
        random_state=filters["random_seed"],
        mode=filters["model_mode"],
        cv_folds=3 if filters["model_mode"] == "fast" else 5,
        sample_size=50000 if filters["model_mode"] == "fast" else None,
    )
    cached = load_cached_comparison(config)

    controls_col1, controls_col2 = st.columns([1.2, 1])
    with controls_col1:
        if cached is not None:
            st.caption("Cached model results found for the current configuration. Loading from disk for a faster experience.")
        else:
            st.caption("No cached model results yet for this configuration. The first run will train and save them.")
    with controls_col2:
        refresh_models = st.button("Refresh model training", use_container_width=True)

    comparison = cached_model_comparison(
        filters["year_range"][0],
        filters["year_range"][1],
        filters["min_votes"],
        filters["success_rating"],
        filters["success_votes"],
        filters["random_seed"],
        filters["model_mode"],
        force_retrain=refresh_models,
    )

    render_callout("Modeling Note", comparison["dataset"]["note"])
    render_mini_stats(
        [
            ("Modeling Rows", f"{comparison['dataset']['rows']:,}"),
            ("Success Rate", f"{comparison['dataset']['success_rate']:.1%}"),
            ("Train Rows", f"{comparison['dataset']['train_rows']:,}"),
            ("Test Rows", f"{comparison['dataset']['test_rows']:,}"),
        ]
    )
    render_callout(
        "Mode",
        f"You are running in {filters['model_mode'].title()} mode. Fast mode uses a smaller sampled training set and lighter validation, while Full mode runs the complete comparison and tuning workflow.",
    )
    comparison_df = pd.DataFrame(comparison["comparison"])
    st.dataframe(comparison_df.round(4), use_container_width=True, height=250)
    st.download_button(
        "Download model comparison results",
        comparison_df.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        chart = px.bar(
            comparison_df.melt(id_vars="model_name", value_vars=["accuracy", "precision", "recall", "f1_score", "roc_auc"]),
            x="model_name",
            y="value",
            color="variable",
            barmode="group",
            title="Test Metrics by Model",
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        chart.update_layout(xaxis_title="")
        style_figure(chart, 460, legend=True)
        st.plotly_chart(chart, use_container_width=True)

    with col2:
        cv_chart = px.line_polar(
            comparison_df,
            r=["cv_accuracy", "cv_f1", "cv_roc_auc"],
            theta=["CV Accuracy", "CV F1", "CV ROC-AUC"],
            line_close=True,
            title="Cross-Validation Stability Snapshot",
        )
        cv_chart.update_layout(
            height=460,
            paper_bgcolor="rgba(0,0,0,0)",
            polar=dict(bgcolor="rgba(15,23,42,0.55)", radialaxis=dict(gridcolor="rgba(148,163,184,0.12)")),
            font=dict(color="#e5edf7"),
        )
        st.plotly_chart(cv_chart, use_container_width=True)

    st.subheader("Results Summary")
    results_df = pd.DataFrame(comparison["results_summary"])
    if not results_df.empty:
        results_df["interpretation"] = results_df.apply(
            lambda row: "Best balance of predictive strength and stability"
            if row["model_name"] == comparison["best_model"]["model_name"]
            else ("Solid but less balanced" if row["f1_score"] >= results_df["f1_score"].median() else "Weaker holdout behavior"),
            axis=1,
        )
        st.dataframe(results_df, use_container_width=True, height=250)

    best = comparison["best_model"]
    render_callout(
        "Best Model Selection",
        f"{best['model_name']} leads this comparison with F1 {best['test_metrics']['f1_score']:.3f} "
        f"and ROC-AUC {best['test_metrics']['roc_auc'] if best['test_metrics']['roc_auc'] is not None else 'N/A'}.",
    )
    render_callout("Why The Best Model Won", best["why_best_model_won"])
    diag1, diag2 = st.columns([1, 1])
    with diag1:
        confusion = best["test_metrics"]["confusion_matrix"]
        confusion_fig = px.imshow(
            confusion,
            text_auto=True,
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 0", "Actual 1"],
            title="Confusion Matrix",
            color_continuous_scale="Blues",
        )
        style_figure(confusion_fig, 400, legend=False)
        st.plotly_chart(confusion_fig, use_container_width=True)

    with diag2:
        roc_data = best["test_metrics"]["roc_curve"]
        if roc_data["fpr"]:
            roc_fig = go.Figure()
            roc_fig.add_trace(
                go.Scatter(
                    x=roc_data["fpr"],
                    y=roc_data["tpr"],
                    mode="lines",
                    name=f"{best['model_name']} (AUC={best['test_metrics']['roc_auc']})",
                    line=dict(color="#d1495b", width=3),
                )
            )
            roc_fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    line=dict(color="#627d98", dash="dash"),
                    name="Chance",
                )
            )
            roc_fig.update_layout(title="ROC Curve", height=400)
            style_figure(roc_fig, 400, legend=True)
            st.plotly_chart(roc_fig, use_container_width=True)
        else:
            st.info("ROC curve is unavailable when only one class is present in the evaluation split.")

    diag3, diag4 = st.columns([1, 1])
    with diag3:
        threshold_df = pd.DataFrame(best["threshold_analysis"]["table"])
        threshold_chart = px.line(
            threshold_df,
            x="threshold",
            y=["precision", "recall", "f1_score"],
            title=f"Threshold Tuning (best F1 at {best['threshold_analysis']['best_threshold']})",
        )
        style_figure(threshold_chart, 420, legend=True)
        st.plotly_chart(threshold_chart, use_container_width=True)
    with diag4:
        calibration_df = pd.DataFrame(best["calibration"])
        if not calibration_df.empty:
            calibration_fig = go.Figure()
            calibration_fig.add_trace(
                go.Scatter(
                    x=calibration_df["predicted_probability"],
                    y=calibration_df["observed_frequency"],
                    mode="lines+markers",
                    name="Observed",
                )
            )
            calibration_fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash"))
            )
            calibration_fig.update_layout(title="Calibration Curve")
            style_figure(calibration_fig, 420, legend=True)
            st.plotly_chart(calibration_fig, use_container_width=True)

    importance_df = pd.DataFrame(best["feature_importance"])
    if not importance_df.empty:
        importance_chart = px.bar(
            importance_df.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Top Predictive Features",
            color="importance",
            color_continuous_scale="OrRd",
        )
        style_figure(importance_chart, 520, legend=False)
        st.plotly_chart(importance_chart, use_container_width=True)

    perm_df = pd.DataFrame(best["permutation_importance"])
    if not perm_df.empty:
        perm_chart = px.bar(
            perm_df.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Permutation Importance",
            color="importance",
            color_continuous_scale="Tealgrn",
        )
        style_figure(perm_chart, 520, legend=False)
        st.plotly_chart(perm_chart, use_container_width=True)

    st.subheader("Simple Error Analysis")
    err1, err2 = st.columns(2)
    with err1:
        st.markdown("**Most Confident False Positives**")
        st.dataframe(pd.DataFrame(best["error_analysis"]["false_positives"]), use_container_width=True)
    with err2:
        st.markdown("**Most Costly False Negatives**")
        st.dataframe(pd.DataFrame(best["error_analysis"]["false_negatives"]), use_container_width=True)

    st.subheader("Prediction Sandbox")
    with st.form("prediction_form"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            title_type = st.selectbox("Title type", sorted(filtered_df["titleType"].dropna().unique().tolist()))
            start_year = st.number_input("Start year", min_value=1920, max_value=2035, value=2024)
        with col_b:
            runtime_minutes = st.number_input("Runtime", min_value=30, max_value=300, value=110)
            is_adult = st.selectbox("Adult content", options=[0, 1], format_func=lambda v: "No" if v == 0 else "Yes")
        with col_c:
            genre_choices = sorted({genre for value in filtered_df["genres"].astype(str) for genre in value.split(",") if genre})
            selected_genres = st.multiselect("Genres", genre_choices, default=genre_choices[:2] if len(genre_choices) > 1 else genre_choices)
        submitted = st.form_submit_button("Predict success")

    if submitted:
        model = load_persisted_model(config)
        if model is None:
            model = load_persisted_model()
        result = predict_success(
            {
                "titleType": title_type,
                "startYear": start_year,
                "runtimeMinutes": runtime_minutes,
                "isAdult": is_adult,
                "genres": selected_genres,
            },
            model,
        )
        confidence_label = (
            "High confidence" if result["success_probability"] >= 0.75 or result["success_probability"] <= 0.25 else "Moderate confidence"
        )
        st.success(
            f"{result['predicted_label']} with probability {result['success_probability']:.2%}"
        )
        render_callout(
            "Probability Interpretation",
            f"{confidence_label}. This probability should be interpreted as a model-estimated success likelihood under the current definition, not a guaranteed outcome.",
        )
