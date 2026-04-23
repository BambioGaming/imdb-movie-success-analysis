from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.dashboard.components.layout import render_callout, render_kpi, render_mini_stats, render_section_intro
from app.services.data_loader import load_dataset
from app.services.modeling import (
    ModelingConfig,
    compare_models_with_artifacts,
    load_cached_comparison,
    normalize_model_weights,
    predict_success_across_models,
)


SANDBOX_STATE_KEYS = {
    "titleType": "sandbox_title_type",
    "startYear": "sandbox_start_year",
    "runtimeMinutes": "sandbox_runtime",
    "isAdult": "sandbox_is_adult",
    "genres": "sandbox_genres",
}


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


def build_modeling_config(filters: dict[str, Any]) -> ModelingConfig:
    return ModelingConfig(
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


@st.cache_resource(show_spinner=False)
def cached_model_bundle(
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
    return compare_models_with_artifacts(load_dataset(), config=config, persist=True, force_retrain=force_retrain)


def get_modeling_bundle(filters: dict[str, Any], force_retrain: bool = False):
    return cached_model_bundle(
        filters["year_range"][0],
        filters["year_range"][1],
        filters["min_votes"],
        filters["success_rating"],
        filters["success_votes"],
        filters["random_seed"],
        filters["model_mode"],
        force_retrain=force_retrain,
    )


def _weighted_leaderboard(comparison_df: pd.DataFrame, normalized_weights: dict[str, float]) -> pd.DataFrame:
    leaderboard = comparison_df.copy()
    leaderboard["roc_auc_filled"] = leaderboard["roc_auc"].fillna(0)
    leaderboard["quality_index"] = (
        leaderboard["f1_score"] * 0.4
        + leaderboard["roc_auc_filled"] * 0.25
        + leaderboard["accuracy"] * 0.15
        + leaderboard["precision"] * 0.1
        + leaderboard["recall"] * 0.1
    )
    leaderboard["assigned_weight"] = leaderboard["model_name"].map(normalized_weights).fillna(0)
    leaderboard["weighted_ensemble_score"] = leaderboard["quality_index"] * leaderboard["assigned_weight"]
    leaderboard["weight_pct"] = leaderboard["assigned_weight"] * 100
    return leaderboard.sort_values(["weighted_ensemble_score", "quality_index"], ascending=False)


def _default_sample_input(filtered_df: pd.DataFrame) -> dict[str, Any]:
    if filtered_df.empty:
        return {
            "titleType": "movie",
            "startYear": 2024,
            "runtimeMinutes": 110,
            "isAdult": 0,
            "genres": ["Drama", "Thriller"],
        }
    base_row = filtered_df.sort_values(["success_score", "numVotes"], ascending=False).iloc[0]
    genres = [genre.strip() for genre in str(base_row["genres"]).split(",") if genre.strip()]
    return {
        "titleType": str(base_row["titleType"]),
        "startYear": int(base_row["startYear"]),
        "runtimeMinutes": int(base_row["runtimeMinutes"]),
        "isAdult": int(base_row["isAdult"]),
        "genres": genres[:3] if genres else ["Drama"],
    }


def _reset_sandbox_state(filtered_df: pd.DataFrame) -> None:
    sample = _default_sample_input(filtered_df)
    for field, key in SANDBOX_STATE_KEYS.items():
        st.session_state[key] = sample[field]


def _seed_sandbox_state(filtered_df: pd.DataFrame) -> None:
    if SANDBOX_STATE_KEYS["titleType"] not in st.session_state:
        _reset_sandbox_state(filtered_df)


def _confidence_label(probability: float) -> str:
    distance = abs(probability - 0.5)
    if distance >= 0.3:
        return "High confidence"
    if distance >= 0.15:
        return "Moderate confidence"
    return "Low confidence"


def _render_model_selector(model_names: list[str]) -> str:
    return st.selectbox(
        "Inspect a model",
        model_names,
        index=0,
        help="Select any trained model to review its metrics, confusion matrix, ROC curve, feature importance, and diagnostics.",
    )


def _render_weight_controls(model_names: list[str], comparison_df: pd.DataFrame) -> dict[str, float]:
    st.subheader("Model Weights")
    render_callout(
        "How weights work",
        "Set each model's influence for the weighted ensemble. The dashboard normalizes your choices automatically so the final weights sum to 100%, then updates the weighted ranking and sandbox ensemble result immediately.",
    )
    default_weights = st.session_state.get(
        "raw_model_weights",
        {name: round(100 / len(model_names), 1) for name in model_names},
    )

    raw_weights: dict[str, float] = {}
    weight_cols = st.columns(2)
    for idx, model_name in enumerate(model_names):
        with weight_cols[idx % 2]:
            raw_weights[model_name] = st.slider(
                f"{model_name} weight",
                min_value=0.0,
                max_value=100.0,
                value=float(default_weights.get(model_name, 0.0)),
                step=1.0,
                key=f"weight_slider_{model_name}",
            )

    normalized = normalize_model_weights(raw_weights)
    st.session_state["raw_model_weights"] = raw_weights
    st.session_state["normalized_model_weights"] = normalized

    leaderboard = _weighted_leaderboard(comparison_df, normalized)
    weight_summary = leaderboard[["model_name", "weight_pct", "quality_index", "weighted_ensemble_score"]].rename(
        columns={
            "model_name": "Model",
            "weight_pct": "Normalized Weight (%)",
            "quality_index": "Quality Index",
            "weighted_ensemble_score": "Weighted Ensemble Score",
        }
    )
    st.dataframe(weight_summary.round(4), use_container_width=True, height=260)

    influence_chart = px.bar(
        leaderboard,
        x="weighted_ensemble_score",
        y="model_name",
        orientation="h",
        color="weight_pct",
        color_continuous_scale="Sunset",
        title="Weighted Ensemble Ranking",
        labels={"model_name": "", "weighted_ensemble_score": "Weighted ensemble score", "weight_pct": "Weight %"},
    )
    style_figure(influence_chart, 420, legend=False)
    st.plotly_chart(influence_chart, use_container_width=True)
    return normalized


def _render_overview_stats(comparison: dict[str, Any], comparison_df: pd.DataFrame) -> None:
    best_model = comparison["best_model"]
    render_callout("Modeling Note", comparison["dataset"]["note"])
    render_callout(
        "Best Model",
        f"{best_model['model_name']} is currently the best model for this configuration, with "
        f"F1 {best_model['test_metrics']['f1_score']:.3f}"
        + (
            f" and ROC-AUC {best_model['test_metrics']['roc_auc']:.3f}."
            if best_model["test_metrics"]["roc_auc"] is not None
            else "."
        ),
    )
    render_callout("Why It Leads", best_model["why_best_model_won"])
    render_mini_stats(
        [
            ("Modeling Rows", f"{comparison['dataset']['rows']:,}"),
            ("Success Rate", f"{comparison['dataset']['success_rate']:.1%}"),
            ("Train Rows", f"{comparison['dataset']['train_rows']:,}"),
            ("Test Rows", f"{comparison['dataset']['test_rows']:,}"),
        ]
    )
    st.dataframe(comparison_df.round(4), use_container_width=True, height=280)

    metrics_chart = px.bar(
        comparison_df.melt(id_vars="model_name", value_vars=["accuracy", "precision", "recall", "f1_score", "roc_auc"]),
        x="model_name",
        y="value",
        color="variable",
        barmode="group",
        title="Test Metrics by Model",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        labels={"model_name": "", "value": "Score", "variable": "Metric"},
    )
    metrics_chart.update_layout(xaxis_tickangle=25)
    style_figure(metrics_chart, 460, legend=True)
    st.plotly_chart(metrics_chart, use_container_width=True)


def _render_selected_model(model_name: str, model_detail: dict[str, Any], comparison_df: pd.DataFrame) -> None:
    selected_row = comparison_df[comparison_df["model_name"] == model_name].iloc[0]
    roc_auc_value = model_detail["test_metrics"]["roc_auc"]
    render_callout(
        "Selected Model",
        f"{model_name} currently scores accuracy {selected_row['accuracy']:.3f}, precision {selected_row['precision']:.3f}, recall {selected_row['recall']:.3f}, and F1 {selected_row['f1_score']:.3f}.",
    )
    metric_cols = st.columns(5)
    metric_cards = [
        ("Accuracy", f"{selected_row['accuracy']:.3f}", "Holdout classification accuracy"),
        ("Precision", f"{selected_row['precision']:.3f}", "How often positive predictions are correct"),
        ("Recall", f"{selected_row['recall']:.3f}", "How many successful titles were captured"),
        ("F1", f"{selected_row['f1_score']:.3f}", "Balance between precision and recall"),
        ("ROC-AUC", f"{roc_auc_value:.3f}" if roc_auc_value is not None else "N/A", "Ranking quality across thresholds"),
    ]
    for column, (label, value, caption) in zip(metric_cols, metric_cards):
        with column:
            render_kpi(label, value, caption)

    diag_left, diag_right = st.columns(2, gap="large")
    with diag_left:
        confusion = model_detail["test_metrics"]["confusion_matrix"]
        confusion_fig = px.imshow(
            confusion,
            text_auto=True,
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 0", "Actual 1"],
            title=f"{model_name} Confusion Matrix",
            color_continuous_scale="Blues",
        )
        style_figure(confusion_fig, 380, legend=False)
        st.plotly_chart(confusion_fig, use_container_width=True)
    with diag_right:
        roc_data = model_detail["test_metrics"]["roc_curve"]
        if roc_data["fpr"]:
            roc_fig = go.Figure()
            roc_fig.add_trace(
                go.Scatter(
                    x=roc_data["fpr"],
                    y=roc_data["tpr"],
                    mode="lines",
                    name=f"{model_name} (AUC={roc_auc_value:.3f})" if roc_auc_value is not None else model_name,
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
            roc_fig.update_layout(title="ROC Curve")
            style_figure(roc_fig, 380, legend=True)
            st.plotly_chart(roc_fig, use_container_width=True)
        else:
            st.info("ROC curve is unavailable for this model because the evaluation split did not contain both classes.")

    signal_left, signal_right = st.columns(2, gap="large")
    with signal_left:
        importance_df = pd.DataFrame(model_detail.get("feature_importance", []))
        if not importance_df.empty:
            importance_chart = px.bar(
                importance_df.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                title="Feature Importance / Coefficients",
                color="importance",
                color_continuous_scale="OrRd",
            )
            style_figure(importance_chart, 460, legend=False)
            st.plotly_chart(importance_chart, use_container_width=True)
        else:
            st.info("This model does not expose native feature importance or coefficient values.")
    with signal_right:
        perm_df = pd.DataFrame(model_detail.get("permutation_importance", []))
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
            style_figure(perm_chart, 460, legend=False)
            st.plotly_chart(perm_chart, use_container_width=True)
        else:
            st.info("Permutation importance is unavailable for this model configuration.")

    threshold_col, calibration_col = st.columns(2, gap="large")
    with threshold_col:
        threshold_df = pd.DataFrame(model_detail["threshold_analysis"]["table"])
        threshold_chart = px.line(
            threshold_df,
            x="threshold",
            y=["precision", "recall", "f1_score"],
            title=f"Threshold Tuning (best F1 at {model_detail['threshold_analysis']['best_threshold']})",
        )
        style_figure(threshold_chart, 400, legend=True)
        st.plotly_chart(threshold_chart, use_container_width=True)
    with calibration_col:
        calibration_df = pd.DataFrame(model_detail.get("calibration", []))
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
            style_figure(calibration_fig, 400, legend=True)
            st.plotly_chart(calibration_fig, use_container_width=True)
        else:
            st.info("Calibration information is unavailable for this model.")

    st.subheader("Error Analysis")
    fp_col, fn_col = st.columns(2, gap="large")
    with fp_col:
        st.markdown("**Most Confident False Positives**")
        st.dataframe(pd.DataFrame(model_detail["error_analysis"]["false_positives"]), use_container_width=True)
    with fn_col:
        st.markdown("**Most Costly False Negatives**")
        st.dataframe(pd.DataFrame(model_detail["error_analysis"]["false_negatives"]), use_container_width=True)


def render_modeling(filtered_df: pd.DataFrame, filters: dict) -> None:
    render_section_intro(
        "Model Lab",
        "Model Selection, Diagnostics, and Weighted Ranking",
        "Use this tab to inspect any trained model, compare diagnostics side by side at the metric level, and tune model weights for a weighted ensemble view that also powers the separate prediction sandbox.",
        icon="🤖",
    )
    config = build_modeling_config(filters)
    cached = load_cached_comparison(config)

    controls_col1, controls_col2 = st.columns([1.3, 1])
    with controls_col1:
        st.caption(
            "Cached comparison found for this configuration."
            if cached is not None
            else "No cached model summary exists yet for this configuration. The first run will train and persist results."
        )
    with controls_col2:
        refresh_models = st.button("Refresh model training", use_container_width=True)

    comparison, fitted_models = get_modeling_bundle(filters, force_retrain=refresh_models)
    comparison_df = pd.DataFrame(comparison["comparison"])
    model_names = comparison_df["model_name"].tolist()
    model_lookup = {item["model_name"]: item for item in comparison["models"]}
    st.session_state["active_fitted_models"] = fitted_models

    _render_overview_stats(comparison, comparison_df)
    normalized_weights = _render_weight_controls(model_names, comparison_df)
    selected_model = _render_model_selector(model_names)
    st.session_state["selected_model_name"] = selected_model
    _render_selected_model(selected_model, model_lookup[selected_model], comparison_df)


def render_prediction_sandbox(filtered_df: pd.DataFrame, filters: dict) -> None:
    render_section_intro(
        "Prediction Sandbox",
        "Scenario Testing Across Models",
        "Create a hypothetical title, compare predictions across one or more models, and review the weighted ensemble probability driven by the model weights you set in Model Lab.",
        icon="🧪",
    )
    _seed_sandbox_state(filtered_df)
    comparison, fitted_models = get_modeling_bundle(filters, force_retrain=False)
    comparison_df = pd.DataFrame(comparison["comparison"])
    model_names = comparison_df["model_name"].tolist()
    source_df = filtered_df if not filtered_df.empty else load_dataset()
    available_genres = sorted({genre.strip() for value in source_df["genres"].astype(str) for genre in value.split(",") if genre.strip()})
    available_types = sorted(source_df["titleType"].dropna().unique().tolist())
    normalized_weights = st.session_state.get(
        "normalized_model_weights",
        normalize_model_weights({name: 1.0 for name in model_names}),
    )

    action_col1, action_col2, action_col3 = st.columns([1, 1, 1.4])
    with action_col1:
        if st.button("Load sample input", use_container_width=True):
            _reset_sandbox_state(filtered_df)
            st.rerun()
    with action_col2:
        if st.button("Reset fields", use_container_width=True):
            _reset_sandbox_state(filtered_df)
            st.rerun()
    with action_col3:
        sample = _default_sample_input(filtered_df)
        render_callout(
            "Sample scenario",
            f"The optional sample uses a high-performing observed title profile: {sample['titleType']}, {sample['startYear']}, {sample['runtimeMinutes']} minutes, genres {', '.join(sample['genres'])}.",
        )

    render_callout(
        "Input guide",
        "Inputs are limited to the metadata used by the trained models: title type, release year, runtime, adult flag, and genres. Outputs report each model's predicted class, estimated success probability, and a confidence label based on distance from the decision boundary.",
    )

    selected_models = st.multiselect(
        "Models to compare",
        model_names,
        default=model_names[: min(3, len(model_names))],
        help="Select any subset of models. The weighted ensemble is calculated from the selected models only.",
    )
    if not selected_models:
        st.warning("Select at least one model to run the sandbox.")
        return

    with st.form("prediction_sandbox_form"):
        col_a, col_b, col_c = st.columns(3, gap="large")
        with col_a:
            st.selectbox(
                "Title type",
                available_types,
                key=SANDBOX_STATE_KEYS["titleType"],
                help="Different content formats can behave differently in the learned success patterns.",
            )
            st.number_input(
                "Release year",
                min_value=1920,
                max_value=2035,
                key=SANDBOX_STATE_KEYS["startYear"],
                help="The models use release year as a signal for changing audience behavior over time.",
            )
        with col_b:
            st.number_input(
                "Runtime (minutes)",
                min_value=30,
                max_value=300,
                key=SANDBOX_STATE_KEYS["runtimeMinutes"],
                help="Runtime captures differences between shorter formats and longer feature-length content.",
            )
            st.selectbox(
                "Adult content",
                options=[0, 1],
                key=SANDBOX_STATE_KEYS["isAdult"],
                format_func=lambda value: "No" if value == 0 else "Yes",
                help="A binary metadata flag included in the training features.",
            )
        with col_c:
            st.multiselect(
                "Genres",
                available_genres,
                key=SANDBOX_STATE_KEYS["genres"],
                help="Genres are multi-label inputs and are one-hot encoded by the model pipeline.",
            )
        submitted = st.form_submit_button("Run predictions", use_container_width=True)

    if not submitted:
        return

    payload = {
        "titleType": st.session_state[SANDBOX_STATE_KEYS["titleType"]],
        "startYear": int(st.session_state[SANDBOX_STATE_KEYS["startYear"]]),
        "runtimeMinutes": int(st.session_state[SANDBOX_STATE_KEYS["runtimeMinutes"]]),
        "isAdult": int(st.session_state[SANDBOX_STATE_KEYS["isAdult"]]),
        "genres": st.session_state[SANDBOX_STATE_KEYS["genres"]],
    }
    chosen_models = {name: fitted_models[name] for name in selected_models}
    chosen_weights = {name: normalized_weights.get(name, 0.0) for name in selected_models}
    prediction_bundle = predict_success_across_models(payload, chosen_models, chosen_weights)

    ensemble = prediction_bundle["ensemble"]
    ensemble_confidence = _confidence_label(ensemble["success_probability"])
    result_cols = st.columns(3)
    result_cards = [
        ("Ensemble Prediction", ensemble["predicted_label"], "Weighted vote across the selected models"),
        ("Ensemble Probability", f"{ensemble['success_probability']:.2%}", "Estimated likelihood of meeting the current success definition"),
        ("Confidence", ensemble_confidence, "Confidence is based on distance from the 0.50 decision threshold"),
    ]
    for column, (label, value, caption) in zip(result_cols, result_cards):
        with column:
            render_kpi(label, value, caption)

    comparison_rows = []
    for row in prediction_bundle["models"]:
        confidence = _confidence_label(row["success_probability"])
        comparison_rows.append(
            {
                "Model": row["model_name"],
                "Prediction": row["predicted_label"],
                "Probability": row["success_probability"],
                "Confidence": confidence,
                "Normalized Weight (%)": row["weight"] * 100,
            }
        )
    comparison_table = pd.DataFrame(comparison_rows).sort_values(["Probability", "Normalized Weight (%)"], ascending=False)
    st.dataframe(comparison_table.round(4), use_container_width=True, height=260)

    probability_chart = px.bar(
        comparison_table,
        x="Model",
        y="Probability",
        color="Normalized Weight (%)",
        color_continuous_scale="Turbo",
        title="Per-Model Success Probability",
    )
    probability_chart.update_layout(yaxis_tickformat=".0%")
    style_figure(probability_chart, 420, legend=False)
    st.plotly_chart(probability_chart, use_container_width=True)

    render_callout(
        "Output interpretation",
        "A higher probability means the model believes the input is more likely to satisfy the current success definition based on IMDb metadata patterns. The weighted ensemble probability is not an average of labels; it is a normalized weighted blend of model probabilities.",
    )
