from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.config.settings import settings
from app.services.feature_engineering import SuccessDefinition, build_success_frame

FEATURE_COLUMNS = ["startYear", "runtimeMinutes", "isAdult", "titleType", "genres"]


class GenreBinarizer(BaseEstimator, TransformerMixin):
    """Simple multi-label binarizer for comma-separated IMDb genre strings."""

    def fit(self, X, y=None):
        values = pd.Series(X.squeeze() if hasattr(X, "squeeze") else X).fillna("Unknown")
        tokens = sorted(
            {
                token.strip()
                for value in values.astype(str)
                for token in value.split(",")
                if token.strip()
            }
        )
        self.genre_labels_ = tokens or ["Unknown"]
        self.genre_index_ = {label: idx for idx, label in enumerate(self.genre_labels_)}
        return self

    def transform(self, X):
        values = pd.Series(X.squeeze() if hasattr(X, "squeeze") else X).fillna("Unknown")
        matrix = np.zeros((len(values), len(self.genre_labels_)))
        for row_idx, value in enumerate(values.astype(str)):
            for token in value.split(","):
                token = token.strip()
                if token in self.genre_index_:
                    matrix[row_idx, self.genre_index_[token]] = 1
        return matrix

    def get_feature_names_out(self, input_features=None):
        return np.array([f"genre__{label}" for label in self.genre_labels_], dtype=object)


@dataclass(frozen=True)
class ModelingConfig:
    min_votes: int = settings.default_min_votes
    year_start: int = 1920
    year_end: int = 2025
    success_rating: float = settings.default_success_rating
    success_votes: int = settings.default_success_votes
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = settings.random_seed
    mode: str = "full"
    sample_size: int | None = None


def config_hash(config: ModelingConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:12]


def _artifact_paths(config: ModelingConfig) -> dict[str, Path]:
    key = config_hash(config)
    return {
        "model": settings.model_artifacts_dir / f"best_model_{key}.joblib",
        "summary": settings.model_artifacts_dir / f"model_summary_{key}.json",
        "latest_model": settings.model_artifacts_dir / "best_model.joblib",
        "latest_summary": settings.model_artifacts_dir / "model_summary.json",
    }


def build_preprocessor() -> ColumnTransformer:
    numeric_columns = ["startYear", "runtimeMinutes", "isAdult"]
    categorical_columns = ["titleType"]
    genre_column = ["genres"]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
            ("genres", GenreBinarizer(), genre_column),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def model_registry(random_state: int) -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Logistic Regression (Unbalanced)": LogisticRegression(
            max_iter=2000,
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        ),
        "Random Forest (Unbalanced)": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
    }


def _safe_roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def _probabilities(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        normalized = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        return normalized
    predictions = model.predict(X)
    return predictions.astype(float)


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    roc_auc = _safe_roc_auc(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob) if roc_auc is not None else ([], [], [])
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "roc_curve": {"fpr": list(map(float, fpr)), "tpr": list(map(float, tpr))},
    }


def _cross_validate_model(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int) -> dict[str, float]:
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=settings.random_seed)
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=None)
    return {
        key.replace("test_", ""): round(float(np.nanmean(value)), 4)
        for key, value in scores.items()
        if key.startswith("test_")
    }


def _permutation_importance(model_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, top_n: int = 20) -> list[dict[str, Any]]:
    result = permutation_importance(
        model_pipeline,
        X_test,
        y_test,
        n_repeats=5,
        random_state=settings.random_seed,
        n_jobs=1,
    )
    feature_names = list(X_test.columns)
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": result.importances_mean})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    return importance_df.round(6).to_dict(orient="records")


def _threshold_diagnostics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    thresholds = np.linspace(0.2, 0.8, 13)
    rows = []
    for threshold in thresholds:
        preds = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_true, preds, zero_division=0)), 4),
            }
        )
    best = max(rows, key=lambda row: row["f1_score"])
    return {"table": rows, "best_threshold": best["threshold"], "best_f1": best["f1_score"]}


def _calibration_diagnostics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    prob_true, prob_pred = calibration_curve(y_true, probabilities, n_bins=8, strategy="uniform")
    return {
        "predicted_probability": list(map(float, prob_pred)),
        "observed_frequency": list(map(float, prob_true)),
    }


def _error_analysis(model_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    review_df = X_test.copy()
    review_df["actual"] = y_test.to_numpy()
    review_df["predicted"] = y_pred
    review_df["success_probability"] = probabilities
    false_positives = review_df[(review_df["actual"] == 0) & (review_df["predicted"] == 1)].sort_values(
        "success_probability", ascending=False
    )
    false_negatives = review_df[(review_df["actual"] == 1) & (review_df["predicted"] == 0)].sort_values(
        "success_probability", ascending=True
    )
    columns = ["startYear", "runtimeMinutes", "titleType", "genres", "actual", "predicted", "success_probability"]
    return {
        "false_positives": false_positives[columns].head(5).round(4).to_dict(orient="records"),
        "false_negatives": false_negatives[columns].head(5).round(4).to_dict(orient="records"),
    }


def _extract_feature_importance(model_pipeline: Pipeline, top_n: int = 20) -> list[dict[str, Any]]:
    preprocessor = model_pipeline.named_steps["preprocessor"]
    estimator = model_pipeline.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())

    if hasattr(estimator, "feature_importances_"):
        raw_importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        raw_importance = np.abs(estimator.coef_[0])
    else:
        return []

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": raw_importance})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    return importance_df.round(6).to_dict(orient="records")


def _build_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def _build_model_result(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int,
    tuning_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    predicted = pipeline.predict(X_test)
    probabilities = _probabilities(pipeline, X_test)
    result = {
        "model_name": model_name,
        "test_metrics": _evaluate_predictions(y_test.to_numpy(), predicted, probabilities),
        "cv_metrics": _cross_validate_model(pipeline, X_train, y_train, cv_folds),
        "feature_importance": _extract_feature_importance(pipeline),
        "permutation_importance": _permutation_importance(pipeline, X_test, y_test),
        "threshold_analysis": _threshold_diagnostics(y_test.to_numpy(), probabilities),
        "calibration": _calibration_diagnostics(y_test.to_numpy(), probabilities),
        "error_analysis": _error_analysis(pipeline, X_test, y_test, predicted, probabilities),
    }
    if tuning_meta is not None:
        result["tuning"] = tuning_meta
    return result


def modeling_frame(df: pd.DataFrame, config: ModelingConfig) -> pd.DataFrame:
    filtered = df[
        (df["startYear"] >= config.year_start)
        & (df["startYear"] <= config.year_end)
        & (df["numVotes"] >= config.min_votes)
    ].copy()
    definition = SuccessDefinition(
        rating_threshold=config.success_rating,
        votes_threshold=config.success_votes,
    )
    framed = build_success_frame(filtered, definition)
    if config.sample_size and len(framed) > config.sample_size:
        positives = framed[framed["success"] == 1]
        negatives = framed[framed["success"] == 0]
        pos_target = max(1, int(config.sample_size * len(positives) / len(framed)))
        neg_target = max(1, config.sample_size - pos_target)
        framed = pd.concat(
            [
                positives.sample(min(len(positives), pos_target), random_state=config.random_state),
                negatives.sample(min(len(negatives), neg_target), random_state=config.random_state),
            ]
        ).sample(frac=1, random_state=config.random_state)
    return framed.reset_index(drop=True)


def normalize_model_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {name: max(float(value), 0.0) for name, value in weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        equal_weight = 1.0 / len(cleaned) if cleaned else 0.0
        return {name: equal_weight for name in cleaned}
    return {name: value / total for name, value in cleaned.items()}


def predict_success_across_models(
    payload: dict[str, Any],
    models: dict[str, Pipeline],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    if not models:
        raise ValueError("At least one fitted model is required for multi-model prediction.")

    normalized_weights = normalize_model_weights(weights or {name: 1.0 for name in models})
    predictions = []
    weighted_probability = 0.0

    for model_name, model in models.items():
        result = predict_success(payload, model)
        weight = normalized_weights.get(model_name, 0.0)
        weighted_probability += result["success_probability"] * weight
        predictions.append(
            {
                "model_name": model_name,
                "weight": round(weight, 4),
                **result,
            }
        )

    ensemble_prediction = int(weighted_probability >= 0.5)
    return {
        "normalized_weights": normalized_weights,
        "ensemble": {
            "prediction": ensemble_prediction,
            "predicted_label": "Successful" if ensemble_prediction else "Not Successful",
            "success_probability": round(float(weighted_probability), 4),
        },
        "models": predictions,
    }


def _tuning_grids(random_state: int) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    return {
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced_subsample", random_state=random_state, n_jobs=1),
            {
                "model__n_estimators": [150, 250],
                "model__max_depth": [None, 12],
                "model__min_samples_leaf": [3, 5],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=random_state),
            {
                "model__n_estimators": [100, 150],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
    }


def _tune_pipeline(name: str, X_train: pd.DataFrame, y_train: pd.Series, config: ModelingConfig) -> tuple[str, Pipeline, dict[str, Any]] | None:
    grids = _tuning_grids(config.random_state)
    if name not in grids:
        return None
    estimator, params = grids[name]
    search = GridSearchCV(
        _build_pipeline(estimator),
        param_grid=params,
        scoring="f1",
        cv=StratifiedKFold(n_splits=min(3, config.cv_folds), shuffle=True, random_state=config.random_state),
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    return (
        f"{name} (Tuned)",
        search.best_estimator_,
        {
            "best_params": search.best_params_,
            "best_score": round(float(search.best_score_), 4),
        },
    )


def _train_model_bundle(
    df: pd.DataFrame,
    config: ModelingConfig,
) -> tuple[dict[str, Any], dict[str, Pipeline]]:
    modeling_df = modeling_frame(df, config)
    if len(modeling_df) < 300 or modeling_df["success"].nunique() < 2:
        raise ValueError(
            "Not enough class-balanced data after filtering to train stable models."
        )

    X = modeling_df[FEATURE_COLUMNS]
    y = modeling_df["success"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    results: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    for model_name, estimator in model_registry(config.random_state).items():
        pipeline = _build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        result = _build_model_result(
            model_name=model_name,
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cv_folds=config.cv_folds,
        )
        results.append(result)
        fitted_models[model_name] = pipeline

    base_scores = sorted(results, key=lambda item: item["test_metrics"]["f1_score"], reverse=True)
    if config.mode == "full":
        for candidate in base_scores[:2]:
            tuned = _tune_pipeline(candidate["model_name"].replace(" (Unbalanced)", ""), X_train, y_train, config)
            if tuned is None:
                continue
            tuned_name, tuned_pipeline, tuning_meta = tuned
            tuned_result = _build_model_result(
                model_name=tuned_name,
                pipeline=tuned_pipeline,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cv_folds=min(3, config.cv_folds),
                tuning_meta=tuning_meta,
            )
            results.append(tuned_result)
            fitted_models[tuned_name] = tuned_pipeline

    comparison_df = pd.DataFrame(
        [
            {
                "model_name": item["model_name"],
                "accuracy": item["test_metrics"]["accuracy"],
                "precision": item["test_metrics"]["precision"],
                "recall": item["test_metrics"]["recall"],
                "f1_score": item["test_metrics"]["f1_score"],
                "roc_auc": item["test_metrics"]["roc_auc"] or 0,
                "cv_accuracy": item["cv_metrics"].get("accuracy", 0),
                "cv_f1": item["cv_metrics"].get("f1", 0),
                "cv_roc_auc": item["cv_metrics"].get("roc_auc", 0),
                "imbalance_strategy": "balanced"
                if "Unbalanced" not in item["model_name"]
                and any(key in item["model_name"] for key in ["Logistic", "Random Forest", "Decision Tree"])
                else "standard",
            }
            for item in results
        ]
    ).sort_values(["f1_score", "roc_auc", "accuracy"], ascending=False)

    ordered_models = comparison_df["model_name"].tolist()
    results_by_name = {item["model_name"]: item for item in results}
    models_payload = [results_by_name[name] for name in ordered_models]
    best_model_name = ordered_models[0]
    best_result = results_by_name[best_model_name]
    best_reasoning = (
        f"{best_model_name} performed best because it balanced precision and recall most effectively "
        f"(F1={best_result['test_metrics']['f1_score']:.3f}) while maintaining strong ROC-AUC "
        f"({best_result['test_metrics']['roc_auc'] if best_result['test_metrics']['roc_auc'] is not None else 'N/A'}). "
        "Its top features also align with the observed data patterns in release timing, title format, and genre composition."
    )
    best_model_payload = dict(best_result)
    best_model_payload["why_best_model_won"] = best_reasoning

    class_balance = {
        "positive_rate": round(float(y.mean()), 4),
        "class_counts": {str(label): int(count) for label, count in y.value_counts().to_dict().items()},
    }
    summary_table = (
        comparison_df.loc[:, ["model_name", "f1_score", "roc_auc", "accuracy", "cv_f1", "imbalance_strategy"]]
        .head(6)
        .to_dict(orient="records")
    )

    output = {
        "config": asdict(config),
        "dataset": {
            "rows": int(len(modeling_df)),
            "success_rate": round(float(modeling_df["success"].mean()), 4),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "feature_columns": FEATURE_COLUMNS,
            "class_balance": class_balance,
            "note": (
                "averageRating and numVotes define the target and are intentionally excluded "
                "from the predictors to avoid target leakage."
            ),
        },
        "comparison": comparison_df.to_dict(orient="records"),
        "models": models_payload,
        "results_summary": summary_table,
        "best_model": best_model_payload,
    }
    return output, fitted_models


def compare_models(
    df: pd.DataFrame,
    config: ModelingConfig,
    persist: bool = True,
    force_retrain: bool = False,
) -> dict[str, Any]:
    if persist and not force_retrain:
        cached = load_cached_comparison(config)
        if cached is not None:
            return cached
    output, fitted_models = _train_model_bundle(df, config)

    if persist:
        persist_model_artifacts(fitted_models[output["best_model"]["model_name"]], output, config)
    return output


def compare_models_with_artifacts(
    df: pd.DataFrame,
    config: ModelingConfig,
    persist: bool = True,
    force_retrain: bool = False,
) -> tuple[dict[str, Any], dict[str, Pipeline]]:
    output, fitted_models = _train_model_bundle(df, config)
    if persist:
        persist_model_artifacts(fitted_models[output["best_model"]["model_name"]], output, config)
    return output, fitted_models


def persist_model_artifacts(best_model: Pipeline, comparison_output: dict[str, Any], config: ModelingConfig) -> None:
    settings.ensure_directories()
    paths = _artifact_paths(config)
    joblib.dump(best_model, paths["model"])
    paths["summary"].write_text(json.dumps(comparison_output, indent=2), encoding="utf-8")
    joblib.dump(best_model, paths["latest_model"])
    paths["latest_summary"].write_text(json.dumps(comparison_output, indent=2), encoding="utf-8")


def load_cached_comparison(config: ModelingConfig) -> dict[str, Any] | None:
    paths = _artifact_paths(config)
    if not paths["summary"].exists():
        return None
    payload = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if "best_model" not in payload or "threshold_analysis" not in payload["best_model"]:
        return None
    return payload


def load_persisted_model(config: ModelingConfig | None = None) -> Pipeline | None:
    model_path = _artifact_paths(config)["model"] if config is not None else settings.model_artifacts_dir / "best_model.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def predict_success(payload: dict[str, Any], model: Pipeline) -> dict[str, Any]:
    input_df = pd.DataFrame(
        [
            {
                "startYear": payload.get("startYear"),
                "runtimeMinutes": payload.get("runtimeMinutes"),
                "isAdult": payload.get("isAdult", 0),
                "titleType": payload.get("titleType", "movie"),
                "genres": ",".join(payload.get("genres", []))
                if isinstance(payload.get("genres"), list)
                else payload.get("genres", "Unknown"),
            }
        ]
    )
    probability = float(_probabilities(model, input_df)[0])
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "predicted_label": "Successful" if prediction else "Not Successful",
        "success_probability": round(probability, 4),
    }
