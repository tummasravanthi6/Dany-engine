import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# ======================================================
# PUBLIC API
# ======================================================

def train_and_evaluate(df: pd.DataFrame, target_col: str):
    """
    Trains baseline models and returns structured, inspectable results.
    Day 5: also persists the best trained pipeline for predictions.
    """

    X = df.drop(columns=[target_col])
    y = df[target_col]

    task_type = _detect_task_type(y)
    preprocessor = _build_preprocessor(X)

    X_train, X_test, y_train, y_test = _split_data(
        X, y, task_type
    )

    models = _get_models(task_type)

    all_results = []

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        warnings = []
        metrics = {}

        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            metrics = _compute_metrics(
                task_type, y_test, y_pred
            )

            trained_pipeline = pipeline

        except ValueError as e:
            warnings.append(str(e))
            trained_pipeline = None

        all_results.append(
            {
                "model_name": model_name,
                "metrics": metrics,
                "warnings": warnings,
                "is_best": False,
                "pipeline": trained_pipeline,  # 👈 persisted
            }
        )

    best_model = _select_best_model(
        all_results, task_type
    )

    best_pipeline = None
    for r in all_results:
        if r["model_name"] == best_model.get("model_name"):
            r["is_best"] = True
            best_pipeline = r.get("pipeline")

    return {
        "task_type": task_type,
        "all_models_results": all_results,
        "best_model_summary": best_model,
        "best_pipeline": best_pipeline,  # 👈 REQUIRED FOR DAY 5
    }


# ======================================================
# PREDICTIONS (DAY 5)
# ======================================================

def generate_predictions(modeling_results, df):
    """
    Generate predictions using the best trained pipeline.
    Called ONLY if modeling succeeded.
    """

    pipeline = modeling_results.get("best_pipeline")
    task_type = modeling_results.get("task_type")

    if pipeline is None:
        return None

    X = df.copy()

    if task_type == "classification":
        preds = pipeline.predict(X)

        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X)
        else:
            probs = None

        return {
            "predictions": preds.tolist(),
            "probabilities": probs.tolist() if probs is not None else None,
        }

    # Regression
    preds = pipeline.predict(X)
    return {
        "predictions": preds.tolist(),
        "probabilities": None,
    }


def compute_prediction_confidence(prediction_output, task_type):
    """
    Compute per-prediction confidence.
    Returned separately from prediction values.
    """

    if prediction_output is None:
        return None

    if task_type == "classification":
        probs = prediction_output.get("probabilities")

        if probs is None:
            return None

        # Confidence = max class probability
        return [max(p) for p in probs]

    # Regression confidence intentionally undefined for now
    return None


# ======================================================
# HELPERS
# ======================================================

def _detect_task_type(y: pd.Series) -> str:
    if y.dtype == "object" or y.nunique() <= 2:
        return "classification"
    return "regression"


def _build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    cat_cols = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    transformers = []

    if num_cols:
        transformers.append(
            (
                "num",
                StandardScaler(),
                num_cols,
            )
        )

    if cat_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                cat_cols,
            )
        )

    return ColumnTransformer(transformers=transformers)


def _split_data(X, y, task_type):
    stratify = None

    if task_type == "classification":
        class_counts = y.value_counts()
        if (class_counts >= 2).all():
            stratify = y

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )


def _get_models(task_type):
    if task_type == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(random_state=42),
        }

    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(random_state=42),
    }


def _compute_metrics(task_type, y_true, y_pred):
    if task_type == "classification":
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "rmse": rmse,
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def _select_best_model(results, task_type):
    valid_results = [
        r for r in results if r["metrics"]
    ]

    if not valid_results:
        return {
            "model_name": None,
            "metrics": {},
            "reason": "No model could be trained on the provided data",
        }

    key = "f1" if task_type == "classification" else "rmse"

    sorted_results = sorted(
        valid_results,
        key=lambda x: (
            -x["metrics"][key]
            if task_type == "classification"
            else x["metrics"][key]
        ),
    )

    return {
        "model_name": sorted_results[0]["model_name"],
        "metrics": sorted_results[0]["metrics"],
    }
