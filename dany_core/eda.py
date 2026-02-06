import pandas as pd
import numpy as np
from scipy.stats import skew


def profile_numerical_columns(df: pd.DataFrame, target_col: str | None = None) -> dict:
    profiles = {}
    num_cols = df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        if col == target_col:
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        profiles[col] = {
            "count": int(series.count()),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)),
            "min": float(series.min()),
            "max": float(series.max()),
            "skewness": float(skew(series)),
            "n_unique": int(series.nunique()),
        }

    return profiles
def profile_categorical_columns(df: pd.DataFrame) -> dict:
    profiles = {}

    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in cat_cols:
        series = df[col]
        value_counts = series.value_counts(dropna=False)
        total = len(series)

        profiles[col] = {
            "n_unique": int(series.nunique(dropna=False)),
            "top_ratio": float(value_counts.iloc[0] / total) if total > 0 else 0.0,
        }

    return profiles
def profile_target(df: pd.DataFrame, target_col: str) -> dict:
    series = df[target_col].dropna()

    # Classification detection
    if series.dtype == "object" or series.nunique() <= 20:
        counts = series.value_counts(normalize=True)

        return {
            "task_type": "classification",
            "class_distribution": counts.to_dict(),
            "min_class_ratio": float(counts.min()),
        }

    # Regression fallback
    return {
        "task_type": "regression",
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)),
        "skewness": float(skew(series)),
    }

