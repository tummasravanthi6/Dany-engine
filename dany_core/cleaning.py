"""
DANY DATA CLEANING CONTRACT
Allowed: drop duplicates, handle missing, fix dtype, cap outliers
Forbidden: drop rows silently, feature engineering, domain assumptions
"""

import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame):
    cleaned_df = df.copy()
    cleaning_steps = []

    cleaned_df, steps = _handle_duplicates(cleaned_df)
    cleaning_steps.extend(steps)

    cleaned_df, steps = _fix_dtypes(cleaned_df)
    cleaning_steps.extend(steps)

    cleaned_df, steps = _handle_missing(cleaned_df)
    cleaning_steps.extend(steps)

    cleaned_df, steps = _cap_outliers(cleaned_df)
    cleaning_steps.extend(steps)

    return cleaned_df, cleaning_steps


def _handle_duplicates(df: pd.DataFrame):
    steps = []
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    dropped = before - after
    if dropped > 0:
        steps.append({"action": "dropped_duplicates", "count": dropped, "reason": "exact row duplicates"})
    return df, steps


def _fix_dtypes(df: pd.DataFrame):
    steps = []
    for col in df.columns:
        if df[col].dtype == object:
            coerced = pd.to_numeric(df[col], errors="coerce")
            valid_ratio = coerced.notna().mean()
            if valid_ratio >= 0.9:
                df[col] = coerced
                steps.append({"action": "converted_dtype", "column": col, "from": "object", "to": "numeric", "reason": "numeric values stored as strings"})
    return df, steps


def _handle_missing(df: pd.DataFrame):
    steps = []
    for col in df.columns:
        missing_ratio = df[col].isna().mean()
        if missing_ratio == 0:
            continue
        if missing_ratio > 0.4:
            steps.append({"action": "flag_high_missingness", "column": col, "ratio": float(missing_ratio), "reason": "missing values exceed 40%"})
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].median()
            method = "median"
        else:
            fill_value = df[col].mode().iloc[0]
            method = "mode"
        df[col] = df[col].fillna(fill_value)
        steps.append({"action": "filled_missing_values", "column": col, "method": method, "reason": "deterministic missing value handling"})
    return df, steps


def _cap_outliers(df: pd.DataFrame):
    steps = []
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        has_outliers = ((df[col] < lower) | (df[col] > upper)).any()
        if has_outliers:
            df[col] = df[col].clip(lower, upper)
            steps.append({"action": "capped_outliers", "column": col, "method": "IQR", "reason": "extreme values detected"})
    return df, steps



def run_cleaning(df):
    """
    Pipeline entrypoint for cleaning stage.
    Must return cleaned_df and cleaning_steps.
    """
    return clean_data(df)
