# src/analysis/validators.py

import pandas as pd

def validate_target(df: pd.DataFrame, target_spec):
    errors = []

    if target_spec.name not in df.columns:
        errors.append(f"Target column '{target_spec.name}' not found")

    else:
        null_ratio = df[target_spec.name].isna().mean()

        if null_ratio > target_spec.allowed_null_ratio:
            errors.append(
                f"Null ratio {null_ratio:.2f} exceeds allowed {target_spec.allowed_null_ratio}"
            )

        unique_values = df[target_spec.name].nunique(dropna=True)

        if target_spec.task_type == "classification" and unique_values < 2:
            errors.append("Classification target must have at least 2 classes")

        if target_spec.task_type == "regression" and unique_values < 5:
            errors.append("Regression target has too few unique values")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors
    }
