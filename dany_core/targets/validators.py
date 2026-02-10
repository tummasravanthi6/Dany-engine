def validate_target(df, target_spec):
    errors = []

    if target_spec.name not in df.columns:
        errors.append(f"Target column '{target_spec.name}' not found")

    null_ratio = df[target_spec.name].isna().mean()
    if null_ratio > target_spec.allowed_null_ratio:
        errors.append(
            f"Target null ratio {null_ratio:.2f} exceeds allowed "
            f"{target_spec.allowed_null_ratio:.2f}"
        )

    if target_spec.task_type not in {"classification", "regression"}:
        errors.append(
            f"Invalid task_type '{target_spec.task_type}'"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "task_type": target_spec.task_type,
    }
