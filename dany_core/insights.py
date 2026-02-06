SEVERITY_SCORE = {
    "info": 1,
    "warning": 2,
    "critical": 3,
}


def generate_insights(
    numerical_profiles: dict,
    categorical_profiles: dict,
    target_profile: dict,
) -> list[dict]:
    insights = []

    # Numerical rules
    for col, stats in numerical_profiles.items():
        if stats["std"] == 0:
            insights.append({
                "severity": "critical",
                "message": "Zero variance numerical column",
                "columns": [col],
                "impact": 1.0,
            })
        elif abs(stats["skewness"]) > 2:
            insights.append({
                "severity": "warning",
                "message": "Highly skewed numerical distribution",
                "columns": [col],
                "impact": 1.0,
            })

    # Categorical rules
    for col, stats in categorical_profiles.items():
        if stats["top_ratio"] > 0.95:
            insights.append({
                "severity": "warning",
                "message": "Dominant categorical value",
                "columns": [col],
                "impact": stats["top_ratio"],
            })

    # Target rules
    if target_profile["task_type"] == "classification":
        if target_profile["min_class_ratio"] < 0.1:
            insights.append({
                "severity": "critical",
                "message": "Severe class imbalance in target",
                "columns": ["target"],
                "impact": target_profile["min_class_ratio"],
            })

    return insights
def prioritize_insights(insights: list[dict]) -> list[dict]:
    def score(insight: dict) -> float:
        return (
            SEVERITY_SCORE.get(insight["severity"], 0) * 10
            + insight.get("impact", 0)
        )

    return sorted(insights, key=score, reverse=True)

def evaluate_trust_risks(modeling_results):
    """
    Day 5: Rule-based trust failure detection.
    Returns explicit reasons why predictions should not be trusted.
    """

    warnings = []

    best = modeling_results.get("best_model_summary", {})
    metrics = best.get("metrics", {})
    model_warnings = best.get("warnings", [])

    # Rule 1: No usable metrics
    if not metrics:
        warnings.append({
            "severity": "high",
            "message": "No valid model metrics available",
            "evidence": best
        })

    # Rule 2: Zero performance
    if metrics.get("accuracy", 0) == 0:
        warnings.append({
            "severity": "high",
            "message": "Model accuracy is zero; predictions are unreliable",
            "evidence": metrics
        })

    # Rule 3: Single-class dataset detected
    if any("only one class" in w.lower() for w in model_warnings):
        warnings.append({
            "severity": "high",
            "message": "Training data contains only one target class",
            "evidence": model_warnings
        })

    # Rule 4: Missing best model summary
    if not best:
        warnings.append({
            "severity": "high",
            "message": "No best model selected during modeling",
            "evidence": modeling_results
        })

    return warnings

