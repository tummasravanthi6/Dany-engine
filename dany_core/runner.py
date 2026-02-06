from dany_core.eda import (
    profile_numerical_columns,
    profile_categorical_columns,
    profile_target,
)

from dany_core.insights import (
    generate_insights,
    prioritize_insights,
    evaluate_trust_risks,
)

from dany_core.modeling import (
    train_and_evaluate,
    generate_predictions,
    compute_prediction_confidence,
)


def run_dany(df, target_col: str):
    """
    Orchestrates EDA + Insights + Modeling.
    Day 5: includes prediction + confidence with trust gating.
    """

    # ------------------
    # EDA
    # ------------------
    numerical_profiles = profile_numerical_columns(df, target_col)
    categorical_profiles = profile_categorical_columns(df)
    target_profile = profile_target(df, target_col)

    # ------------------
    # Insights (descriptive only)
    # ------------------
    insights = generate_insights(
        numerical_profiles,
        categorical_profiles,
        target_profile,
    )
    ranked_insights = prioritize_insights(insights)

    # ------------------
    # Modeling (Day 4)
    # ------------------
    modeling_results = train_and_evaluate(
        df=df,
        target_col=target_col,
    )

    trust_warnings = evaluate_trust_risks(modeling_results)

    # ------------------
    # Modeling gate (Day 5)
    # ------------------
    best_model_summary = modeling_results.get("best_model_summary", {})
    metrics = best_model_summary.get("metrics", {})

    modeling_succeeded = metrics.get("accuracy", 0) > 0

    if not modeling_succeeded:
        return {
            "profiles": {
                "numerical": numerical_profiles,
                "categorical": categorical_profiles,
                "target": target_profile,
            },
            "insights": ranked_insights,
            "modeling": modeling_results,
            "predictions": None,
            "prediction_confidence": None,
            "confidence_summary": None,
            "trust_warnings": trust_warnings,
            "confidence_warnings": [
                "Predictions skipped: model performance is insufficient"
            ],
        }

    # ------------------
    # Predictions (Day 5)
    # ------------------
    prediction_output = generate_predictions(
        modeling_results, df
    )

    prediction_confidence = compute_prediction_confidence(
        prediction_output,
        modeling_results.get("task_type"),
    )

    # ------------------
    # Dataset-level confidence summary
    # ------------------
    confidence_summary = None
    confidence_warnings = []

    if prediction_confidence:
        avg_conf = sum(prediction_confidence) / len(prediction_confidence)

        confidence_summary = {
            "average_confidence": avg_conf,
            "low_confidence_pct": sum(
                c < 0.6 for c in prediction_confidence
            ) / len(prediction_confidence),
            "confidence_buckets": {
                "high": sum(c >= 0.8 for c in prediction_confidence),
                "medium": sum(0.6 <= c < 0.8 for c in prediction_confidence),
                "low": sum(c < 0.6 for c in prediction_confidence),
            },
        }
    else:
        confidence_warnings.append(
            "Per-prediction confidence unavailable"
        )

    if not confidence_warnings:
        confidence_warnings.append(
            "No major confidence risks detected"
        )

    # ------------------
    # Final structured output
    # ------------------
    return {
        "profiles": {
            "numerical": numerical_profiles,
            "categorical": categorical_profiles,
            "target": target_profile,
        },
        "insights": ranked_insights,
        "modeling": modeling_results,
        "predictions": prediction_output,
        "prediction_confidence": prediction_confidence,
        "confidence_summary": confidence_summary,
        "trust_warnings": trust_warnings or ["No major trust risks detected"],
        "confidence_warnings": confidence_warnings,
    }
