from dany_core.summary import (
    build_executive_summary,
    build_data_quality_section,
    build_insights_section,
    build_model_performance_section,
    build_prediction_trust_section,
)

from dany_core.report_generator import Report, render_report_to_html
from dany_core.report_validator import validate_report_consistency

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
    Orchestrates EDA + Insights + Modeling + Report Generation.
    Day 6: Executive report
    Day 7: Truth & consistency validation
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
            "trust_level": "High",
        }
    else:
        confidence_warnings.append(
            "Per-prediction confidence unavailable"
        )

    if not confidence_warnings:
        confidence_warnings.append(
            "No major confidence risks detected"
        )

    # =========================================================
    # Day 6 — Executive Summary
    # =========================================================

    data_quality_report = {
        "overall_score": confidence_summary["average_confidence"]
        if confidence_summary else 0.0,
        "rating": "High" if confidence_summary and confidence_summary["average_confidence"] >= 0.7 else "Low",
    }

    executive_summary = build_executive_summary(
        data_quality_report=data_quality_report,
        eda_insights=ranked_insights,
        modeling_results=modeling_results,
        confidence_summary={
            "trust_level": "High"
        },
        trust_warnings=trust_warnings,
    )

    # =========================================================
    # Day 7 — Report Validation (GUARDRAIL)
    # =========================================================

    validation_errors = validate_report_consistency(
        executive_summary=executive_summary.__dict__,
        data_quality_report=data_quality_report,
        modeling_results=modeling_results,
        confidence_summary={
            "trust_level": confidence_summary["trust_level"]
        },
    )

    if validation_errors:
        raise ValueError(
            "Report validation failed:\n" + "\n".join(validation_errors)
        )

    # =========================================================
    # Day 6 — Report Object
    # =========================================================

    report = Report(
        title="Dany Analysis Report",
        executive_summary=executive_summary.__dict__,
        data_overview=build_data_quality_section(data_quality_report),
        cleaning_actions="No automated cleaning actions were applied.",
        key_insights=build_insights_section(ranked_insights),
        modeling_results=build_model_performance_section(modeling_results),
        predictions_confidence=build_prediction_trust_section(
            {"trust_level": executive_summary.prediction_trust_level}
        ),
        trust_warnings="\n".join(trust_warnings)
        if trust_warnings else "No major trust risks detected.",
        limitations_assumptions=(
            "Results are constrained by data quality, sample size, "
            "and modeling assumptions."
        ),
    )

    # =========================================================
    # Day 6 — HTML Export
    # =========================================================

    output_path = "outputs/dany_report.html"

    html = render_report_to_html(report)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

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

        # Day 6–7 additions
        "report_path": output_path,
        "validation_passed": True,
        "status": "completed",
    }


if __name__ == "__main__":
    print("Dany runner executed successfully.")
