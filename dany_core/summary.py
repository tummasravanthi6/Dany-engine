from dataclasses import dataclass
from typing import List, Optional


# =========================================================
# Executive Summary (STRUCTURED — NO STORYTELLING)
# =========================================================

@dataclass
class ExecutiveSummary:
    data_quality_rating: str            # High / Medium / Low
    modeling_status: str                # Succeeded / Skipped / Weak
    best_model_name: Optional[str]
    prediction_trust_level: str         # High / Medium / Low / Do Not Trust
    key_takeaways: List[str]            # ranked, max 5


def build_executive_summary(
    data_quality_report: dict,
    eda_insights: list,
    modeling_results: dict | None,
    confidence_summary: dict,
    trust_warnings: list
) -> ExecutiveSummary:
    """
    Builds a structured executive summary from internal system outputs.
    No natural language paragraphs here — only conclusions.
    """

    # -----------------------------
    # Data quality rating
    # -----------------------------
    dq_score = data_quality_report.get("overall_score", 0.0)

    if dq_score >= 0.85:
        data_quality = "High"
    elif dq_score >= 0.65:
        data_quality = "Medium"
    else:
        data_quality = "Low"

    # -----------------------------
    # Modeling status
    # -----------------------------
    if modeling_results is None:
        modeling_status = "Skipped"
        best_model = None
    else:
        best_score = modeling_results.get("best_score", 0.0)
        best_model = modeling_results.get("best_model")

        if best_score >= 0.55:
            modeling_status = "Succeeded"
        else:
            modeling_status = "Weak"

    # -----------------------------
    # Prediction trust
    # -----------------------------
    prediction_trust = confidence_summary.get("trust_level", "Do Not Trust")

    # -----------------------------
    # Key takeaways (ranked, factual)
    # -----------------------------
    takeaways: List[str] = []

    if data_quality != "High":
        takeaways.append(
            "Data quality limitations may affect reliability of results."
        )

    for insight in eda_insights[:3]:
        if "summary" in insight:
            takeaways.append(insight["summary"])

    if modeling_status != "Succeeded":
        takeaways.append(
            "Model outputs should not be relied upon for decision-making."
        )

    if trust_warnings:
        takeaways.append(
            "Trust warnings were raised and should be reviewed carefully."
        )

    return ExecutiveSummary(
        data_quality_rating=data_quality,
        modeling_status=modeling_status,
        best_model_name=best_model,
        prediction_trust_level=prediction_trust,
        key_takeaways=takeaways[:5]
    )


# =========================================================
# Section-wise Narrative Builders (CLEAN PARAGRAPHS)
# =========================================================

def build_data_quality_section(data_quality_report: dict) -> str:
    return (
        f"The dataset contains {data_quality_report.get('row_count', 'N/A')} records "
        f"and {data_quality_report.get('column_count', 'N/A')} variables. "
        f"Overall data quality was rated as "
        f"{data_quality_report.get('rating', 'Unknown')}. "
        f"Common issues included missing values and inconsistent entries."
    )


def build_insights_section(eda_insights: list) -> str:
    if not eda_insights:
        return "No statistically meaningful insights were identified."

    lines = []
    for insight in eda_insights:
        if "summary" in insight:
            lines.append(f"- {insight['summary']}")

    return "\n".join(lines)


def build_model_performance_section(modeling_results: dict | None) -> str:
    if modeling_results is None:
        return (
            "Modeling was skipped due to insufficient data quality or "
            "failure to meet minimum reliability thresholds."
        )

    return (
        f"The strongest model evaluated was "
        f"{modeling_results.get('best_model', 'Unknown')}. "
        f"It achieved a performance score of "
        f"{modeling_results.get('best_score', 0.0):.2f}, "
        f"which limits confidence in its outputs."
    )


def build_prediction_trust_section(confidence_summary: dict) -> str:
    return (
        f"Overall prediction trust was assessed as "
        f"{confidence_summary.get('trust_level', 'Unknown')}. "
        f"This assessment reflects data quality, model stability, "
        f"and consistency across validation checks."
    )
