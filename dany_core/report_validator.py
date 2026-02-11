from typing import List


def validate_report_consistency(
    executive_summary: dict,
    data_quality_report: dict,
    modeling_results: dict | None,
    confidence_summary: dict,
) -> List[str]:
    """
    Returns a list of validation errors.
    Empty list means report is internally consistent.
    """

    errors = []

    
    # Data quality consistency
    
    dq_rating = executive_summary.get("data_quality_rating")
    dq_score = data_quality_report.get("overall_score", 0.0)

    if dq_rating == "High" and dq_score < 0.8:
        errors.append("Report claims high data quality, but score is low.")

    if dq_rating == "Low" and dq_score > 0.7:
        errors.append("Report claims low data quality, but score is moderate.")

    
    # Modeling consistency

    modeling_status = executive_summary.get("modeling_status")

    if modeling_results is None and modeling_status != "Skipped":
        errors.append("Modeling was skipped but summary says otherwise.")

    if modeling_results is not None and modeling_status == "Skipped":
        errors.append("Modeling results exist but summary says skipped.")

    
    # Trust consistency

    trust_level = executive_summary.get("prediction_trust_level")
    internal_trust = confidence_summary.get("trust_level")

    if trust_level != internal_trust:
        errors.append("Prediction trust level mismatch.")

    return errors
