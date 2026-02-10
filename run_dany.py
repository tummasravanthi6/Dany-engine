import pandas as pd

from dany_core.cleaning import clean_data
from dany_core.report import basic_data_report
from dany_core.insights import generate_cleaning_insights
from dany_core.modeling import train_and_evaluate

from dany_core.summary import (
    build_executive_summary,
    build_data_quality_section,
    build_insights_section,
    build_model_performance_section,
    build_prediction_trust_section,
)

from dany_core.report_generator import Report, render_report_to_html


def run_dany(
    input_csv,
    cleaned_csv,
    log_csv,
    target_col,
    task_type
):
    # =========================================================
    # Load raw data
    # =========================================================
    df = pd.read_csv(input_csv)

    # =========================================================
    # Data quality report
    # =========================================================
    data_quality_report = basic_data_report(df)

    # =========================================================
    # Cleaning
    # =========================================================
    cleaned_df, cleaning_steps = clean_data(df)

    cleaned_df.to_csv(cleaned_csv, index=False)
    pd.DataFrame(cleaning_steps).to_csv(log_csv, index=False)

    # =========================================================
    # Cleaning / EDA insights
    # =========================================================
    eda_insights = generate_cleaning_insights(cleaning_steps)

    # =========================================================
    # Modeling
    # =========================================================
    modeling_results = train_and_evaluate(
        df=cleaned_df,
        target_col=target_col,
        task_type=task_type
    )

    # =========================================================
    # Confidence + trust (deterministic)
    # =========================================================
    if modeling_results is None:
        confidence_summary = {"trust_level": "Low"}
        trust_warnings = ["Modeling was skipped or failed reliability checks."]
    else:
        confidence_summary = {"trust_level": "Medium"}
        trust_warnings = []

    # =========================================================
    # Executive Summary
    # =========================================================
    executive_summary = build_executive_summary(
        data_quality_report=data_quality_report,
        eda_insights=eda_insights,
        modeling_results=modeling_results,
        confidence_summary=confidence_summary,
        trust_warnings=trust_warnings,
    )

    # =========================================================
    # Narrative Sections
    # =========================================================
    data_quality_text = build_data_quality_section(data_quality_report)
    insights_text = build_insights_section(eda_insights)
    modeling_text = build_model_performance_section(modeling_results)
    prediction_trust_text = build_prediction_trust_section(confidence_summary)

    # =========================================================
    # Report Object
    # =========================================================
    report = Report(
        title="Dany Analysis Report",
        executive_summary=executive_summary.__dict__,
        data_overview=data_quality_text,
        cleaning_actions="Cleaning steps were applied deterministically using predefined rules.",
        key_insights=insights_text,
        modeling_results=modeling_text,
        predictions_confidence=prediction_trust_text,
        trust_warnings=(
            "\n".join(trust_warnings)
            if trust_warnings
            else "No critical trust warnings were raised."
        ),
        limitations_assumptions=(
            "Results are constrained by data quality, sample size, "
            "and modeling assumptions."
        ),
    )

    # =========================================================
    # HTML Export
    # =========================================================
    output_path = "outputs/dany_report.html"
    html = render_report_to_html(report)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    # =========================================================
    # Final Output
    # =========================================================
    return {
        "cleaned_df": cleaned_df,
        "cleaning_steps": cleaning_steps,
        "cleaning_insights": eda_insights,
        "modeling": modeling_results,
        "report_path": output_path,
        "status": "completed",
    }


if __name__ == "__main__":
    run_dany(
        input_csv="datasets/messy_data.csv",
        cleaned_csv="outputs/cleaned_data.csv",
        log_csv="outputs/cleaning_log.csv",
        target_col="target",          # change if needed
        task_type="classification"    # or "regression"
    )
