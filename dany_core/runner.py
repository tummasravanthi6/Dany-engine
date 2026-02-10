from typing import Dict, Any
import traceback
import pandas as pd

from dany_core.targets.target_spec import TargetSpec
from dany_core.targets.validators import validate_target

from dany_core.cleaning import run_cleaning
from dany_core.eda import (
    profile_numerical_columns,
    profile_categorical_columns,
    profile_target,
)
from dany_core.modeling import train_and_evaluate  # your modeling.py function
from dany_core.reports.html_report import generate_html_report


def run_dany_pipeline(
    dataframe: pd.DataFrame,
    target_spec: TargetSpec
) -> Dict[str, Any]:
    """
    Full DANY pipeline: validates target, cleans, profiles, models, and generates HTML report.
    """

    results: Dict[str, Any] = {
        "status": "started",
        "validation_passed": False,
    }

    try:
        # ======================================================
        # STEP 0 — TARGET VALIDATION
        # ======================================================
        target_validation = validate_target(dataframe, target_spec)
        results["target_validation"] = target_validation

        if not target_validation.get("valid", False):
            results["status"] = "failed"
            results["reason"] = "Target validation failed"
            return results

        results["validation_passed"] = True

        # ======================================================
        # STEP 1 — CLEANING
        # ======================================================
        cleaned_df, cleaning_report = run_cleaning(dataframe)
        results["cleaning"] = cleaning_report

        # ======================================================
        # STEP 2 — EDA
        # ======================================================
        numerical_profiles = profile_numerical_columns(cleaned_df, target_spec.name)
        categorical_profiles = profile_categorical_columns(cleaned_df)
        target_profile = profile_target(cleaned_df, target_spec.name)

        results["profiles"] = {
            "numerical": numerical_profiles,
            "categorical": categorical_profiles,
            "target": target_profile,
        }

        # ======================================================
        # STEP 3 — MODELING
        # ======================================================
        modeling_results = train_and_evaluate(cleaned_df, target_spec.name)
        results["modeling"] = modeling_results

        # ======================================================
        # STEP 4 — REPORT GENERATION
        # ======================================================
        report_path = generate_html_report(results)
        results["report_path"] = report_path

        # ======================================================
        # DONE
        # ======================================================
        results["status"] = "completed"
        return results

    except Exception as e:
        # 🚫 Never expose raw stack traces to end users
        results["status"] = "error"
        results["error"] = str(e)
        results["trace"] = traceback.format_exc()
        return results
