
from dany_core.utils.timing import StageTimer

from typing import Dict, Any
import traceback
import pandas as pd

MAX_ROWS = 200_000
MAX_COLS = 500
EDA_SAMPLE_THRESHOLD = 20_000
RANDOM_STATE = 42





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

    results: Dict[str, Any] = {
        "status": "started",
        "validation_passed": False,
    }

    timer = StageTimer()

    try:
        # ======================================================
        # DATASET SAFETY CHECK
        # ======================================================
        if dataframe.shape[0] > MAX_ROWS:
            raise ValueError(f"Dataset exceeds maximum row limit ({MAX_ROWS})")

        if dataframe.shape[1] > MAX_COLS:
            raise ValueError(f"Dataset exceeds maximum column limit ({MAX_COLS})")

        # ======================================================
        # STEP 0 — TARGET VALIDATION
        # ======================================================
        timer.start("target_validation")
        target_validation = validate_target(dataframe, target_spec)
        timer.stop("target_validation")

        results["target_validation"] = target_validation

        if not target_validation.get("valid", False):
            results["status"] = "failed"
            results["reason"] = "Target validation failed"
            results["timing"] = timer.summary()
            return results

        results["validation_passed"] = True

        # ======================================================
        # STEP 1 — CLEANING
        # ======================================================
        timer.start("cleaning")
        cleaned_df, cleaning_report = run_cleaning(dataframe)
        timer.stop("cleaning")

        results["cleaning"] = cleaning_report

        # ======================================================
        # STEP 2 — EDA (with optional sampling)
        # ======================================================
        timer.start("eda")

        if len(cleaned_df) > EDA_SAMPLE_THRESHOLD:
            eda_df = cleaned_df.sample(
                n=EDA_SAMPLE_THRESHOLD,
                random_state=RANDOM_STATE
            )
        else:
            eda_df = cleaned_df

        numerical_profiles = profile_numerical_columns(eda_df, target_spec.name)
        categorical_profiles = profile_categorical_columns(eda_df)
        target_profile = profile_target(eda_df, target_spec.name)

        timer.stop("eda")

        results["profiles"] = {
            "numerical": numerical_profiles,
            "categorical": categorical_profiles,
            "target": target_profile,
        }

        # ======================================================
        # STEP 3 — MODELING
        # ======================================================
        timer.start("modeling")
        modeling_results = train_and_evaluate(
            cleaned_df,
            target_spec.name
        )
        timer.stop("modeling")

        results["modeling"] = modeling_results

        # ======================================================
        # STEP 4 — REPORT GENERATION
        # ======================================================
        timer.start("report_generation")
        report_path = generate_html_report(results)
        timer.stop("report_generation")

        results["report_path"] = report_path

        # ======================================================
        # DONE
        # ======================================================
        results["status"] = "completed"
        results["timing"] = timer.summary()

        return results

    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        results["trace"] = traceback.format_exc()
        results["timing"] = timer.summary()
        return results

