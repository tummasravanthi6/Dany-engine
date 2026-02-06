import pandas as pd

from dany_core.cleaning import clean_data
from dany_core.report import basic_data_report
from dany_core.insights import generate_cleaning_insights
from dany_core.modeling import train_and_evaluate


def run_dany(
    input_csv,
    cleaned_csv,
    log_csv,
    target_col,
    task_type
):
    # -------------------------------
    # Load raw data
    # -------------------------------
    df = pd.read_csv(input_csv)

    # -------------------------------
    # Basic data report
    # -------------------------------
    report = basic_data_report(df)
    print("Data Report:", report)

    # -------------------------------
    # Cleaning
    # -------------------------------
    cleaned_df, cleaning_steps = clean_data(df)

    cleaned_df.to_csv(cleaned_csv, index=False)
    pd.DataFrame(cleaning_steps).to_csv(log_csv, index=False)

    # -------------------------------
    # Cleaning insights
    # -------------------------------
    insights = generate_cleaning_insights(cleaning_steps)
    print("\nCleaning Insights:")
    for insight in insights:
        print("-", insight)

    # -------------------------------
    # Day 4 — Modeling
    # -------------------------------
    modeling_results = train_and_evaluate(
        df=cleaned_df,
        target_col=target_col,
        task_type=task_type
    )

    # -------------------------------
    # Final output (structured)
    # -------------------------------
    return {
        "cleaned_df": cleaned_df,
        "cleaning_steps": cleaning_steps,
        "cleaning_insights": insights,
        "modeling": modeling_results,
    }


if __name__ == "__main__":
    run_dany(
        input_csv="datasets/messy_data.csv",
        cleaned_csv="outputs/cleaned_data.csv",
        log_csv="outputs/cleaning_log.csv",
        target_col="target",          # 👈 replace with real column
        task_type="classification"    # or "regression"
    )
