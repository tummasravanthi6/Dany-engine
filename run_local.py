import pandas as pd

from dany_core.runner import run_dany


def main():
    # ------------------
    # Load data
    # ------------------
    df = pd.read_csv(
        r"C:\Users\sravanthi\OneDrive\Desktop\dany\datasets\messy_data.csv"
    )

    target_col = "target"  # change ONLY if your column name is different

    # Optional: verify columns (safe to remove later)
    print("Loaded columns:", list(df.columns))

    # ------------------
    # Run DANY pipeline
    # ------------------
    result = run_dany(df, target_col)

    # ------------------
    # Print insights
    # ------------------
    print("\n=== INSIGHTS ===")
    if not result.get("insights"):
        print("No insights generated (insufficient evidence).")
    else:
        for ins in result["insights"]:
            print("-", ins)

    # ------------------
    # Print modeling results
    # ------------------
    modeling = result.get("modeling", {})
    print("\n=== MODELING RESULTS ===")
    print("Task type:", modeling.get("task_type"))

    print("\nAll models:\n")
    for m in modeling.get("all_models_results", []):
        print(f"Model: {m.get('model_name')}")
        print("Metrics:", m.get("metrics"))
        print("Warnings:", m.get("warnings"))
        print("Is best:", m.get("is_best"))
        print()

    print("Best model summary:")
    print(modeling.get("best_model_summary"))

    # ------------------
    # Print trust warnings (Day 5)
    # ------------------
    print("\n=== TRUST WARNINGS ===")
    trust_warnings = result.get("trust_warnings", [])
    if not trust_warnings:
        print("No trust warnings.")
    else:
        for w in trust_warnings:
            print("-", w)

    # ------------------
    # Print confidence warnings (Day 5)
    # ------------------
    print("\n=== CONFIDENCE WARNINGS ===")
    confidence_warnings = result.get("confidence_warnings", [])
    if not confidence_warnings:
        print("No confidence warnings.")
    else:
        for w in confidence_warnings:
            print("-", w)


if __name__ == "__main__":
    main()
