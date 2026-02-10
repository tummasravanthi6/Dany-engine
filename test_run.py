import pandas as pd
from dany_core.runner import run_dany_pipeline
from dany_core.targets.target_spec import TargetSpec

# 1. Load test data
df = pd.read_csv("datasets/messy_data.csv")

# 2. Define target
target_spec = TargetSpec(
    name="target",  # match your CSV column
    task_type="classification",  # or "regression"
    description="Target column for testing",
    allowed_null_ratio=0.05
)

# 3. Run pipeline
results = run_dany_pipeline(
    dataframe=df,
    target_spec=target_spec
)

# 4. Print confirmation
print("Pipeline finished")
print("Keys returned:", results.keys())
print("Report path:", results.get("report_path"))
print("Target validation:", results.get("target_validation"))


import pprint

print("\n--- CLEANING REPORT ---")
pprint.pprint(results["cleaning"])

print("\n--- EDA PROFILES ---")
pprint.pprint(results["profiles"])

print("\n--- MODELING RESULTS ---")
pprint.pprint(results["modeling"])

