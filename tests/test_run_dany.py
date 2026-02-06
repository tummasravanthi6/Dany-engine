import pandas as pd
from dany_core.runner import run_dany

# Load any real CSV file
df = pd.read_csv("datasets/messy_data.csv")
 # replace with an actual CSV path
target_col = df.columns[-1]

result = run_dany(df, target_col)

assert result["profiles"]["numerical"] is not None
assert result["profiles"]["categorical"] is not None
assert result["profiles"]["target"] is not None
assert isinstance(result["insights"], list)

print(f"Generated {len(result['insights'])} insights")
print("DAY 3 VALIDATION PASSED")
