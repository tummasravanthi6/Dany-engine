import pandas as pd

def basic_data_report(df: pd.DataFrame):
    missing_values = {col: int(df[col].isna().sum()) for col in df.columns}
    dtypes = {col: str(df[col].dtype) for col in df.columns}
    duplicate_rows = int(df.duplicated().sum())
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "dtypes": dtypes
    }
