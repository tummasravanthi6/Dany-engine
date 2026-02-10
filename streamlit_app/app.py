import sys
import os

# Ensure the project root is in Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




import streamlit as st
import pandas as pd
import os

from dany_core.runner import run_dany_pipeline
from dany_core.targets.target_spec import TargetSpec
from dany_core.reports.html_report import generate_html_report

# ----------------------
# Streamlit UI
# ----------------------
st.title("DANY — Data Analysis & Modeling")

# 1️⃣ CSV Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
df = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"CSV loaded! {df.shape[0]} rows, {df.shape[1]} columns detected.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

# 2️⃣ Input Selection
target_col = None
task_type = None
if df is not None:
    target_col = st.selectbox("Select target column", options=df.columns)
    task_type = st.selectbox("Select task type", options=["classification", "regression"])

# 3️⃣ Run Pipeline Button
run_pipeline = st.button("Run Analysis")

if run_pipeline:
    if df is None:
        st.error("Please upload a CSV first.")
    elif not target_col or not task_type:
        st.error("Please select target column and task type.")
    else:
        st.info("Running Dany pipeline... please wait")
        try:
            # ----------------------
            # Prepare target spec
            # ----------------------
            target_spec = TargetSpec(
                name=target_col,
                task_type=task_type,
                description=f"Target column: {target_col}",
                allowed_null_ratio=0.05
            )

            # ----------------------
            # Run pipeline
            # ----------------------
            results = run_dany_pipeline(dataframe=df, target_spec=target_spec)

            st.success("Pipeline finished!")

            # ----------------------
            # Display results
            # ----------------------
            st.subheader("Target Validation")
            st.json(results.get("target_validation", {}))

            st.subheader("Cleaning Report")
            cleaning = results.get("cleaning", [])
            if cleaning:
                st.json(cleaning)
            else:
                st.write("No cleaning actions were performed.")

            st.subheader("EDA Profiles")
            profiles = results.get("profiles", {})
            if profiles:
                st.json(profiles)
            else:
                st.write("No EDA profiles available.")

            st.subheader("Modeling Results")
            modeling = results.get("modeling", {})
            if modeling:
                st.json(modeling)
            else:
                st.write("Modeling skipped or no results available.")

            # ----------------------
            # Generate & Download HTML report
            # ----------------------
            report_path = generate_html_report(results)
            results["report_path"] = report_path

            if report_path and os.path.exists(report_path):
                st.success("Report generated successfully!")

                with open(report_path, "rb") as f:
                    report_bytes = f.read()

                st.download_button(
                    label="Download DANY Report",
                    data=report_bytes,
                    file_name="dany_report.html",
                    mime="text/html"
                )
            else:
                st.warning("Report not generated or missing.")

        except Exception as e:
            st.error(f"Pipeline failed: {e}")
