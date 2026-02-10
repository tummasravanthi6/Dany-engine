import streamlit as st
import pandas as pd

from dany_core.runner import run_dany_pipeline

# =====================================================
# Page config (MUST be first Streamlit command)
# =====================================================
st.set_page_config(page_title="DANY", layout="wide")

# =====================================================
# Header
# =====================================================
st.title("DANY – Decision Intelligence System")
st.caption("Streamlit UI · Controlled pipeline execution")

# =====================================================
# Step 1 — Upload Dataset
# =====================================================
st.divider()
st.subheader("1️⃣ Upload dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
)

df = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(
            f"Dataset loaded successfully ({df.shape[0]} rows, {df.shape[1]} columns)"
        )
        st.dataframe(df.head())
    except Exception:
        st.error("Invalid CSV file. Please upload a valid CSV.")

# =====================================================
# Step 2 — Target & Task Selection
# =====================================================
target_column = None
task_type = None

if df is not None:
    st.divider()
    st.subheader("2️⃣ Analysis configuration")

    target_column = st.selectbox(
        "Select target column",
        options=df.columns.tolist(),
        index=None,
        placeholder="Choose the target variable",
    )

    task_type = st.selectbox(
        "Select task type",
        options=["classification", "regression"],
        index=None,
        placeholder="Choose task type",
    )

    if target_column and task_type:
        st.info(
            f"Target: **{target_column}** | Task: **{task_type}**"
        )

# =====================================================
# Step 3 — Run Analysis (Controlled Execution)
# =====================================================
if df is not None and target_column and task_type:
    st.divider()
    st.subheader("3️⃣ Run analysis")

    run_clicked = st.button("▶️ Run DANY Analysis")

    if run_clicked:
        with st.spinner("Running DANY pipeline... please wait"):
            try:
                results = run_dany_pipeline(
                    dataframe=df,
                    target_column=target_column,
                    task_type=task_type,
                )

                st.success("Analysis completed successfully ✅")
                st.session_state["results"] = results

            except Exception:
                st.error(
                    "Analysis failed. Please check the dataset and configuration."
                )

# =====================================================
# Step 4 — Results Display
# =====================================================
results = st.session_state.get("results")

if results:
    st.divider()
    st.header("📊 Analysis Results")

    # -------------------------------------------------
    # 1. Executive Summary
    # -------------------------------------------------
    st.subheader("1️⃣ Executive Summary")

    exec_summary = results.get("executive_summary")

    if exec_summary:
        for key, value in exec_summary.items():
            st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
    else:
        st.info("Executive summary not available.")

    # -------------------------------------------------
    # 2. Key Insights
    # -------------------------------------------------
    st.subheader("2️⃣ Key Insights")

    insights = results.get("insights", [])

    if insights:
        for i, insight in enumerate(insights[:5], start=1):
            st.markdown(f"{i}. {insight}")
    else:
        st.info("No insights generated.")

    # -------------------------------------------------
    # 3. Modeling Summary
    # -------------------------------------------------
    st.subheader("3️⃣ Modeling Summary")

    modeling = results.get("modeling", {})

    if modeling:
        st.json(modeling)
    else:
        st.info("Modeling was skipped or unavailable.")

    # -------------------------------------------------
    # 4. Prediction Trust & Warnings
    # -------------------------------------------------
    st.subheader("4️⃣ Prediction Trust & Warnings")

    trust_warnings = results.get("trust_warnings", [])
    confidence_warnings = results.get("confidence_warnings", [])

    if trust_warnings:
        for warning in trust_warnings:
            st.warning(warning)

    if confidence_warnings:
        for warning in confidence_warnings:
            st.warning(warning)

    if not trust_warnings and not confidence_warnings:
        st.success("No major trust or confidence risks detected.")

    # -------------------------------------------------
    # 5. Report Download
    # -------------------------------------------------
    st.subheader("5️⃣ Download Report")

    report_path = results.get("report_path")

    if report_path:
        try:
            with open(report_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download HTML Report",
                    data=f,
                    file_name="dany_report.html",
                    mime="text/html",
                )
        except Exception:
            st.error("Report file not found.")
    else:
        st.info("No report available for download.")
