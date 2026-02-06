




import streamlit as st
import pandas as pd
from dany_core.runner import run_dany

st.title("Dany – Day 1 Demo")

# 1️⃣ Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # 2️⃣ Select target column
    target_col = st.selectbox("Select target column", df.columns)

    # 3️⃣ Select task type
    task_type = st.selectbox("Select task type", ["regression", "classification"])

    # 4️⃣ Run Dany
    output = run_dany(df, target_col, task_type)

    # 5️⃣ Display results
    st.subheader("Data Report")
    st.write(output["data_report"])

    st.subheader("EDA Insights")
    for insight in output["eda_insights"]:
        st.write(f"- [{insight['type'].upper()}] {insight['message']} ({insight['why_it_matters']})")

    st.subheader("Executive Summary")
    st.write(output["executive_summary"])
