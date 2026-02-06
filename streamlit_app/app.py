import sys
import os

# Add parent folder to path so dany_core imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from dany_core.cleaning import clean_data
from dany_core.report import basic_data_report
from dany_core.insights import generate_cleaning_insights

st.title("Dany: Data Cleaning Dashboard")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Basic Data Report")
    report = basic_data_report(df)
    st.json(report)

    st.subheader("Cleaning Data")
    cleaned_df, cleaning_steps = clean_data(df)

    st.subheader("Cleaning Steps")
    for step in cleaning_steps:
        # Highlight high missingness
        if step.get("action") == "flag_high_missingness":
            st.warning(
                f"Column '{step['column']}' has high missing values ({float(step['ratio'])*100:.0f}%), which may affect reliability."
            )
        else:
            st.write(step)

    st.subheader("Cleaning Insights")
    insights = generate_cleaning_insights(cleaning_steps)
    for insight in insights:
        st.write("-", insight)

    st.subheader("Cleaned Data Preview (first 100 rows)")
    st.dataframe(cleaned_df.head(100))
