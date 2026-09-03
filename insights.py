import streamlit as st

def generate_insights(df):

    st.subheader("AI Insights")

    st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

    missing = df.isnull().sum().sum()

    st.write(f"Total missing values: {missing}")

    st.write("Top Columns:")

    st.write(df.describe())