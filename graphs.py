import plotly.express as px
import streamlit as st

def show_graphs(df):

    numeric_cols = df.select_dtypes(include='number').columns

    if len(numeric_cols) >= 2:

        x_axis = st.selectbox("Select X-Axis", numeric_cols)
        y_axis = st.selectbox("Select Y-Axis", numeric_cols)

        chart_type = st.selectbox(
            "Select Chart",
            ["Scatter", "Line", "Bar", "Histogram"]
        )

        if chart_type == "Scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis)

        elif chart_type == "Line":
            fig = px.line(df, x=x_axis, y=y_axis)

        elif chart_type == "Bar":
            fig = px.bar(df, x=x_axis, y=y_axis)

        else:
            fig = px.histogram(df, x=x_axis)

        st.plotly_chart(fig, use_container_width=True)