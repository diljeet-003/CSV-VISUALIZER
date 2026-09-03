import pandas as pd

def recommend_chart(df):

    recommendations = []

    numeric_cols = df.select_dtypes(
        include=['number']
    ).columns.tolist()

    categorical_cols = df.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()

    datetime_cols = []

    for col in df.columns:
        try:
            pd.to_datetime(df[col])
            datetime_cols.append(col)
        except:
            pass

    if datetime_cols and numeric_cols:
        recommendations.append({
            "chart": "Line Chart",
            "x": datetime_cols[0],
            "y": numeric_cols[0]
        })

    if len(numeric_cols) >= 2:
        recommendations.append({
            "chart": "Scatter Plot",
            "x": numeric_cols[0],
            "y": numeric_cols[1]
        })

    if len(numeric_cols) >= 2:
        recommendations.append({
            "chart": "Heatmap"
        })

    if numeric_cols:
        recommendations.append({
            "chart": "Histogram",
            "x": numeric_cols[0]
        })

    if numeric_cols:
        recommendations.append({
            "chart": "Box Plot",
            "y": numeric_cols[0]
        })

    if numeric_cols and categorical_cols:
        recommendations.append({
            "chart": "Violin Plot",
            "x": categorical_cols[0],
            "y": numeric_cols[0]
        })

    if categorical_cols:
        recommendations.append({
            "chart": "Pie Chart",
            "names": categorical_cols[0]
        })

    if categorical_cols and numeric_cols:
        recommendations.append({
            "chart": "Treemap",
            "path": categorical_cols[0],
            "values": numeric_cols[0]
        })

    if len(categorical_cols) >= 2:
        recommendations.append({
            "chart": "Sunburst",
            "path": categorical_cols[:2]
        })

    if len(numeric_cols) >= 3:
        recommendations.append({
            "chart": "Radar Chart"
        })

    return recommendations