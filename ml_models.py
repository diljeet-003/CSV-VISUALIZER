from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def apply_kmeans(df, n_clusters=3):

    df_copy = df.copy()

    # Select only numeric columns
    numeric_cols = df_copy.select_dtypes(include=np.number).columns

    # If no numeric columns exist
    if len(numeric_cols) == 0:
        return df_copy

    # Fill missing values with column mean
    df_copy[numeric_cols] = df_copy[numeric_cols].fillna(
        df_copy[numeric_cols].mean()
    )

    # Remove columns that are entirely NaN
    df_copy = df_copy.dropna(axis=1, how="all")

    numeric_cols = df_copy.select_dtypes(include=np.number).columns

    if len(numeric_cols) == 0:
        return df_copy

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        df_copy[numeric_cols]
    )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(scaled_data)

    df_copy["Cluster"] = clusters

    return df_copy