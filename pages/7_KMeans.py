import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
from utils.loader import load_data
from utils.preprocess import preprocess

st.set_page_config(page_title="K-Means Clustering", page_icon="📊", layout="wide")
st.title("K-Means Clustering")

df = load_data()
X_train, X_test, y_train, y_test, preprocessor = preprocess(df)

st.sidebar.header("Model Parameters")
n_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 10, 2)

# Train Model
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(preprocessor.transform(df.drop(columns=["HeartDisease"])))

st.subheader("Cluster Visualization")
fig = px.scatter(
    df, x="Age", y="Cholesterol", 
    color=df['Cluster'].astype(str), 
    symbol="HeartDisease",
    title="K-Means Clusters (Age vs Cholesterol)"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Cluster Summary")
st.dataframe(df.groupby('Cluster').mean(numeric_only=True), use_container_width=True)