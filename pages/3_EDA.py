import streamlit as st
import pandas as pd
import plotly.express as px
from utils.loader import load_data

st.set_page_config(layout="wide")

df = load_data()

st.title("Exploratory Data Analysis")
st.markdown("Heart Disease Analysis Dashboard")

# Numeric / Categorical Columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "int64"]).columns.tolist()

# Histogram
st.subheader("Histogram")
feature = st.selectbox("Feature for Histogram", numeric_cols)
fig = px.histogram(df, x=feature, nbins=30, color="HeartDisease")
st.plotly_chart(fig, use_container_width=True)

# Boxplot
st.subheader("Box Plot")
feature2 = st.selectbox("Feature for Box Plot", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
fig2 = px.box(df, x="HeartDisease", y=feature2, color="HeartDisease")
st.plotly_chart(fig2, use_container_width=True)

# Pie Chart
st.subheader("Target Distribution")
count = df["HeartDisease"].value_counts()
pie = px.pie(values=count.values, names=count.index, title="Heart Disease Proportion")
st.plotly_chart(pie, use_container_width=True)

# Count Plot
st.subheader("Categorical Distribution")
cat = st.selectbox("Categorical Feature", categorical_cols)
count_fig = px.histogram(df, x=cat, color="HeartDisease", barmode="group")
st.plotly_chart(count_fig, use_container_width=True)

# Correlation
st.subheader("Correlation Heatmap")
corr = df[numeric_cols].corr()
heat = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu")
st.plotly_chart(heat, use_container_width=True)

# Scatter Plot
st.subheader("Scatter Plot")
c1, c2 = st.columns(2)
with c1:
    x = st.selectbox("X Axis", numeric_cols, key="x")
with c2:
    y = st.selectbox("Y Axis", numeric_cols, index=2 if len(numeric_cols) > 2 else 0, key="y")

scatter = px.scatter(df, x=x, y=y, color="HeartDisease", hover_data=df.columns)
st.plotly_chart(scatter, use_container_width=True)