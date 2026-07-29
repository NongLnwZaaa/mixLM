import streamlit as st
import pandas as pd
import plotly.express as px
from utils.loader import load_data

st.set_page_config(page_title="วิเคราะห์ข้อมูล (EDA)", page_icon="📊", layout="wide")
df = load_data()

st.title("📊 การวิเคราะห์ข้อมูลเชิงสำรวจ (EDA)")
st.markdown("แดชบอร์ดแสดงผลการวิเคราะห์ปัจจัยความเสี่ยงโรคหัวใจ")

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "int64"]).columns.tolist()

# Histogram
st.subheader("📉 กราฟแจกแจงความถี่ (Histogram)")
feature = st.selectbox("เลือกคุณลักษณะ (Feature) ที่ต้องการดู กราฟ Histogram", numeric_cols)
fig = px.histogram(df, x=feature, nbins=30, color="HeartDisease", title=f"การแจกแจงของ {feature} แยกตามการเป็นโรคหัวใจ")
st.plotly_chart(fig, use_container_width=True)

# Boxplot
st.subheader("📦 กราฟกล่อง (Box Plot)")
feature2 = st.selectbox("เลือกคุณลักษณะสำหรับ Box Plot", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
fig2 = px.box(df, x="HeartDisease", y=feature2, color="HeartDisease", title=f"เปรียบเทียบ {feature2} กับกลุ่มที่เป็น/ไม่เป็นโรคหัวใจ")
st.plotly_chart(fig2, use_container_width=True)

# Pie Chart
st.subheader("🥧 สัดส่วนผู้ป่วยโรคหัวใจ (Target Distribution)")
count = df["HeartDisease"].value_counts()
pie = px.pie(values=count.values, names=count.index, title="สัดส่วนผู้ป่วยโรคหัวใจ (1 = เป็นโรค, 0 = ปกติ)")
st.plotly_chart(pie, use_container_width=True)

# Count Plot
st.subheader("📊 การแจกแจงของข้อมูลเชิงกลุ่ม (Categorical Distribution)")
cat = st.selectbox("เลือกคุณลักษณะเชิงกลุ่ม", categorical_cols)
count_fig = px.histogram(df, x=cat, color="HeartDisease", barmode="group", title=f"เปรียบเทียบ {cat} แยกตามสถิติโรคหัวใจ")
st.plotly_chart(count_fig, use_container_width=True)

# Correlation
st.subheader("🔥 แผนภาพความสัมพันธ์ (Correlation Heatmap)")
corr = df[numeric_cols].corr()
heat = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu", title="ระดับความสัมพันธ์ระหว่างตัวแปรเชิงตัวเลข")
st.plotly_chart(heat, use_container_width=True)

# Scatter Plot
st.subheader("📍 กราฟกระจายตัว (Scatter Plot)")
c1, c2 = st.columns(2)
with c1:
    x = st.selectbox("แกน X", numeric_cols, key="x")
with c2:
    y = st.selectbox("แกน Y", numeric_cols, index=2 if len(numeric_cols) > 2 else 0, key="y")
scatter = px.scatter(df, x=x, y=y, color="HeartDisease", hover_data=df.columns, title=f"ความสัมพันธ์ระหว่าง {x} และ {y}")
st.plotly_chart(scatter, use_container_width=True)