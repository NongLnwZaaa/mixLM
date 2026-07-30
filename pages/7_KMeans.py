import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
from utils.loader import load_data
from utils.preprocess import preprocess

st.set_page_config(page_title="K-Means", page_icon="📊", layout="wide")
st.title("📊 การแบ่งกลุ่มด้วย K-Means (K-Means Clustering)")

df = load_data()
X_train, X_test, y_train, y_test, preprocessor = preprocess(df)

# Sidebar
st.sidebar.header("⚙️ ปรับแต่งพารามิเตอร์")
n_clusters = st.sidebar.slider("จำนวนกลุ่มที่ต้องการจัด (K Clusters)", 2, 10, 2)

# Train K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
X_all = preprocessor.transform(df.drop(columns=["HeartDisease"]))
df['Cluster'] = kmeans.fit_predict(X_all)

# แสดงผลการจัดกลุ่ม
st.subheader("📍 แผนภาพการจัดกลุ่ม (Cluster Visualization)")
fig = px.scatter(
    df, x="Age", y="Cholesterol", 
    color=df['Cluster'].astype(str), 
    symbol="HeartDisease",
    title="ผลการจัดกลุ่มด้วย K-Means (เทียบระหว่าง อายุ กับ คลอเรสเตอรอล)"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("📋 ค่าเฉลี่ยของคุณลักษณะในแต่ละกลุ่ม (Cluster Summary)")
st.dataframe(df.groupby('Cluster').mean(numeric_only=True), use_container_width=True)

st.divider()

# ส่วนจัดกลุ่มคนไข้ใหม่
st.subheader("🔍 ระบุกลุ่มให้กับคนไข้ใหม่ (Predict Cluster)")
col_a, col_b = st.columns(2)
with col_a:
    age = st.number_input("อายุ (Age)", 20, 100, 40)
    sex = st.selectbox("เพศ (Sex)", [1, 0], format_func=lambda x: "ชาย (1)" if x == 1 else "หญิง (0)")
    pain = st.selectbox("ประเภทอาการปวดหน้าอก (ChestPainType)", [1, 2, 3, 4], format_func=lambda x: f"แบบที่ {x}")
    bp = st.number_input("ความดันโลหิตขณะพัก (RestingBP)", 50, 250, 120)
    chol = st.number_input("คลอเรสเตอรอล (Cholesterol)", 0, 700, 200)
    fast = st.selectbox("น้ำตาลในเลือดหลังอดอาหาร > 120 mg/dl (FastingBS)", [0, 1], format_func=lambda x: "ใช่ (1)" if x == 1 else "ไม่ใช่ (0)")

with col_b:
    ecg = st.selectbox("ผลคลื่นไฟฟ้าหัวใจ (RestingECG)", [0, 1, 2, 3])
    hr = st.number_input("อัตราการเต้นหัวใจสูงสุด (MaxHR)", 50, 220, 150)
    angina = st.selectbox("อาการเจ็บหน้าอกขณะออกกำลังกาย (ExerciseAngina)", [0, 1], format_func=lambda x: "มีอาการ (1)" if x == 1 else "ไม่มีอาการ (0)")
    oldpeak = st.number_input("ค่า Oldpeak", 0.0, 10.0, 1.0)
    slope = st.selectbox("ความชัน ST_Slope", [1, 2, 3])

if st.button("🔍 ตรวจสอบกลุ่มของคนไข้"):
    sample = pd.DataFrame({
        "Age": [age], "Sex": [sex], "ChestPainType": [pain], "RestingBP": [bp],
        "Cholesterol": [chol], "FastingBS": [fast], "RestingECG": [ecg],
        "MaxHR": [hr], "ExerciseAngina": [angina], "Oldpeak": [oldpeak], "ST_Slope": [slope]
    })
    
    sample_scaled = preprocessor.transform(sample)
    cluster_pred = kmeans.predict(sample_scaled)[0]
    
    st.info(f"📌 คนไข้รายนี้ถูกจัดอยู่ใน **กลุ่มที่ {cluster_pred} (Cluster {cluster_pred})**")
    
    # แสดงลักษณะเฉลี่ยของกลุ่มที่ตกไปอยู่
    cluster_info = df[df['Cluster'] == cluster_pred]
    avg_age = cluster_info['Age'].mean()
    avg_chol = cluster_info['Cholesterol'].mean()
    disease_rate = (cluster_info['HeartDisease'].sum() / len(cluster_info)) * 100
    
    st.write(f"- อายุเฉลี่ยของคนในกลุ่มนี้: **{avg_age:.1f} ปี**")
    st.write(f"- คลอเรสเตอรอลเฉลี่ยของคนในกลุ่มนี้: **{avg_chol:.1f} mg/dl**")
    st.write(f"- สัดส่วนผู้ป่วยโรคหัวใจในกลุ่มนี้: **{disease_rate:.1f}%**")