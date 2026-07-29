import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils.loader import load_data
from utils.preprocess import preprocess
from utils.charts import show_confusion

st.set_page_config(page_title="KNN", page_icon="🎯", layout="wide")
st.title("🎯 K-Nearest Neighbor (KNN)")

df = load_data()
X_train, X_test, y_train, y_test, preprocessor = preprocess(df)

# Sidebar
st.sidebar.header("⚙️ ปรับแต่งพารามิเตอร์")
k = st.sidebar.slider("จำนวนเพื่อนบ้านใกล้ที่สุด (K)", 1, 15, 5)
weight = st.sidebar.selectbox("น้ำหนัก (Weights)", ["uniform", "distance"])
metric = st.sidebar.selectbox("วัดระยะห่าง (Distance Metric)", ["euclidean", "manhattan", "minkowski"])

model = KNeighborsClassifier(n_neighbors=k, weights=weight, metric=metric)
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

c1, c2, c3 = st.columns(3)
c1.metric("ความแม่นยำ (Accuracy)", f"{acc:.2%}")
c2.metric("จำนวนข้อมูลฝึกลอง (Train)", len(X_train))
c3.metric("จำนวนข้อมูลทดสอบ (Test)", len(X_test))

st.divider()

left, right = st.columns([1, 1])
with left:
    st.subheader("📊 ตารางความสับสน (Confusion Matrix)")
    cm = confusion_matrix(y_test, pred)
    show_confusion(cm)
with right:
    st.subheader("📋 รายงานการประเมินโมเดล (Classification Report)")
    report = classification_report(y_test, pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

st.divider()

# ส่วนการทำนายข้อมูลใหม่
st.subheader("🩺 ทำนายผลความเสี่ยงคนไข้ใหม่")
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

if st.button("🩺 ทำนายความเสี่ยงโรคหัวใจ"):
    sample = pd.DataFrame({
        "Age": [age], "Sex": [sex], "ChestPainType": [pain], "RestingBP": [bp],
        "Cholesterol": [chol], "FastingBS": [fast], "RestingECG": [ecg],
        "MaxHR": [hr], "ExerciseAngina": [angina], "Oldpeak": [oldpeak], "ST_Slope": [slope]
    })
    
    sample_scaled = preprocessor.transform(sample)
    result = model.predict(sample_scaled)[0]
    prob = model.predict_proba(sample_scaled)[0]
    
    if result == 1:
        st.error("🚨 ตรวจพบความเสี่ยงเป็นโรคหัวใจ (Heart Disease Detected)")
    else:
        st.success("✅ ผลตรวจอยู่ในเกณฑ์ปกติ (Normal)")
        
    st.write(f"โอกาสปกติ: **{prob[0]*100:.2f}%**")
    st.write(f"โอกาสเสี่ยงเป็นโรคหัวใจ: **{prob[1]*100:.2f}%**")