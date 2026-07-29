import streamlit as st
from utils.loader import load_data

st.set_page_config(page_title="หน้าแรก", page_icon="🏠", layout="wide")

df = load_data()

st.title("🏠 ระบบพยากรณ์และประเมินความเสี่ยงโรคหัวใจ")
st.markdown("""
ระบบสาธิตการทำนายความเสี่ยงการเกิดโรคหัวใจด้วยอัลกอริทึม Machine Learning แบบหลากหลาย
- **KNN** (K-Nearest Neighbor)
- **Decision Tree** (ต้นไม้ตัดสินใจ)
- **SVM** (Support Vector Machine)
- **K-Means** (การจัดกลุ่มข้อมูล)
- **Logistic Regression** (การถดถอยโลจิสติก)
- **Random Forest** (ป่าสุ่มตัดสินใจ)
""")

st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("จำนวนแถวทั้งหมด", len(df))
col2.metric("จำนวนคอลัมน์", len(df.columns))
col3.metric("ฟีเจอร์ที่ใช้ทำนาย", len(df.columns) - 1)
col4.metric("ตัวแปรเป้าหมาย (Target)", "HeartDisease")

st.divider()

left, right = st.columns([2, 1])
with left:
    st.subheader("📊ตัวอย่างชุดข้อมูล (Dataset Preview)")
    st.dataframe(df.head(10), use_container_width=True)

with right:
    st.subheader("🤖 อัลกอริทึมที่พร้อมใช้งาน")
    st.success("KNN")
    st.success("Decision Tree")
    st.success("SVM")
    st.success("K-Means")
    st.success("Logistic Regression")
    st.success("Random Forest")

st.divider()
st.info("💡 เลือกเมนูจากแถบข้าง (Sidebar) เพื่อสำรวจข้อมูล หรือทดสอบทำนายผลด้วยโมเดลต่างๆ")