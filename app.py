import streamlit as st

st.set_page_config(
    page_title="ระบบทำนายโรคหัวใจ",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("❤️ ระบบทำนายความเสี่ยงโรคหัวใจ")
st.markdown("""
## โปรเจกต์เรียนรู้ Machine Learning
ระบบสาธิตการทำงานของโมเดล Machine Learning สำหรับการประเมินความเสี่ยงโรคหัวใจ

### อัลกอริทึมที่ใช้ในระบบ
- K-Nearest Neighbor (KNN)
- ต้นไม้ตัดสินใจ (Decision Tree)
- Support Vector Machine (SVM)
- การแบ่งกลุ่มด้วย K-Means (K-Means Clustering)
- การถดถอยโลจิสติก (Logistic Regression)
- ป่าสุ่ม (Random Forest)
""")