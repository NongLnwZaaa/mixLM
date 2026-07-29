import streamlit as st

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("❤️ Heart Disease Prediction")

st.markdown("""
## Machine Learning Project

โปรเจกต์นี้ใช้ Machine Learning วิเคราะห์โรคหัวใจ

### Algorithms

- K-Nearest Neighbor
- Decision Tree
- SVM
- K-Means
- Logistic Regression
- Random Forest

เลือกเมนูทางด้านซ้ายเพื่อเริ่มใช้งาน
""")