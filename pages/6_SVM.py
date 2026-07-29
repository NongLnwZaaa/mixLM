import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils.loader import load_data
from utils.preprocess import preprocess
from utils.charts import show_confusion

st.set_page_config(page_title="SVM", page_icon="⚡", layout="wide")
st.title("⚡ Support Vector Machine (SVM)")

df = load_data()
X_train, X_test, y_train, y_test, preprocessor = preprocess(df)

st.sidebar.header("⚙️ ปรับแต่งพารามิเตอร์")
c_val = st.sidebar.slider("ค่า C (Regularization)", 0.01, 10.0, 1.0)
kernel = st.sidebar.selectbox("ฟังก์ชันเคอร์เนล (Kernel)", ["rbf", "linear", "poly", "sigmoid"])

model = SVC(C=c_val, kernel=kernel, probability=True, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

st.metric("ความแม่นยำ (Accuracy)", f"{acc:.2%}")

left, right = st.columns([1, 1])
with left:
    st.subheader("📊 ตารางความสับสน (Confusion Matrix)")
    cm = confusion_matrix(y_test, pred)
    show_confusion(cm)
with right:
    st.subheader("📋 รายงานการประเมินโมเดล (Classification Report)")
    report = classification_report(y_test, pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)