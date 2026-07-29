import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils.loader import load_data
from utils.preprocess import preprocess
from utils.charts import show_confusion

st.set_page_config(page_title="Decision Tree", page_icon="🌳", layout="wide")
st.title("🌳 ต้นไม้ตัดสินใจ (Decision Tree Model)")

df = load_data()
X_train, X_test, y_train, y_test, preprocessor = preprocess(df)

st.sidebar.header("⚙️ ปรับแต่งพารามิเตอร์")
criterion = st.sidebar.selectbox("เกณฑ์การแยกข้อมูล (Criterion)", ["gini", "entropy", "log_loss"])
max_depth = st.sidebar.slider("ความลึกสูงสุดของต้นไม้ (Max Depth)", 1, 20, 5)

model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
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