import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils.loader import load_data
from utils.preprocess import preprocess
from utils.charts import show_confusion

st.set_page_config(page_title="Logistic Regression", page_icon="📈", layout="wide")
st.title("Logistic Regression")

df = load_data()
X_train, X_test, y_train, y_test, preprocessor = preprocess(df)

st.sidebar.header("Model Parameters")
c_val = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 200)

model = LogisticRegression(C=c_val, max_iter=max_iter, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

st.metric("Accuracy", f"{acc:.2%}")

left, right = st.columns([1, 1])
with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, pred)
    show_confusion(cm)
with right:
    st.subheader("Classification Report")
    report = classification_report(y_test, pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)