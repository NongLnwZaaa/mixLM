import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils.loader import load_data
from utils.preprocess import preprocess
from utils.charts import show_confusion

st.set_page_config(page_title="Random Forest", page_icon="🌲", layout="wide")
st.title("Random Forest Classifier")

df = load_data()
X_train, X_test, y_train, y_test, preprocessor = preprocess(df)

st.sidebar.header("Model Parameters")
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100, step=10)
max_depth = st.sidebar.slider("Max Depth", 1, 20, 10)

model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
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