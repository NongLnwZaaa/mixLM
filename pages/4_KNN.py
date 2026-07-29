import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils.loader import load_data
from utils.preprocess import preprocess
from utils.charts import show_confusion

st.set_page_config(page_title="KNN", page_icon="🔍", layout="wide")
st.title("K-Nearest Neighbor (KNN)")

df = load_data()
X_train, X_test, y_train, y_test, preprocessor = preprocess(df)

# Sidebar Parameters
st.sidebar.header("Model Parameters")
k = st.sidebar.slider("K", 1, 15, 5)
weight = st.sidebar.selectbox("Weights", ["uniform", "distance"])
metric = st.sidebar.selectbox("Distance Metric", ["euclidean", "manhattan", "minkowski"])

# Train Model
model = KNeighborsClassifier(n_neighbors=k, weights=weight, metric=metric)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

# Metrics Cards
c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{acc:.2%}")
c2.metric("Train Samples", len(X_train))
c3.metric("Test Samples", len(X_test))

st.divider()

# Confusion Matrix & Classification Report
left, right = st.columns([1, 1])
with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, pred)
    show_confusion(cm)

with right:
    st.subheader("Classification Report")
    report = classification_report(y_test, pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

st.divider()

# Prediction Interface
st.subheader("Predict New Patient")
age = st.number_input("Age", 20, 100, 40)
sex = st.selectbox("Sex (1: Male, 0: Female)", [1, 0])
pain = st.selectbox("ChestPainType (1-4)", [1, 2, 3, 4])
bp = st.number_input("RestingBP", 50, 250, 120)
chol = st.number_input("Cholesterol", 0, 700, 200)
fast = st.selectbox("FastingBS", [0, 1])
ecg = st.selectbox("RestingECG", [0, 1, 2, 3])
hr = st.number_input("MaxHR", 50, 220, 150)
angina = st.selectbox("ExerciseAngina", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
slope = st.selectbox("ST_Slope", [1, 2, 3])

if st.button("Predict"):
    sample = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [pain],
        "RestingBP": [bp],
        "Cholesterol": [chol],
        "FastingBS": [fast],
        "RestingECG": [ecg],
        "MaxHR": [hr],
        "ExerciseAngina": [angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [slope]
    })
    
    sample_scaled = preprocessor.transform(sample)
    result = model.predict(sample_scaled)[0]
    prob = model.predict_proba(sample_scaled)[0]
    
    if result == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ Normal")
        
    st.write(f"Normal Probability: {prob[0]*100:.2f}%")
    st.write(f"Heart Disease Probability: {prob[1]*100:.2f}%")

st.divider()

# Download Trained Model
joblib.dump(model, "models/knn.pkl")
with open("models/knn.pkl", "rb") as f:
    st.download_button(
        "Download KNN Model (.pkl)",
        f,
        file_name="knn.pkl"
    )