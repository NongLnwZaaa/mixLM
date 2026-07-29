import streamlit as st

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Heart Disease Prediction")
st.markdown("""
## Machine Learning Project
Machine Learning Model Demonstration

### Algorithms
- K-Nearest Neighbor (KNN)
- Decision Tree
- Support Vector Machine (SVM)
- K-Means Clustering
- Logistic Regression
- Random Forest
""")