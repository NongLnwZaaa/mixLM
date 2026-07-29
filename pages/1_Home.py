import streamlit as st

from utils.loader import load_data

df = load_data()

st.title("❤️ Heart Disease Prediction System")

st.markdown("""

ระบบนี้ใช้ Machine Learning สำหรับทำนายความเสี่ยงโรคหัวใจ

รองรับทั้งหมด 6 โมเดล

- K-Nearest Neighbor
- Decision Tree
- SVM
- K-Means
- Logistic Regression
- Random Forest

""")

st.divider()

col1,col2,col3,col4 = st.columns(4)

col1.metric(

    "Rows",

    len(df)

)

col2.metric(

    "Columns",

    len(df.columns)

)

col3.metric(

    "Features",

    len(df.columns)-1

)

col4.metric(

    "Target",

    "HeartDisease"

)

st.divider()

left,right = st.columns([2,1])

with left:

    st.subheader("Dataset Preview")

    st.dataframe(

        df.head(10),

        use_container_width=True

    )

with right:

    st.subheader("Algorithms")

    st.success("KNN")

    st.success("Decision Tree")

    st.success("SVM")

    st.success("K-Means")

    st.success("Regression")

    st.success("Random Forest")

st.divider()

st.info("""

เลือกเมนูด้านซ้ายเพื่อทดลองแต่ละโมเดล

""")