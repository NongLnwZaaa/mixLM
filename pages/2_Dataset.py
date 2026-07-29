import streamlit as st
import pandas as pd

from utils.loader import load_data

st.set_page_config(layout="wide")

df = load_data()

st.title("📊 Dataset Explorer")

st.markdown("สำรวจข้อมูลชุด **Heart3.csv**")

st.divider()

# ===========================
# Summary Cards
# ===========================

c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", len(df))
c2.metric("Columns", len(df.columns))
c3.metric("Missing Values", int(df.isnull().sum().sum()))
c4.metric("Duplicate", int(df.duplicated().sum()))

st.divider()

# ===========================
# Search
# ===========================

st.subheader("🔍 Search")

keyword = st.text_input("ค้นหาข้อมูล")

if keyword:

    result = df[
        df.astype(str)
        .apply(lambda x: x.str.contains(keyword, case=False))
        .any(axis=1)
    ]

    st.dataframe(result, use_container_width=True)

else:

    st.dataframe(df, use_container_width=True)

st.divider()

# ===========================
# Column Filter
# ===========================

st.subheader("📑 Select Columns")

cols = st.multiselect(

    "Columns",

    df.columns,

    default=df.columns

)

st.dataframe(

    df[cols],

    use_container_width=True

)

st.divider()

# ===========================
# Data Types
# ===========================

st.subheader("📋 Data Types")

dtype = pd.DataFrame({

    "Column":df.columns,

    "Type":df.dtypes.astype(str)

})

st.dataframe(

    dtype,

    hide_index=True,

    use_container_width=True

)

st.divider()

# ===========================
# Missing Values
# ===========================

st.subheader("❗ Missing Values")

missing = pd.DataFrame({

    "Column":df.columns,

    "Missing":df.isnull().sum(),

    "Percent":round(df.isnull().mean()*100,2)

})

st.dataframe(

    missing,

    hide_index=True,

    use_container_width=True

)

st.divider()

# ===========================
# Statistics
# ===========================

st.subheader("📈 Descriptive Statistics")

st.dataframe(

    df.describe(include="all"),

    use_container_width=True

)

st.divider()

# ===========================
# Download
# ===========================

csv = df.to_csv(index=False).encode("utf-8-sig")

st.download_button(

    "⬇ Download CSV",

    csv,

    "Heart3.csv",

    "text/csv"

)