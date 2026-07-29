import streamlit as st
import pandas as pd
from utils.loader import load_data

st.set_page_config(page_title="สำรวจชุดข้อมูล", page_icon="📁", layout="wide")
df = load_data()

st.title("📁 สำรวจชุดข้อมูล (Dataset Explorer)")
st.markdown("**ไฟล์ข้อมูลหลัก: Heart3.csv**")
st.divider()

# การ์ดสรุปข้อมูล
c1, c2, c3, c4 = st.columns(4)
c1.metric("จำนวนรายการ (Rows)", len(df))
c2.metric("จำนวนคอลัมน์ (Columns)", len(df.columns))
c3.metric("ค่าที่สูญหาย (Missing Values)", int(df.isnull().sum().sum()))
c4.metric("ข้อมูลซ้ำ (Duplicates)", int(df.duplicated().sum()))
st.divider()

# ค้นหาข้อมูล
st.subheader("🔍 ค้นหาข้อมูล")
keyword = st.text_input("พิมพ์คำค้นหาที่ต้องการในชุดข้อมูล:")
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

# เลือกแสดงคอลัมน์
st.subheader("📌 เลือกคอลัมน์ที่ต้องการดู")
cols = st.multiselect("คอลัมน์", df.columns, default=list(df.columns))
st.dataframe(df[cols], use_container_width=True)
st.divider()

# ประเภทข้อมูล
st.subheader("🏷️ ชนิดข้อมูลของแต่ละคอลัมน์ (Data Types)")
dtype = pd.DataFrame({
    "ชื่อคอลัมน์": df.columns,
    "ชนิดข้อมูล": df.dtypes.astype(str)
})
st.dataframe(dtype, hide_index=True, use_container_width=True)
st.divider()

# สรุปค่าสูญหาย
st.subheader("⚠️ ตรวจสอบค่าที่สูญหาย (Missing Values)")
missing = pd.DataFrame({
    "ชื่อคอลัมน์": df.columns,
    "จำนวนที่สูญหาย": df.isnull().sum(),
    "คิดเป็นร้อยละ (%)": round(df.isnull().mean() * 100, 2)
})
st.dataframe(missing, hide_index=True, use_container_width=True)
st.divider()

# สถิติเชิงพรรณนา
st.subheader("📈 สถิติเชิงพรรณนา (Descriptive Statistics)")
st.dataframe(df.describe(include="all"), use_container_width=True)
st.divider()

# ดาวน์โหลด
csv = df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "📥 ดาวน์โหลดไฟล์ CSV",
    csv,
    "Heart3.csv",
    "text/csv"
)