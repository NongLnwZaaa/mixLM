import streamlit as st

st.set_page_config(
    page_title="ข้อมูลผู้พัฒนา - ระบบทำนายโรคหัวใจ",
    page_icon="👨‍💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("👨‍💻 ข้อมูลผู้พัฒนาโปรเจกต์")
st.markdown("---")

# ส่วนข้อมูลผู้พัฒนา
col1, col2 = st.columns([1, 2])

with col1:
    # คุณสามารถเปลี่ยนเป็นรูปของคุณได้โดยใส่ลิงก์รูป หรือ path ไฟล์รูป
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=180)

with col2:
    st.subheader("ชื่อผู้พัฒนา")
    # ✏️ กรอกชื่อ-นามสกุล และรหัสนักศึกษาของคุณตรงนี้ได้เลยครับ
    st.markdown("นาย กรภัทร์ ถิ่นผาแดง")
    st.write("รหัสนักศึกษา: 664245016")
    st.write("สาขาวิชา: วิทยาการคอมพิวเตอรร์")
    st.write("มหาวิทยาลัย: ราชภัฏเนครปฐม")

st.divider()

# ส่วนอธิบายรายละเอียดโปรเจกต์
st.subheader("📌 เกี่ยวกับโปรเจกต์")
st.markdown("""
ระบบนี้พัฒนาขึ้นเพื่อเป็นส่วนหนึ่งของวิชา **Machine Learning / Data Science** 
มุ่งเน้นการเปรียบเทียบประสิทธิภาพของโมเดล Machine Learning หลากหลายอัลกอริทึม ในการทำนายและประเมินความเสี่ยงโรคหัวใจจากข้อมูลสุขภาพของคนไข้
""")

st.subheader("🤖 อัลกอริทึมที่นำมาศึกษาและพัฒนาในระบบ")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    - 🎯 **K-Nearest Neighbor (KNN)**
    - 🌳 **ต้นไม้ตัดสินใจ (Decision Tree)**
    - ⚡ **Support Vector Machine (SVM)**
    """)

with col_b:
    st.markdown("""
    - 📊 **การแบ่งกลุ่มด้วย K-Means (K-Means Clustering)**
    - 📈 **การถดถอยโลจิสติก (Logistic Regression)**
    - 🌲 **ป่าสุ่มตัดสินใจ (Random Forest)**
    """)

st.divider()
st.info("💡 เลือกเมนูจากแถบข้าง (Sidebar) ด้านซ้าย เพื่อเริ่มต้นใช้งานหน้าอื่นๆ ของระบบ")