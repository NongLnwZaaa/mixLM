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
    st.image("https://scontent.fbkk31-1.fna.fbcdn.net/v/t39.30808-6/397864360_2110730222603303_7087100876165517819_n.jpg?stp=dst-jpg_tt6&cstp=mx959x960&ctp=s959x960&_nc_cat=106&ccb=1-7&_nc_sid=6ee11a&_nc_ohc=7tO5552GG-cQ7kNvwHbP5pv&_nc_oc=AdoRQAu1RwBVric8QeTlCLvQsjg7A3y0t3YZW-e5CrpQxjJQgZnyAgt7y7Stw1Vil5wWu7VREqjjh-09loXH-s5_&_nc_zt=23&_nc_ht=scontent.fbkk31-1.fna&_nc_gid=jh6YWjQ-kWUKjg_3jaFbEg&_nc_ss=7b2a8&oh=00_AQA-W8--ixzIgSa0va_Sok49FeiRsM71ATJYceBVyJWbhg&oe=6A6F9855", width=180)

with col2:
    st.subheader("ชื่อผู้พัฒนา")
    st.markdown("นาย กรภัทร์ ถิ่นผาแดง")
    st.write("รหัสนักศึกษา: 664245016")
    st.write("สาขาวิชา: วิทยาการคอมพิวเตอรร์")
    st.write("มหาวิทยาลัย: ราชภัฏนครปฐม")

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