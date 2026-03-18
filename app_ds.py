import streamlit as st
import pandas as pd
import joblib
import json
import os
import sklearn

# 1. การตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="BMW Price Predictor",
    page_icon="🚗",
    layout="wide"
)


# 2. ฟังก์ชันโหลดโมเดลจากโฟลเดอร์ model_artifacts
@st.cache_resource
def load_all_artifacts():
    try:
        # หาตำแหน่งโฟลเดอร์ปัจจุบันที่ไฟล์ app_ds.py วางอยู่
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # ชี้ไปยังโฟลเดอร์ย่อย model_artifacts ที่เก็บไฟล์โมเดลไว้
        artifacts_dir = os.path.join(current_dir, "model_artifacts")

        # กำหนด Path ของไฟล์ทั้ง 3 ตัว
        model_path = os.path.join(artifacts_dir, "bmw_pipeline.pkl")
        feature_path = os.path.join(artifacts_dir, "feature_names_bmw.json")
        meta_path = os.path.join(artifacts_dir, "model_metadata_bmw.json")

        # เริ่มโหลดไฟล์
        model = joblib.load(model_path)
        with open(feature_path, "r") as f:
            features = json.load(f)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        return model, features, meta
    except Exception as e:
        # แจ้งเตือนหากหาไฟล์ไม่เจอ
        st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
        return None, None, None


# เรียกใช้งานฟังก์ชันโหลดข้อมูล
pipeline, feature_names, metadata = load_all_artifacts()

# 3. ส่วนการแสดงผลบนหน้าเว็บ (UI)
st.title("🚗 BMW Price Predictor")
st.write(f"ระบบทำนายราคารถยนต์ BMW | scikit-learn v{sklearn.__version__}")

if pipeline is not None:
    # สร้างฟอร์มรับข้อมูล
    with st.form("bmw_prediction_form"):
        st.subheader("กรอกรายละเอียดรถยนต์เพื่อประเมินราคา")
        col1, col2, col3 = st.columns(3)

        with col1:
            model_car = st.selectbox("เลือกรุ่นรถ (Model)",
                                     ["X5", "1 Series", "X1", "7 Series", "3 Series", "5 Series", "X3", "X7", "i7"])
            year = st.number_input("ปีที่จดทะเบียน (Year)", 2010, 2026, 2022)
            hp = st.number_input("แรงม้า (Horsepower)", 100, 800, 250)
            engine = st.number_input("ขนาดเครื่องยนต์ (Engine Size)", 1.0, 6.6, 2.0)

        with col2:
            mileage = st.number_input("เลขไมล์ (Mileage KM)", 0, 500000, 30000)
            fuel = st.selectbox("ประเภทเชื้อเพลิง", ["diesel", "petrol", "hybrid", "electric"])
            transmission = st.selectbox("ระบบเกียร์", ["automatic", "manual"])
            color = st.selectbox("สีรถ", ["white", "black", "blue", "grey", "silver"])

        with col3:
            owner = st.number_input("ลำดับเจ้าของ", 1, 10, 1)
            accident = st.selectbox("ประวัติอุบัติเหตุ", ["no", "yes"])
            service = st.selectbox("ประวัติเช็คศูนย์", ["full", "partial", "none"])
            country = st.selectbox("ประเทศที่วางจำหน่าย", ["Germany", "UK", "USA", "Thailand"])

        submit_btn = st.form_submit_button("💰 ทำนายราคาประเมิน")

    # 4. ส่วนประมวลผลเมื่อกดปุ่มทำนาย
    if submit_btn:
        # เตรียมข้อมูลดิบให้ครบ 19 คอลัมน์ตามที่โมเดลต้องการ
        raw_data = {
            'car_id': 0,
            'model': model_car,
            'year': year,
            'engine_size': engine,
            'horsepower': hp,
            'fuel_type': fuel,
            'transmission': transmission,
            'drivetrain': 'AWD',
            'mileage_km': mileage,
            'fuel_consumption_l_per_100km': 7.5,
            'co2_emissions_g_km': 140,
            'doors': 4,
            'seats': 5,
            'body_type': 'SUV' if "X" in model_car else 'sedan',
            'color': color,
            'owner_count': owner,
            'accident_history': accident,
            'service_history': service,
            'country_sold': country
        }

        try:
            # จัดเรียงคอลัมน์ให้ตรงตามลำดับใน feature_names_bmw.json เป๊ะๆ
            input_df = pd.DataFrame([raw_data])[feature_names]

            # ทำนายราคา
            price = pipeline.predict(input_df)[0]

            # แสดงผลลัพธ์
            st.divider()
            st.balloons()
            st.success("✨ ประเมินราคาสำเร็จ!")

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label="ราคาประเมิน (USD)", value=f"${price:,.2f}")
            with res_col2:
                # คำนวณเป็นเงินไทย (สมมติอัตราแลกเปลี่ยน 35 บาท)
                thb_price = price * 35
                st.metric(label="ราคาประเมิน (THB)", value=f"฿{thb_price:,.0f}")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการคำนวณ: {e}")
            if "_RemainderColsList" in str(e):
                st.warning("⚠️ แนะนำให้รันคำสั่ง: pip install --upgrade scikit-learn ใน Terminal")
else:
    st.warning("⚠️ ไม่พบไฟล์ในโฟลเดอร์ model_artifacts โปรดตรวจสอบว่าไฟล์ทั้ง 3 อยู่ในโฟลเดอร์นั้นครบถ้วน")