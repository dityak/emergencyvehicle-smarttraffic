
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib

# Placeholder for actual models
def load_cnn_model():
    return lambda img: np.random.choice([0, 1])  # 0: Non-emergency, 1: Emergency

def load_xgb_model():
    return lambda features: round(np.random.uniform(10, 60), 2)  # Random duration prediction

# Load models
cnn_model = load_cnn_model()
xgb_model = load_xgb_model()

st.set_page_config(page_title="Smart Traffic System", layout="centered")
st.title("🚦 Smart Traffic Light System")
st.markdown("Dual-model system: CNN for emergency vehicle detection and XGBoost for adaptive green signal prediction.")

# Image input for CNN
st.header("📷 Emergency Vehicle Detection (CNN)")
uploaded_img = st.file_uploader("Upload traffic image", type=["jpg", "jpeg", "png"])
if uploaded_img:
    img = Image.open(uploaded_img).resize((128, 128))
    st.image(img, caption="Uploaded Image", use_column_width=True)
    is_emergency = cnn_model(img)
    st.success(f"Prediction: {'🚨 Emergency Vehicle Detected' if is_emergency else '🟢 No Emergency Vehicle'}")

# Input for XGBoost
st.header("📊 Signal Duration Prediction (XGBoost)")
st.markdown("Enter traffic stats to estimate green signal duration.")

vehicle_count = st.slider("Total Vehicle Count", 0, 100, 30)
avg_speed = st.slider("Average Speed (km/h)", 0, 100, 35)
weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Foggy"])
accident_reported = st.checkbox("Accident Reported")
if uploaded_img:
    emergency_flag = 1 if is_emergency else 0
else:
    emergency_flag = st.selectbox("Emergency Flag (if no image)", [0, 1])

weather_map = {"Clear": 0, "Rainy": 1, "Foggy": 2}
input_data = pd.DataFrame([{
    "vehicle_count": vehicle_count,
    "avg_speed": avg_speed,
    "weather": weather_map[weather],
    "accident_reported": int(accident_reported),
    "emergency_flag": emergency_flag
}])

if st.button("Predict Signal Duration"):
    duration = xgb_model(input_data)
    st.success(f"⏱️ Recommended Green Signal Duration: {duration} seconds")
