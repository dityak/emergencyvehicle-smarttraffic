
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import os
import time

# Set Streamlit page configuration
st.set_page_config(page_title="Smart Traffic AI", layout="wide", initial_sidebar_state="expanded")

# Styling and theme
st.markdown(
    '''
    <style>
    body {
        background-color: #121212;
        color: white;
    }
    .stApp {
        background-color: #121212;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .card {
        background-color: #1e1e1e;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 8px rgba(255,255,255,0.05);
    }
    .stButton>button {
        color: white;
        background-color: #e50914;
        border-radius: 10px;
        font-weight: bold;
        padding: 0.6rem 1.2rem;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# Load models (update these paths with real model files)
@st.cache_resource
def load_cnn_model():
    try:
        return joblib.load("cnn_model.joblib")
    except:
        return lambda img: np.random.choice([0, 1])

@st.cache_resource
def load_xgb_model():
    try:
        return joblib.load("xgb_model.joblib")
    except:
        return lambda features: round(np.random.uniform(10, 60), 2)

cnn_model = load_cnn_model()
xgb_model = load_xgb_model()

# Add a logo and title
col1, col2 = st.columns([1, 8])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/7434/7434501.png", width=70)
with col2:
    st.title("Smart Traffic Management System 🚦")

# Tabs
tab1, tab2 = st.tabs(["🚨 Emergency Detection", "📊 Signal Duration Prediction"])

with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Upload Traffic Image for Emergency Detection")
    uploaded_img = st.file_uploader("Upload traffic image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    is_emergency = None

    if uploaded_img:
        img = Image.open(uploaded_img).resize((128, 128))
        st.image(img, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Analyzing image..."):
            time.sleep(1)
            is_emergency = cnn_model(img)
            st.success("🚨 Emergency Vehicle Detected!" if is_emergency else "🟢 No Emergency Vehicle")
    else:
        st.info("Please upload an image to detect emergency vehicle.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Signal Duration Prediction")

    vehicle_count = st.slider("🚗 Total Vehicle Count", 0, 100, 30)
    avg_speed = st.slider("⚡ Average Speed (km/h)", 0, 100, 35)
    weather = st.selectbox("🌦️ Weather Condition", ["Clear", "Rainy", "Foggy"])
    accident_reported = st.checkbox("🚧 Accident Reported")
    if uploaded_img and is_emergency is not None:
        emergency_flag = 1 if is_emergency else 0
    else:
        emergency_flag = st.selectbox("🚨 Emergency Flag (if no image)", [0, 1])

    weather_map = {"Clear": 0, "Rainy": 1, "Foggy": 2}
    input_data = pd.DataFrame([{
        "vehicle_count": vehicle_count,
        "avg_speed": avg_speed,
        "weather": weather_map[weather],
        "accident_reported": int(accident_reported),
        "emergency_flag": emergency_flag
    }])

    if st.button("🧠 Predict Signal Duration"):
        with st.spinner("Running XGBoost model..."):
            time.sleep(1)
            duration = xgb_model(input_data)
            st.success(f"⏱️ Recommended Green Signal Duration: {duration} seconds")

    st.markdown("</div>", unsafe_allow_html=True)
