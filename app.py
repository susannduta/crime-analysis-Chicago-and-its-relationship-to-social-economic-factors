import joblib
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# ==============================================================================
# 1. PAGE CONFIG & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Chicago Crime Predictor 🌸✨",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern cards (Removed global background to fix the white text bug!)
st.markdown("""
<style>
    /* Cute Header Banner */
    .cute-header {
        background: linear-gradient(135deg, #FF9A9E 0%, #FECFEF 100%);
        padding: 24px;
        border-radius: 20px;
        color: #2D2D2D;
        text-align: center;
        box-shadow: 0px 8px 16px rgba(255, 154, 158, 0.2);
        margin-bottom: 25px;
    }
    .cute-header h1 { color: #2D2D2D !important; font-weight: 800; margin: 0; font-size: 2.2rem; }
    .cute-header p { color: #2D2D2D !important; font-size: 1.05rem; margin-top: 8px; margin-bottom: 0; }

    /* Result Card Styling - Enforcing text colors so they never turn white */
    .result-card-violent {
        background: #FFF0F2;
        border: 2px solid #FF85A1;
        border-radius: 18px;
        padding: 20px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(255, 133, 161, 0.15);
    }
    .result-card-violent h2, .result-card-violent h3, .result-card-violent p {
        color: #D90429 !important;
    }
    
    .result-card-safe {
        background: #F0FFF4;
        border: 2px solid #70E000;
        border-radius: 18px;
        padding: 20px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(112, 224, 0, 0.15);
    }
    .result-card-safe h2, .result-card-safe h3, .result-card-safe p {
        color: #2B9348 !important;
    }
    
    /* Info Box */
    .info-box {
        background: #F8F9FA;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #EAEAEA;
        margin-top: 15px;
        color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. LOAD BUNDLE (WITH BULLETPROOF FAILSAFES)
# ==============================================================================
@st.cache_resource
def load_bundle():
    try:
        return joblib.load('streamlit_deployment_bundle.pkl')
    except FileNotFoundError:
        st.error("⚠️ Asset bundle `streamlit_deployment_bundle.pkl` not found!")
        st.stop()

bundle = load_bundle()

# FAILSAFE: Find ANY valid model in the dictionary even if the names don't match
available_models = [v for k, v in bundle.items() if hasattr(v, 'predict')]

if not available_models:
    st.error("❌ No valid ML model found inside your .pkl file!")
    st.stop()

# Assign models safely
model_dt = bundle.get('model_dt') or bundle.get('model') or available_models[0]
model_lr = bundle.get('model_lr') or available_models[0]
scaler = bundle.get('scaler')
encoder = bundle.get('encoder')
feature_names = bundle.get('features', ['latitude', 'longitude', 'hour', 'month', 'day_encoded'])

# Default location (Downtown Chicago) if not set
if 'lat' not in st.session_state:
    st.session_state.lat = 41.8781
if 'lon' not in st.session_state:
    st.session_state.lon = -87.6298


# ==============================================================================
# 3. HEADER & TITLE
# ==============================================================================
st.markdown("""
<div class="cute-header">
    <h1>🌸 Chicago Public Safety & Crime Risk Explorer ✨</h1>
    <p>Click anywhere on the interactive map below to predict localized crime risk!</p>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 4. SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.markdown("### ⚙️ Dispatch Settings")

selected_model = st.sidebar.radio(
    "🤖 Active AI Model:",
    options=["Decision Tree (High Sensitivity)", "Logistic Regression (Linear)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🕒 Date & Time Inputs")

hour = st.sidebar.slider("Hour of Day (24-Hour Clock)", 0, 23, 14, format="%d:00")
month = st.sidebar.selectbox("Month", options=list(range(1, 13)), format_func=lambda x: ['Jan ❄️', 'Feb 💌', 'Mar 🌿', 'Apr 🌸', 'May 🌺', 'Jun ☀️', 'Jul 🎆', 'Aug 🌻', 'Sep 🍂', 'Oct 🎃', 'Nov 🍁', 'Dec 🎄'][x-1])
day_encoded = st.sidebar.selectbox("Day of Week", options=list(range(7)), format_func=lambda x: ['Monday ☕', 'Tuesday 🌮', 'Wednesday 🐪', 'Thursday 🌿', 'Friday 🎉', 'Saturday 🎈', 'Sunday ☀️'][x])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📍 Manual Coordinates Override")
lat_input = st.sidebar.number_input("Latitude", value=float(st.session_state.lat), format="%.4f", step=0.005)
lon_input = st.sidebar.number_input("Longitude", value=float(st.session_state.lon), format="%.4f", step=0.005)

st.session_state.lat = lat_input
st.session_state.lon = lon_input


# ==============================================================================
# 5. MAIN CONTENT - INTERACTIVE MAP & PREDICTION
# ==============================================================================
col_left, col_right = st.columns([1.2, 1], gap="large")

with col_left:
    st.markdown("### 🗺️ Click on Chicago Map to Select Location")
    
    m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=11, tiles="CartoDB positron")
    
    folium.Marker(
        [st.session_state.lat, st.session_state.lon],
        popup="Selected Incident Point",
        tooltip="Selected Point",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    map_data = st_folium(m, height=450, width="100%")

    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lng = map_data["last_clicked"]["lng"]
        
        if clicked_lat != st.session_state.lat or clicked_lng != st.session_state.lon:
            st.session_state.lat = clicked_lat
            st.session_state.lon = clicked_lng
            st.rerun()

with col_right:
    st.markdown("### 🔮 Predicted Incident Category")
    
    # Create input dictionary matching feature names
    input_dict = {
        'latitude': st.session_state.lat,
        'longitude': st.session_state.lon,
        'hour': hour,
        'month': month,
        'day_encoded': day_encoded
    }
    
    # 1. Format raw input dataframe
    input_df = pd.DataFrame([[input_dict.get(col, 0) for col in feature_names]], columns=feature_names)

    # 2. Apply scaler properly if available
    if scaler is not None:
        try:
            scaled_array = scaler.transform(input_df)
            input_data = pd.DataFrame(scaled_array, columns=feature_names)
        except Exception:
            input_data = scaler.transform(input_df)
    else:
        input_data = input_df

    # 3. Select active model safely
    active_model = model_dt if "Decision Tree" in selected_model else model_lr

    # 4. Perform prediction using correctly formatted input
    try:
        pred_prob = active_model.predict_proba(input_data)[0]
        prediction = active_model.predict(input_data)[0]
    except Exception:
        # Fallback if model was trained without feature names
        pred_prob = active_model.predict_proba(input_df)[0]
        prediction = active_model.predict(input_df)[0]

    # Handle binary probability array output
    violent_prob = pred_prob[1] if len(pred_prob) > 1 else pred_prob[0]

    # Render risk alert card based on calculated risk
    if violent_prob >= 0.5:
        st.markdown(f"""
        <div class="result-card-violent">
            <h2 style="margin: 0;">🚨 High Risk Incident</h2>
            <h3 style="margin-top: 10px;">Violent Crime Likely</h3>
            <p style="font-size: 1.2rem; font-weight: bold;">
                Estimated Probability: {violent_prob * 100:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card-safe">
            <h2 style="margin: 0;">🟢 Lower Risk Incident</h2>
            <h3 style="margin-top: 10px;">Property / Other Crime Likely</h3>
            <p style="font-size: 1.2rem; font-weight: bold;">
                Violent Risk Likelihood: {violent_prob * 100:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.write("**Risk Level Gauge:**")
    st.progress(float(min(max(violent_prob, 0.0), 1.0)))
    
    # Active inputs breakdown
    st.markdown(f"""
    <div class="info-box">
        <b>📍 Selected Location:</b> {st.session_state.lat:.4f}, {st.session_state.lon:.4f}<br>
        <b>⏰ Time:</b> {hour:02d}:00 HRS<br>
        <b>📅 Schedule:</b> Month {month}, Day {day_encoded}
    </div>
    """, unsafe_allow_html=True)