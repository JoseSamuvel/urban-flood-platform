"""
app.py
Streamlit Native Web Application for Urban Flood Impact Classification & Preparedness Platform.
Compatible with Streamlit Community Cloud deployment.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import pydeck as pdk
import streamlit as st
from datetime import datetime

# Ensure project directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUB_DIR = os.path.join(BASE_DIR, "Internal Hackathon")
if os.path.exists(SUB_DIR) and SUB_DIR not in sys.path:
    sys.path.insert(0, SUB_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from flood_engine import FloodPreparednessEngine, HISTORICAL_BENCHMARKS

LOG_FILE = os.path.join(BASE_DIR, "Internal Hackathon", "flood_action_log.json")
if not os.path.exists(os.path.dirname(LOG_FILE)):
    LOG_FILE = os.path.join(BASE_DIR, "flood_action_log.json")

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Urban Flood Impact Classification & Preparedness Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom CSS for Dark Premium Aesthetic
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Dark Theme Custom Adjustments */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.8));
        padding: 1.5rem 2rem;
        border-radius: 16px;
        border: 1px solid rgba(56, 189, 248, 0.2);
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    
    .brand-title {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .brand-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.2rem;
    }

    /* KPI Cards */
    .kpi-card {
        padding: 1.2rem;
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
    }
    .kpi-val {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.2rem 0;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #94a3b8;
    }

    /* Preset Badge */
    .sdg-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 0.5rem;
    }
    .sdg-11 { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .sdg-13 { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .ml-live { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Initialize Engine & Helper Functions
# -----------------------------------------------------------------------------
@st.cache_resource
def get_ml_engine():
    return FloodPreparednessEngine()

try:
    engine = get_ml_engine()
except Exception as e:
    st.error(f"Failed to initialize ML Engine: {e}")
    st.stop()

def load_action_logs():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_action_log(entry):
    logs = load_action_logs()
    logs.insert(0, entry)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)
    return logs

# -----------------------------------------------------------------------------
# Header Component
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
        <div>
            <h1 class="brand-title">🌊 Urban Flood Impact Classification & Preparedness Platform</h1>
            <p class="brand-subtitle">Real-Time Hydrological ML Classification & Municipal Disaster Preparedness</p>
        </div>
        <div>
            <span class="sdg-badge sdg-11">🏙️ SDG 11: Sustainable Cities</span>
            <span class="sdg-badge sdg-13">⚡ SDG 13: Climate Action</span>
            <span class="sdg-badge ml-live">🟢 ML Engine Live</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar: Scenario Simulator
# -----------------------------------------------------------------------------
st.sidebar.header("🎛️ Scenario Simulator")
st.sidebar.caption("Adjust meteorological & infrastructural parameters to re-score urban flood risk in real-time.")

preset_options = {
    "☀️ Normal Seasonal Rain (25 mm)": {"rain": 25.0, "intensity": 6.0, "water": 1.0, "silt": 1.0, "discharge": 500.0},
    "🌧️ Monsoon Deluge Wave (95 mm)": {"rain": 95.0, "intensity": 22.0, "water": 1.3, "silt": 1.2, "discharge": 3200.0},
    "🌀 2021 Cyclone Michaung (210 mm)": {"rain": 210.0, "intensity": 42.0, "water": 1.8, "silt": 1.5, "discharge": 8500.0},
    "🚨 2015 Catastrophic Cloudburst (340 mm)": {"rain": 340.0, "intensity": 65.0, "water": 2.4, "silt": 1.9, "discharge": 15000.0}
}

selected_preset = st.sidebar.selectbox("Disaster Benchmark Presets:", list(preset_options.keys()), index=1)
preset_vals = preset_options[selected_preset]

# Sliders initialized with preset values
rainfall_24h = st.sidebar.slider("24h Rainfall (mm)", 5.0, 380.0, float(preset_vals["rain"]), step=5.0)
rainfall_intensity = st.sidebar.slider("Rain Intensity (mm/hr)", 2.0, 90.0, float(preset_vals["intensity"]), step=2.0)
water_level_factor = st.sidebar.slider("River/Canal Water Level Factor", 0.5, 2.8, float(preset_vals["water"]), step=0.1)
siltation_factor = st.sidebar.slider("Drain Siltation / Choke Factor", 0.5, 2.5, float(preset_vals["silt"]), step=0.1)
reservoir_discharge = st.sidebar.slider("Reservoir Discharge (cusecs)", 200.0, 18000.0, float(preset_vals["discharge"]), step=200.0)

# -----------------------------------------------------------------------------
# Run City Wards Simulation
# -----------------------------------------------------------------------------
wards_data = engine.simulate_city_wards(
    rainfall_24h=rainfall_24h,
    rainfall_intensity=rainfall_intensity,
    water_level_factor=water_level_factor,
    siltation_factor=siltation_factor,
    reservoir_discharge=reservoir_discharge
)

counts = {"Low": 0, "Moderate": 0, "High": 0, "Severe Inundation": 0}
for w in wards_data:
    label = w["prediction"]["impact_label"]
    for key in counts.keys():
        if key in label:
            counts[key] += 1
            break

# -----------------------------------------------------------------------------
# KPI Summary Row
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card" style="border-left: 4px solid #10b981;">
        <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600;">Low Impact Wards</div>
        <div class="kpi-val" style="color: #34d399;">{counts['Low']}</div>
        <div class="kpi-sub">Standard municipal runoff handling</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card" style="border-left: 4px solid #f59e0b;">
        <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600;">Moderate Advisory</div>
        <div class="kpi-val" style="color: #fbbf24;">{counts['Moderate']}</div>
        <div class="kpi-sub">10–30 cm arterial waterlogging</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card" style="border-left: 4px solid #f97316;">
        <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600;">High Risk Zones</div>
        <div class="kpi-val" style="color: #fb923c;">{counts['High']}</div>
        <div class="kpi-sub">Residential ingress; pump staging</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card" style="border-left: 4px solid #ef4444;">
        <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600;">Severe Inundation</div>
        <div class="kpi-val" style="color: #f87171;">{counts['Severe Inundation']}</div>
        <div class="kpi-sub">Immediate evacuation & boats</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Interactive Map View (Pydeck)
# -----------------------------------------------------------------------------
st.subheader("🗺️ City-Wide Inundation Hazard Map")

map_records = []
for w in wards_data:
    pred = w["prediction"]
    cls = pred["predicted_class"]
    
    # RGB color mapping
    if cls == 0:
        color = [16, 185, 129, 200]    # Green
        radius = 400
    elif cls == 1:
        color = [245, 158, 11, 200]   # Yellow
        radius = 550
    elif cls == 2:
        color = [249, 115, 22, 220]   # Orange
        radius = 700
    else:
        color = [239, 68, 68, 240]    # Red
        radius = 900
        
    map_records.append({
        "ward_id": w["ward_id"],
        "ward_name": w["ward_name"],
        "zone": w["zone"],
        "lat": w["latitude"],
        "lon": w["longitude"],
        "elevation": w["elevation_m"],
        "land_use": w["land_use"].replace("_", " "),
        "impact_label": pred["impact_label"],
        "confidence": f"{pred['confidence'] * 100:.1f}%",
        "summary": pred["summary"],
        "action": pred["action_protocol"][0],
        "color": color,
        "radius": radius
    })

df_map = pd.DataFrame(map_records)

# Pydeck Scatterplot Layer
scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_map,
    get_position=["lon", "lat"],
    get_color="color",
    get_radius="radius",
    pickable=True,
    opacity=0.85,
    stroked=True,
    filled=True,
    radius_min_pixels=10,
    radius_max_pixels=30,
    line_width_min_pixels=2,
    get_line_color=[255, 255, 255, 220]
)

view_state = pdk.ViewState(
    latitude=13.04,
    longitude=80.22,
    zoom=10.5,
    pitch=25
)

tooltip_html = {
    "html": "<b>{ward_name} ({ward_id})</b><br>"
            "<b>Impact:</b> {impact_label} ({confidence})<br>"
            "<b>Zone:</b> {zone} | <b>Elevation:</b> {elevation}m<br>"
            "<b>Land Use:</b> {land_use}<br>"
            "<b>Action:</b> {action}",
    "style": {
        "backgroundColor": "#0f172a",
        "color": "#f8fafc",
        "fontSize": "12px",
        "borderRadius": "8px",
        "padding": "10px",
        "border": "1px solid #38bdf8"
    }
}

st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v11" if os.environ.get("MAPBOX_API_KEY") else "carto-dark",
        initial_view_state=view_state,
        layers=[scatter_layer],
        tooltip=tooltip_html
    )
)

# -----------------------------------------------------------------------------
# Diagnostic Tabs Section
# -----------------------------------------------------------------------------
tab_matrix, tab_history, tab_ml, tab_logs = st.tabs([
    "📋 Ward Risk Matrix & Actions",
    "🕰️ Historical Event Benchmarking",
    "📈 ML Validation & Threshold Curves",
    "🚨 Emergency Action Audit Log"
])

# Tab 1: Ward Risk Matrix
with tab_matrix:
    st.markdown("#### Ward Inundation Risk Matrix")
    matrix_rows = []
    for w in wards_data:
        pred = w["prediction"]
        matrix_rows.append({
            "Ward ID": w["ward_id"],
            "Ward Name": w["ward_name"],
            "Zone": w["zone"],
            "Elevation (m)": w["elevation_m"],
            "Drain Cap (%)": f"{w['inputs']['drain_capacity_pct']:.0f}%",
            "Land Use": w["land_use"].replace("_", " "),
            "Predicted Impact": pred["impact_label"],
            "Confidence": f"{pred['confidence'] * 100:.1f}%",
            "Primary Action Directive": pred["action_protocol"][0]
        })
    df_matrix = pd.DataFrame(matrix_rows)
    st.dataframe(df_matrix, use_container_width=True, hide_index=True)

# Tab 2: Historical Benchmarking
with tab_history:
    st.markdown("#### Historical Event Benchmarking Comparison")
    hist_comps = engine.compare_with_historical_events(
        current_rainfall=rainfall_24h,
        current_water_level=2.5 * water_level_factor
    )
    df_hist = pd.DataFrame(hist_comps)
    df_hist = df_hist.rename(columns={
        "event": "Benchmark Event",
        "year": "Year",
        "historical_rainfall_mm": "Benchmarked Rain (mm)",
        "rainfall_variance_mm": "Current Rain Variance (mm)",
        "match_assessment": "Hydrological Match Assessment",
        "description": "Historical Context"
    })
    st.dataframe(df_hist, use_container_width=True, hide_index=True)

# Tab 3: ML Validation & Threshold Curves
with tab_ml:
    st.markdown("#### Model Architecture & Validation Artifacts")
    
    col_img1, col_img2, col_img3 = st.columns(3)
    
    out_dir = os.path.join(BASE_DIR, "output_models")
    if not os.path.exists(out_dir):
        out_dir = os.path.join(BASE_DIR, "Internal Hackathon", "output_models")
        
    thresh_img = os.path.join(out_dir, "threshold_analysis.png")
    fi_img = os.path.join(out_dir, "feature_importance.png")
    cm_img = os.path.join(out_dir, "confusion_matrix.png")
    
    with col_img1:
        if os.path.exists(thresh_img):
            st.image(thresh_img, caption="Rainfall Threshold Sensitivity", use_container_width=True)
        else:
            st.info("Threshold Analysis plot generated during training.")
            
    with col_img2:
        if os.path.exists(fi_img):
            st.image(fi_img, caption="Model Feature Importance", use_container_width=True)
        else:
            st.info("Feature Importance plot generated during training.")
            
    with col_img3:
        if os.path.exists(cm_img):
            st.image(cm_img, caption="Confusion Matrix (96.3% Accuracy)", use_container_width=True)
        else:
            st.info("Confusion Matrix plot generated during training.")
            
    # Model Metadata Display
    meta_path = os.path.join(out_dir, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        st.markdown(f"**Best Model**: `{meta.get('best_model_name')}` | **Training Samples**: `{meta.get('training_samples')}` | **Test Samples**: `{meta.get('test_samples')}`")

# Tab 4: Emergency Action Audit Log
with tab_logs:
    st.markdown("#### Log & Audit Official Municipal Preparedness Directives")
    
    with st.form("action_log_form", clear_on_submit=True):
        st.markdown("##### 🚨 Log New Preparedness Directive")
        
        ward_names = [f"{w['ward_id']} - {w['ward_name']}" for w in wards_data]
        ward_names.insert(0, "CITY-WIDE - City-Wide General Action")
        
        f_target = st.selectbox("Target Location / Ward", ward_names)
        f_unit = st.selectbox("Authorized Unit", [
            "NDRF / SDRF Taskforce",
            "Municipal Stormwater Engineering",
            "Traffic Management & Police",
            "Disaster Shelter & Relief Operations",
            "Public Works Dewatering Team"
        ])
        f_category = st.selectbox("Action Category", [
            "Precautionary Dewatering Pump Dispatch",
            "Traffic Diversion / Road Closure",
            "Emergency Inundation Evacuation",
            "Relief Shelter & Food Distribution Setup",
            "Storm Drain Grate De-clogging"
        ])
        f_notes = st.text_area("Directive Details / Operator Notes", placeholder="Enter equipment IDs, deployed personnel count, or specific directives...")
        
        submitted = st.form_submit_button("🚨 Save & Dispatch Directive")
        if submitted:
            entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "target": f_target,
                "unit": f_unit,
                "category": f_category,
                "notes": f_notes if f_notes else "Standard protocol dispatched."
            }
            save_action_log(entry)
            st.success(f"Directive logged successfully for {f_target}!")

    st.markdown("##### 📜 Official Disaster Response Audit Trail")
    logs = load_action_logs()
    if logs:
        df_logs = pd.DataFrame(logs)
        st.dataframe(df_logs, use_container_width=True, hide_index=True)
    else:
        st.info("No emergency action logs recorded yet.")
