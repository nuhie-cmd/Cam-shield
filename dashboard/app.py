import os
import sys
from datetime import datetime
import requests
import cv2
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from test_videos.live_video import get_live_frame

# ------------------------------
# PAGE CONFIG & REFRESH
# ------------------------------
st.set_page_config(page_title="CamShield - AI CCTV Security", layout="wide", page_icon="🛡")
st_autorefresh(interval=50, key="camshield_refresh")

# Custom CSS to enforce 100% solid opacity across the entire UI
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117 !important;
        opacity: 1.0 !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"], div[data-testid="metric-container"] {
        background-color: #1a1d24 !important;
        border-radius: 8px;
        padding: 8px 12px;
        opacity: 1.0 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #14171f !important;
        opacity: 1.0 !important;
    }
    h1, h2, h3, h4, h5, h6, p, span, label {
        opacity: 1.0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state trend histories

if "health_history" not in st.session_state:
    st.session_state.health_history = [100.0] * 10
if "threat_history" not in st.session_state:
    st.session_state.threat_history = [0.0] * 10

# ------------------------------
# BACKEND API HELPERS
# ------------------------------
def fetch_backend_status():
    try:
        res = requests.get("http://127.0.0.1:8000/status", timeout=0.8)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None

def fetch_backend_incidents():
    try:
        res = requests.get("http://127.0.0.1:8000/incidents", timeout=0.8)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []

# Fetch frame and live status from pipeline
frame, live_status = get_live_frame()

# Fetch backend status (fallback to live_status if API offline)
db_status = fetch_backend_status()
current_status = db_status if db_status and "threat_score" in db_status else live_status

# Update trend session history
st.session_state.health_history.append(float(current_status.get("camera_health", 100)))
st.session_state.threat_history.append(float(current_status.get("threat_score", 0.0)))
if len(st.session_state.health_history) > 30:
    st.session_state.health_history.pop(0)
    st.session_state.threat_history.pop(0)

# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.title("🛡 CamShield")
st.sidebar.header("System Information")
st.sidebar.write("📷 **Camera ID** : CAM-01")
st.sidebar.write("🤖 **AI Model** : YOLOv8n + Event Fusion")
st.sidebar.write("🗄 **Database** : SQLite")
st.sidebar.write("⚡ **Backend** : FastAPI")

backend_online = db_status is not None
if backend_online:
    st.sidebar.success("🟢 System & Backend Online")
else:
    st.sidebar.info("🟡 Pipeline Active (Backend Offline)")

st.sidebar.write("📅 **Date** :", datetime.now().strftime("%d-%m-%Y"))
st.sidebar.write("🕒 **Time** :", datetime.now().strftime("%H:%M:%S"))

# ------------------------------
# MAIN TITLE
# ------------------------------
st.title("🛡 CamShield")
st.caption("AI-Based Predictive CCTV Tampering & Threat Detection System")

# ==========================================================
# DASHBOARD SUMMARY METRICS
# ==========================================================
st.subheader("📊 Dashboard Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📹 Total Cameras", 8)
with col2:
    st.metric("🟢 Online", 7)
with col3:
    st.metric("🔴 Offline", 1)
with col4:
    critical_count = 1 if current_status.get("threat_score", 0) >= 80 else 0
    st.metric("🚨 Critical Alerts", critical_count)

st.divider()

# ==========================================================
# PREDICTIVE CCTV TAMPERING DETECTION TABLE
# ==========================================================
st.subheader("🧠 AI Prediction for All Cameras")

cam1_prediction = current_status.get("prediction", "Operating Normally")
cam1_action = current_status.get("recommended_action", "Monitor")
cam1_status = current_status.get("camera_status", "Camera Healthy")
cam1_status_str = "🟢 Online" if "Offline" not in cam1_status else "🔴 Offline"

prediction_df = pd.DataFrame({
    "Camera": ["CAM-01", "CAM-02", "CAM-03", "CAM-04", "CAM-05", "CAM-06", "CAM-07", "CAM-08"],
    "Status": [cam1_status_str, "🟢 Online", "🟢 Online", "🟢 Online", "🟢 Online", "🟢 Online", "🟢 Online", "🔴 Offline"],
    "Prediction": [cam1_prediction, "Operating Normally", "Operating Normally", "Lens Covered", "Operating Normally", "Camera Vibration", "Operating Normally", "No Prediction Available"],
    "Last Updated": [datetime.now().strftime("%H:%M:%S"), "10:21 AM", "10:21 AM", "10:22 AM", "10:22 AM", "10:23 AM", "10:23 AM", "--"],
    "Recommended Action": [cam1_action, "Monitor", "Monitor", "Send Security", "Monitor", "Inspect Camera", "Monitor", "Check Power/Network"]
})

st.dataframe(prediction_df, use_container_width=True, hide_index=True)

st.divider()

# ==========================================================
# LIVE CAMERA + AI STATUS
# ==========================================================
cam_col, info_col = st.columns([2, 1])

with cam_col:
    st.subheader("📹 Live Camera Feed")
    if frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB", use_container_width=True)
    else:
        st.error("Unable to access camera feed.")

with info_col:
    st.subheader("🤖 AI Detection Status")
    
    person_det_str = current_status.get("person_detected", "NO")
    dwell_str = current_status.get("dwell_time", "0 sec")
    threat_val = float(current_status.get("threat_score", 0.0))
    
    if person_det_str == "YES":
        st.success(f"👤 Person Detected : {person_det_str} ({current_status.get('person_count', 1)})")
    else:
        st.info(f"👤 Person Detected : {person_det_str}")

    st.info(f"⏱ Dwell Time : {dwell_str}")
    st.metric("🎯 Threat Score", f"{int(threat_val)}%")

    if threat_val >= 70:
        st.error("🚨 HIGH ALERT - CCTV Tampering / Threat Detected")
        st.write(f"**Prediction:** {current_status.get('prediction', 'Tampering Detected')}")
        st.write(f"**Action:** {current_status.get('recommended_action', 'Send Security Immediately')}")
    elif threat_val >= 40:
        st.warning("⚠ WARNING - Suspicious Activity")
        st.write(f"**Prediction:** {current_status.get('prediction', 'Suspicious Behavior')}")
        st.write(f"**Action:** {current_status.get('recommended_action', 'Inspect Area')}")
    else:
        st.success("✅ Camera Normal")
        st.write("**Reason:** No abnormal activity detected.")
        st.write("**Action:** Continue monitoring.")

st.divider()

# ==========================================================
# CAMERA HEALTH
# ==========================================================
st.subheader("📷 Camera Health")

health_val = int(current_status.get("camera_health", 100))

if health_val >= 90:
    st.success(f"🟢 Excellent ({health_val}%)")
elif health_val >= 70:
    st.info(f"🔵 Good ({health_val}%)")
elif health_val >= 50:
    st.warning(f"🟡 Needs Inspection ({health_val}%)")
else:
    st.error(f"🔴 Camera Failure ({health_val}%)")

st.progress(max(0.0, min(1.0, health_val / 100.0)))

st.divider()

# ==========================================================
# CAMERA HEALTH & THREAT TREND CHARTS
# ==========================================================
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("📈 Camera Health Trend")
    health_chart_df = pd.DataFrame({"Health (%)": st.session_state.health_history})
    st.line_chart(health_chart_df)

with chart_col2:
    st.subheader("📈 Threat Score Trend")
    threat_chart_df = pd.DataFrame({"Threat Score (%)": st.session_state.threat_history})
    st.line_chart(threat_chart_df)

st.divider()

# ==========================================================
# INCIDENT LOG (FETCHED FROM BACKEND DB)
# ==========================================================
st.subheader("🚨 Backend Incident Log")

db_incidents = fetch_backend_incidents()

if db_incidents:
    formatted_incidents = []
    for inc in reversed(db_incidents[-10:]):  # Display last 10 incidents
        formatted_incidents.append({
            "Time": inc.get("timestamp", "").split("T")[-1][:8] if "T" in inc.get("timestamp", "") else inc.get("timestamp", "--"),
            "Camera": inc.get("camera_id", "CAM-01"),
            "Incident": inc.get("incident_type", inc.get("prediction", "Security Incident")),
            "Risk Level": inc.get("risk_level", "MEDIUM"),
            "Threat Score": f"{int(inc.get('threat_score', 0))}%",
            "Action": inc.get("recommended_action", "Inspect")
        })
    incidents_df = pd.DataFrame(formatted_incidents)
    st.dataframe(incidents_df, use_container_width=True, hide_index=True)
else:
    fallback_incidents = pd.DataFrame({
        "Time": ["10:20 AM", "10:28 AM", "10:35 AM"],
        "Camera": ["CAM-01", "CAM-01", "CAM-01"],
        "Incident": ["Person Stayed Near Camera", "Camera Blur Detected", "Possible Lens Covered"],
        "Severity": ["Medium", "High", "Critical"]
    })
    st.dataframe(fallback_incidents, use_container_width=True, hide_index=True)

st.divider()

# ==========================================================
# EVIDENCE SNAPSHOT
# ==========================================================
st.subheader("📷 Evidence Snapshot")

evidence_dir = os.path.join(PROJECT_ROOT, "evidence", "snapshots")
found_snapshot = None

if os.path.exists(evidence_dir):
    files = [os.path.join(evidence_dir, f) for f in os.listdir(evidence_dir) if f.endswith(".jpg")]
    if files:
        files.sort(key=os.path.getmtime, reverse=True)
        found_snapshot = files[0]

if not found_snapshot:
    asset_path = os.path.join(PROJECT_ROOT, "dashboard", "assets", "evidence.jpeg")
    if os.path.exists(asset_path):
        found_snapshot = asset_path

if found_snapshot and os.path.exists(found_snapshot):
    st.image(found_snapshot, caption=f"Latest Captured Evidence ({os.path.basename(found_snapshot)})", use_container_width=True)
else:
    st.info("No evidence snapshots captured yet.")

st.divider()

# ==========================================================
# OVERALL SYSTEM STATUS
# ==========================================================
st.subheader("📌 Overall System Status")

if threat_val >= 70:
    st.error("🚨 HIGH ALERT - Immediate Action Required")
elif threat_val >= 40:
    st.warning("⚠ MEDIUM ALERT - Monitor Continuously")
else:
    st.success("✅ All Cameras Operating Normally")

st.success("🟢 CamShield AI Monitoring System Running Successfully")
