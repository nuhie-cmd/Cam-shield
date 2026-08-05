import streamlit as st
import cv2
import pandas as pd
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
from fake_data import status, incidents
import os

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="CamShield", layout="wide")
st_autorefresh(interval=3000, key="refresh")

# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.title("🛡 CamShield")

st.sidebar.header("System Information")
st.sidebar.write("📷 Camera ID : CAM-01")
st.sidebar.write("🤖 AI Model : YOLOv8n")
st.sidebar.write("🗄 Database : SQLite")
st.sidebar.write("⚡ Backend : FastAPI")
st.sidebar.success("🟢 System Online")

st.sidebar.write("📅 Date :", datetime.now().strftime("%d-%m-%Y"))
st.sidebar.write("🕒 Time :", datetime.now().strftime("%H:%M:%S"))

# ------------------------------
# TITLE
# ------------------------------
st.title("🛡 CamShield")
st.caption("AI Based Predictive CCTV Tampering Detection System")

# ==========================================================
# DASHBOARD SUMMARY
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
    st.metric("🚨 Critical Alerts", 2)

st.divider()

# ==========================================================
# PREDICTIVE CCTV TAMPERING DETECTION
# ==========================================================

st.subheader("🧠 AI Prediction for All Cameras")

prediction_df = pd.DataFrame({

    "Camera": [
        "CAM-01",
        "CAM-02",
        "CAM-03",
        "CAM-04",
        "CAM-05",
        "CAM-06",
        "CAM-07",
        "CAM-08"
    ],

    "Status": [
        "🟢 Online",
        "🟢 Online",
        "🟢 Online",
        "🟢 Online",
        "🟢 Online",
        "🟢 Online",
        "🟢 Online",
        "🔴 Offline"
    ],

    "Prediction": [
        "Operating Normally",
        "Suspicious Activity",
        "Operating Normally",
        "Lens Covered",
        "Operating Normally",
        "Camera Vibration",
        "Operating Normally",
        "No Prediction Available"
    ],

    "Last Updated": [
        "10:20 AM",
        "10:21 AM",
        "10:21 AM",
        "10:22 AM",
        "10:22 AM",
        "10:23 AM",
        "10:23 AM",
        "--"
    ],

    "Recommended Action": [
        "Monitor",
        "Inspect Camera",
        "Monitor",
        "Send Security",
        "Monitor",
        "Inspect Camera",
        "Monitor",
        "Check Power/Network"
    ]

})

st.dataframe(
    prediction_df,
    use_container_width=True,
    hide_index=True
)
# ==========================================================
# LIVE CAMERA + AI STATUS
# ==========================================================

cam_col, info_col = st.columns([2, 1])

with cam_col:

    st.subheader("📹 Live Camera Feed")

    camera = cv2.VideoCapture(0)

    ret, frame = camera.read()

    if ret:

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        st.image(
            frame,
            channels="RGB",
            use_container_width=True
        )

    else:

        st.error("Unable to access webcam.")

    camera.release()

with info_col:

    st.subheader("🤖 AI Detection Status")

    st.success(f"👤 Person Detected : {status['person_detected']}")

    st.info(f"⏱ Dwell Time : {status['dwell_time']}")

    st.metric(
        "🎯 Threat Score",
        f"{status['threat_score']}%"
    )

    if status["threat_score"] >= 70:

        st.error("🚨 Possible CCTV Tampering")

        st.write("**Reason:** Camera lens appears blocked.")

        st.write("**Action:** Send security personnel immediately.")

    elif status["threat_score"] >= 40:

        st.warning("⚠ Suspicious Behaviour")

        st.write("**Reason:** Person stayed near camera for long time.")

        st.write("**Action:** Inspect the CCTV.")

    else:

        st.success("✅ Camera Normal")

        st.write("**Reason:** No abnormal activity detected.")

        st.write("**Action:** Continue monitoring.")

st.divider()

# =========================================================
# CAMERA HEALTH
# ==========================================================

st.subheader("📷 Camera Health")

if status["camera_health"] >= 90:

    st.success("🟢 Excellent")

elif status["camera_health"] >= 70:

    st.info("🔵 Good")

elif status["camera_health"] >= 50:

    st.warning("🟡 Needs Inspection")

else:

    st.error("🔴 Camera Failure")

st.progress(status["camera_health"] / 100)

st.divider()
# ==========================================================
# CAMERA HEALTH TREND
# ==========================================================

st.subheader("📈 Camera Health Trend")

health_data = pd.DataFrame({
    "Health (%)": [95, 94, 92, 91, 90, 92, status["camera_health"]]
})

st.line_chart(health_data)

st.divider()

# ==========================================================
# THREAT SCORE TREND
# ==========================================================

st.subheader("📈 Threat Score Trend")

threat_data = pd.DataFrame({
    "Threat Score": [15, 22, 30, 42, 55, 63, status["threat_score"]]
})

st.line_chart(threat_data)

st.divider()

# ==========================================================
# INCIDENT LOG
# ==========================================================

st.subheader("🚨 Incident Log")

incident_table = pd.DataFrame({
    "Time": [
        "09:10 AM",
        "09:35 AM",
        "10:05 AM"
    ],
    "Camera": [
        "CAM-01",
        "CAM-02",
        "CAM-01"
    ],
    "Incident": [
        "Person Stayed Near Camera",
        "Camera Vibration",
        "Possible Lens Covered"
    ],
    "Severity": [
        "Medium",
        "Medium",
        "High"
    ]
})

st.dataframe(
    incident_table,
    use_container_width=True,
    hide_index=True
)

st.divider()

# ==========================================================
# EVIDENCE SNAPSHOT
# ==========================================================

st.subheader("📷 Evidence Snapshot")

image_path = os.path.join(
    os.path.dirname(__file__),
    "assets",
    "evidence.jpeg"
)

if os.path.exists(image_path):

    st.image(
        image_path,
        caption="Latest Captured Evidence",
        use_container_width=True
    )

else:

    st.warning("Evidence image not found.")

st.divider()

# ==========================================================
# SYSTEM STATUS
# ==========================================================

st.subheader("📌 Overall System Status")

if status["threat_score"] >= 70:

    st.error("🚨 HIGH ALERT - Immediate Action Required")

elif status["threat_score"] >= 40:

    st.warning("⚠ MEDIUM ALERT - Monitor Continuously")

else:

    st.success("✅ All Cameras Operating Normally")

st.success("🟢 CamShield AI Monitoring System Running Successfully")