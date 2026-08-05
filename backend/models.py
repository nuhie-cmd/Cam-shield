from sqlalchemy import Column, Integer, Float, String, Boolean
from backend.database import Base


class Incident(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True)

    timestamp = Column(String)
    camera_id = Column(String)

    # Video Integrity
    health_score = Column(Float)
    blur_detected = Column(Boolean)
    tilt_detected = Column(Boolean)
    brightness_changed = Column(Boolean)

    # Person Behaviour
    person_detected = Column(Boolean)
    dwell_time_seconds = Column(Float)
    approaching_camera = Column(Boolean)

    # Camera Security
    stream_disconnected = Column(Boolean)
    config_changed = Column(Boolean)

    # Threat Engine
    threat_score = Column(Float)
    risk_level = Column(String)
    xai_explanation = Column(String)

    # Evidence & Alerts
    alert_sent = Column(Boolean)
    evidence_path = Column(String)