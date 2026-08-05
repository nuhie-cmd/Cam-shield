from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from backend.database import engine, Base, get_db
from backend.models import Incident
from backend.schemas import IncidentCreate

Base.metadata.create_all(bind=engine)

app = FastAPI(title="CamShield Backend")


@app.get("/")
def home():
    return {
        "message": "CamShield Backend is Running"
    }


@app.get("/health")
def health():
    return {
        "status": "Backend Healthy"
    }


@app.post("/incident")
def create_incident(
    incident: IncidentCreate,
    db: Session = Depends(get_db)
):

    new_incident = Incident(

        timestamp=incident.timestamp,
        camera_id=incident.camera_id,

        health_score=incident.video_integrity.health_score,
        blur_detected=incident.video_integrity.blur_detected,
        tilt_detected=incident.video_integrity.tilt_detected,
        brightness_changed=incident.video_integrity.brightness_changed,

        person_detected=incident.person_behavior.person_detected,
        dwell_time_seconds=incident.person_behavior.dwell_time_seconds,
        approaching_camera=incident.person_behavior.approaching_camera,

        stream_disconnected=incident.camera_security.stream_disconnected,
        config_changed=incident.camera_security.config_changed,

        threat_score=incident.fusion_output.threat_score,
        risk_level=incident.fusion_output.risk_level,
        xai_explanation=incident.fusion_output.xai_explanation,

        alert_sent=False,
        evidence_path=""
    )

    db.add(new_incident)
    db.commit()
    db.refresh(new_incident)

    return {
        "message": "Incident Logged Successfully",
        "incident_id": new_incident.id
    }


@app.get("/incidents")
def get_incidents(db: Session = Depends(get_db)):
    return db.query(Incident).all()


@app.get("/status")
def get_latest_status(db: Session = Depends(get_db)):
    latest = db.query(Incident).order_by(Incident.id.desc()).first()

    if latest is None:
        return {
            "message": "No incidents found."
        }

    return latest