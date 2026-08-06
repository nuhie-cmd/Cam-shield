from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from backend.gemini_service import generate_incident_summary
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

        confidence=incident.confidence,
        prediction=incident.prediction,
        recommended_action=incident.recommended_action,
        incident_type=incident.incident_type,
        camera_status=incident.camera_status,

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
            "timestamp": "",
            "camera_id": "CAM-01",
            "health_score": 100.0,
            "camera_health": 100,
            "blur_detected": False,
            "tilt_detected": False,
            "brightness_changed": False,
            "person_detected": "NO",
            "dwell_time": "0 sec",
            "dwell_time_seconds": 0.0,
            "approaching_camera": False,
            "stream_disconnected": False,
            "config_changed": False,
            "threat_score": 0.0,
            "risk_level": "LOW",
            "xai_explanation": "Operating Normally",
            "confidence": 1.0,
            "prediction": "Operating Normally",
            "recommended_action": "Monitor",
            "incident_type": "None",
            "camera_status": "ONLINE"
        }

    return {
        "id": latest.id,
        "timestamp": latest.timestamp,
        "camera_id": latest.camera_id,
        "health_score": latest.health_score,
        "camera_health": int(latest.health_score) if latest.health_score is not None else 100,
        "blur_detected": latest.blur_detected,
        "tilt_detected": latest.tilt_detected,
        "brightness_changed": latest.brightness_changed,
        "person_detected": "YES" if latest.person_detected else "NO",
        "dwell_time": f"{int(latest.dwell_time_seconds or 0)} sec",
        "dwell_time_seconds": latest.dwell_time_seconds,
        "approaching_camera": latest.approaching_camera,
        "stream_disconnected": latest.stream_disconnected,
        "config_changed": latest.config_changed,
        "threat_score": latest.threat_score,
        "risk_level": latest.risk_level,
        "xai_explanation": latest.xai_explanation,
        "confidence": latest.confidence,
        "prediction": latest.prediction,
        "recommended_action": latest.recommended_action,
        "incident_type": latest.incident_type,
        "camera_status": latest.camera_status
    }



@app.get("/stats")
def get_statistics(db: Session = Depends(get_db)):

    incidents = db.query(Incident).all()

    total_incidents = len(incidents)

    if total_incidents == 0:
        return {
            "total_incidents": 0,
            "critical_incidents": 0,
            "average_threat_score": 0
        }

    critical_incidents = len(
        [i for i in incidents if i.threat_score >= 80]
    )

    average_threat = sum(
        i.threat_score for i in incidents
    ) / total_incidents

    return {
        "total_incidents": total_incidents,
        "critical_incidents": critical_incidents,
        "average_threat_score": round(average_threat, 2)
    }


@app.get("/camera-security")
def camera_security():

    return {
        "stream_disconnected": False,
        "config_changed": False,
        "unauthorized_login": False,
        "firmware_changed": False,
        "network_attack": False,
        "security_status": "SECURE"
    }


@app.get("/alert")
def alert(db: Session = Depends(get_db)):

    latest = db.query(Incident).order_by(Incident.id.desc()).first()

    if latest is None:
        return {
            "alert": False,
            "message": "No incidents available."
        }

    if latest.threat_score >= 80:
        return {
            "alert": True,
            "level": "CRITICAL",
            "message": "Immediate operator intervention required."
        }

    elif latest.threat_score >= 50:
        return {
            "alert": True,
            "level": "HIGH",
            "message": "Potential camera tampering detected."
        }

    else:
        return {
            "alert": False,
            "level": "LOW",
            "message": "System operating normally."
        }


@app.get("/ai-summary")
def ai_summary(db: Session = Depends(get_db)):

    latest = db.query(Incident).order_by(Incident.id.desc()).first()

    if latest is None:
        return {
            "summary": "No incidents available."
        }

    risk = "LOW"

    if latest.threat_score >= 80:
        risk = "CRITICAL"
    elif latest.threat_score >= 50:
        risk = "HIGH"

    try:
        summary = generate_incident_summary(
            latest.threat_score,
            risk,
            latest.xai_explanation
        )

        return {
            "summary": summary
        }

    except Exception as e:
        return {
            "error": str(e)
        }