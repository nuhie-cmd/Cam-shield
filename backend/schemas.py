from pydantic import BaseModel


class VideoIntegrity(BaseModel):
    health_score: float
    blur_detected: bool
    tilt_detected: bool
    brightness_changed: bool


class PersonBehavior(BaseModel):
    person_detected: bool
    dwell_time_seconds: float
    approaching_camera: bool


class CameraSecurity(BaseModel):
    stream_disconnected: bool
    config_changed: bool


class FusionOutput(BaseModel):
    threat_score: float
    risk_level: str
    xai_explanation: str


class IncidentCreate(BaseModel):
    timestamp: str
    camera_id: str

    video_integrity: VideoIntegrity
    person_behavior: PersonBehavior
    camera_security: CameraSecurity
    fusion_output: FusionOutput

    confidence: float
    prediction: str
    recommended_action: str
    incident_type: str
    camera_status: str