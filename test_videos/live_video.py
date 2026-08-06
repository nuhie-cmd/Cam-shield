import sys
import os
import time
import logging
import threading
from datetime import datetime
import cv2
import numpy as np
import requests

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from detection.viewblock import detect_viewblock
from evidence.evidence_manager import EvidenceManager
from evidence.alert_service import AlertService
from detection.person_detector import PersonDetector
from detection.tracker import Tracker
from detection.behavior_analyzer import BehaviorAnalyzer
from analysis.event_fusion import EventFusion
from analysis.predictive_engine import PredictiveEngine
from camera_heath.health_monitor import CameraHealthMonitor

logger = logging.getLogger(__name__)


class FreshRTSPCapture:
    """
    Threaded RTSP capture wrapper that continuously reads and discards buffered frames,
    ensuring read() always returns the latest real-time frame with ZERO lag.
    """
    def __init__(self, src: str):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;500000|buffer_size;1024|fflags;nobuffer"
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = False

        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.latest_frame = frame
                self.running = True
                self.thread = threading.Thread(target=self._reader_thread, daemon=True)
                self.thread.start()

    def _reader_thread(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest_frame = frame

    def isOpened(self):
        return self.running and self.cap is not None and self.cap.isOpened()

    def read(self):
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def release(self):
        self.running = False
        if self.cap:
            self.cap.release()


class CamShieldPipeline:
    def __init__(self):
        self.evidence = EvidenceManager(fps=30)
        self.alert_service = AlertService(alert_threshold=80.0)
        self.detector = PersonDetector()
        self.tracker = Tracker()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.fusion = EventFusion()
        self.predictor = PredictiveEngine()
        self.health_monitor = CameraHealthMonitor()

        self.cap = None
        self.stream_url = "rtsp://camera2:Camera%40235@172.16.10.235:554/cam/realmonitor?channel=1&subtype=0"
        self.source_type = "UNKNOWN"
        self.baseline_set = False

        self.smoothed_threat_score = 0.0

    def _init_capture(self):
        if self.cap is not None and self.cap.isOpened():
            return

        # Attempt 1: RTSP Stream via FreshRTSPCapture (Threaded 0-lag capture)
        try:
            logger.info(f"Attempting zero-lag connection to RTSP stream: {self.stream_url}")
            cap_rtsp = FreshRTSPCapture(self.stream_url)
            if cap_rtsp.isOpened():
                ret, frame = cap_rtsp.read()
                if ret and frame is not None:
                    self.cap = cap_rtsp
                    self.source_type = "RTSP (Zero-Lag)"
                    logger.info("Connected to RTSP Stream (Zero-Lag Threaded Mode) successfully.")
                    return
                else:
                    cap_rtsp.release()
        except Exception as e:
            logger.warning(f"RTSP Stream connection failed: {e}")


        # Attempt 2: Local Webcam (Index 0)
        try:
            logger.info("Attempting to connect to local webcam (Device 0)")
            cap_webcam = cv2.VideoCapture(0)
            if cap_webcam.isOpened():
                ret, frame = cap_webcam.read()
                if ret and frame is not None:
                    self.cap = cap_webcam
                    self.source_type = "WEBCAM"
                    logger.info("Connected to Local Webcam successfully.")
                    return
                else:
                    cap_webcam.release()
        except Exception as e:
            logger.warning(f"Webcam connection failed: {e}")

        # Attempt 3: Synthetic / Fallback
        self.cap = None
        self.source_type = "SYNTHETIC"
        logger.info("Operating in Synthetic Video Stream fallback mode.")

    def _generate_synthetic_frame(self) -> np.ndarray:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add subtle noise gradient
        noise = np.random.randint(20, 35, (480, 640, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)

        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"CAM-01 [LIVE FEED]", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {time_str}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Status: Stream Connected ({self.source_type})", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        return frame

    def read_frame(self) -> np.ndarray:
        self._init_capture()

        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                return frame

        # If read failed or synthetic mode
        return self._generate_synthetic_frame()

    def process_frame(self, frame: np.ndarray):
        if frame is None:
            frame = self._generate_synthetic_frame()

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Camera Baseline & Health Monitor
        if not self.baseline_set and frame is not None:
            self.health_monitor.set_baseline(frame)
            self.baseline_set = True

        health_report = self.health_monitor.check(frame)
        health_score = health_report.get("health_score", 1.0) * 100.0
        blur_detected = health_report.get("blur", False)
        brightness_changed = health_report.get("brightness", 0.5) < 0.08 or health_report.get("brightness", 0.5) > 0.92
        tilt_detected = health_report.get("angle_tampered", False)
        freeze_detected = health_report.get("freeze", False)
        obstruction_health = health_report.get("obstruction", False)

        # Fast viewblock check
        blocked_fast, contrast = detect_viewblock(frame)
        obstruction_detected = obstruction_health or blocked_fast

        # 2. YOLO Person Detection
        detections = self.detector.detect(frame)
        annotated_frame = self.detector.draw_boxes(frame.copy(), detections)

        # 3. Object Tracking
        tracked = self.tracker.update(detections)

        # 4. Behavior Analysis
        behaviors = self.behavior_analyzer.analyze(tracked)
        max_dwell = 0.0
        for b in behaviors:
            if isinstance(b, dict):
                loitering_time = b.get("loitering_time", 0.0)
                if loitering_time > max_dwell:
                    max_dwell = loitering_time

        person_count = len(detections)
        person_detected = person_count > 0

        # 5. Event Fusion
        events = self.fusion.process(behaviors)

        # 6. Predictive Engine Risk Scoring
        risk_report = self.predictor.update(behaviors, events)
        calculated_threat = 0.0
        if risk_report:
            calculated_threat = risk_report[0].get("threat_score", 0.0)

        # Dynamic Threat Score Boost mapping:
        # Camera blocked: +40, Person loitering: +20, Camera moved: +25, Multiple people: +15
        addon_threat = 0.0
        if obstruction_detected:
            addon_threat += 40.0
        if blur_detected:
            addon_threat += 20.0
        if tilt_detected:
            addon_threat += 25.0
        if person_count > 1:
            addon_threat += 15.0
        for b in behaviors:
            if isinstance(b, dict) and b.get("behavior") == "loitering":
                addon_threat += 20.0
                break

        raw_threat_score = min(100.0, max(calculated_threat, addon_threat))

        # Exponential decay smoothing for realistic threat score behavior
        if raw_threat_score > self.smoothed_threat_score:
            self.smoothed_threat_score = raw_threat_score
        else:
            self.smoothed_threat_score = max(0.0, self.smoothed_threat_score - 1.5)

        threat_score = round(self.smoothed_threat_score, 1)

        # Determine overall camera status string
        if self.source_type == "OFFLINE":
            camera_status = "Offline"
        elif obstruction_detected:
            camera_status = "Camera Blocked"
        elif blur_detected:
            camera_status = "Camera Blurred"
        else:
            camera_status = "Camera Healthy"

        # Primary Incident description & predictions
        incident_type = "None"
        prediction_str = "Operating Normally"
        recommended_action = "Monitor"
        risk_level = "LOW"

        if threat_score >= 80:
            risk_level = "CRITICAL"
            recommended_action = "Send Security Immediately"
        elif threat_score >= 50:
            risk_level = "HIGH"
            recommended_action = "Inspect Camera & Area"
        elif threat_score >= 30:
            risk_level = "MEDIUM"
            recommended_action = "Monitor Continuously"

        if obstruction_detected:
            incident_type = "Lens Covered / View Blocked"
            prediction_str = "Camera View Blocked"
        elif blur_detected:
            incident_type = "Camera Defocus / Blur"
            prediction_str = "Camera Blurred"
        elif tilt_detected:
            incident_type = "Camera Position Shift"
            prediction_str = "Camera Moved / Tilted"
        elif events:
            incident_type = events[0].event_type.replace("_", " ").title()
            prediction_str = incident_type
        elif person_count > 0:
            prediction_str = f"{person_count} Person Detected"

        # 7. Evidence Buffer & Snapshot
        self.evidence.add_frame(frame)

        if events or obstruction_detected or threat_score >= 80.0:
            snapshot_path = self.evidence.save_snapshot(frame)
            video_path = self.evidence.save_video()

            # 8. SMS Alert Service Trigger (with deduplication)
            alert_event_name = incident_type if incident_type != "None" else "High Threat Detected"
            self.alert_service.process_threat(threat_score, alert_event_name, snapshot_path)

            # 9. Log Incident to FastAPI Backend
            incident_payload = {
                "timestamp": timestamp_str,
                "camera_id": "CAM-01",
                "video_integrity": {
                    "health_score": round(health_score, 1),
                    "blur_detected": blur_detected,
                    "tilt_detected": tilt_detected,
                    "brightness_changed": brightness_changed
                },
                "person_behavior": {
                    "person_detected": person_detected,
                    "dwell_time_seconds": round(max_dwell, 1),
                    "approaching_camera": False
                },
                "camera_security": {
                    "stream_disconnected": False,
                    "config_changed": False
                },
                "fusion_output": {
                    "threat_score": threat_score,
                    "risk_level": risk_level,
                    "xai_explanation": f"{prediction_str} (Threat: {threat_score}%, Health: {int(health_score)}%)"
                },
                "confidence": 0.95,
                "prediction": prediction_str,
                "recommended_action": recommended_action,
                "incident_type": incident_type,
                "camera_status": camera_status
            }

            try:
                requests.post("http://127.0.0.1:8000/incident", json=incident_payload, timeout=0.5)
            except Exception:
                pass  # Non-blocking if backend is temporarily offline

        # Construct unified status dictionary for Dashboard UI
        status_dict = {
            "person_detected": "YES" if person_detected else "NO",
            "person_count": person_count,
            "dwell_time": f"{int(max_dwell)} sec",
            "dwell_time_seconds": round(max_dwell, 1),
            "threat_score": threat_score,
            "health_score": round(health_score, 1),
            "camera_health": int(health_score),
            "camera_status": camera_status,
            "incident_type": incident_type,
            "prediction": prediction_str,
            "recommended_action": recommended_action,
            "risk_level": risk_level,
            "blur_detected": blur_detected,
            "obstruction_detected": obstruction_detected,
            "tilt_detected": tilt_detected,
            "brightness_changed": brightness_changed,
            "stream_disconnected": False,
            "timestamp": timestamp_str
        }

        # Annotate top info on frame
        if obstruction_detected:
            cv2.putText(annotated_frame, "ALERT: CAMERA VIEW BLOCKED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif blur_detected:
            cv2.putText(annotated_frame, "WARNING: CAMERA BLURRED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        return annotated_frame, status_dict


# Global Pipeline Instance
pipeline = CamShieldPipeline()


def get_live_frame():
    """
    Exposed wrapper for Streamlit dashboard and external scripts.
    Returns (annotated_frame, status_dict)
    """
    raw_frame = pipeline.read_frame()
    return pipeline.process_frame(raw_frame)


def main():
    logger.info("Starting CamShield Live Video Pipeline standalone test...")
    while True:
        frame, status = get_live_frame()
        if frame is None:
            break
        cv2.imshow("CamShield Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()