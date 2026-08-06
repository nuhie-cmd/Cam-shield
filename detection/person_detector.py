"""
Person Detector Module for CamShield
Handles person-only object detection using Ultralytics YOLOv8n.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Configure module-level logger
logger = logging.getLogger(__name__)

# Module-level constants
PERSON_CLASS_ID = 0
PERSON_CLASS_NAME = "person"
_BOX_COLOR = (0, 255, 0)
_BOX_THICKNESS = 2
_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE = 0.5
_LABEL_THICKNESS = 1

# Type alias for clarity
PersonDetection = Dict[str, Any]


class PersonDetector:
    """
    A specialized Computer Vision detector that uses YOLOv8n to identify
    and extract bounding boxes for people in video frames.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu"
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model: Optional[YOLO] = None

        self.load_model()

    def load_model(self) -> None:
        """Load the YOLO model into memory."""
        try:
            logger.info(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            # Run a dummy prediction to warm up the model
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy_img, verbose=False, device=self.device)
            logger.info("YOLO model loaded and warmed up successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"Critical error initializing PersonDetector: {e}") from e

    @staticmethod
    def _prepare_frame(frame: np.ndarray) -> Optional[np.ndarray]:
        """Validate and normalize a frame for OpenCV / YOLO inference."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        # Ensure the array is contiguous in memory (required for some CV2/YOLO ops)
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        # Ensure it has 3 channels (convert grayscale if necessary)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    @staticmethod
    def _geometry_from_bbox(bbox: List[float]) -> Tuple[List[float], float, float, float]:
        """Derive center, width, height, and area from an ``[x1, y1, x2, y2]`` box."""
        x1, y1, x2, y2 = bbox
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        area = width * height
        return center, width, height, area

    def detect(self, frame: Optional[np.ndarray]) -> List[PersonDetection]:
        """Process an incoming image frame and detect human targets."""
        prepared_frame = self._prepare_frame(frame)
        if prepared_frame is None or self.model is None:
            return []

        try:
            results = self.model.predict(
                source=prepared_frame,
                classes=[PERSON_CLASS_ID],
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )

            detections: List[PersonDetection] = []
            if not results or len(results) == 0 or results[0].boxes is None:
                return []

            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            frame_timestamp = time.time()

            for bbox, confidence, class_id in zip(xyxy, confidences, class_ids):
                if int(class_id) != PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = (float(v) for v in bbox)
                bbox_list = [x1, y1, x2, y2]
                center, width, height, area = self._geometry_from_bbox(bbox_list)
                
                # Cleaned up duplicate dictionary keys
                detections.append(
                    {
                        "id": None,
                        "class": PERSON_CLASS_NAME,
                        "confidence": round(float(confidence), 4),
                        "bbox": bbox_list,
                        "center": center,
                        "width": round(width, 4),
                        "height": round(height, 4),
                        "area": round(area, 4),
                        "timestamp": frame_timestamp,
                    }
                )

            return detections

        except Exception as e:
            logger.error(f"Error during detection inference: {e}")
            return []

    @staticmethod
    def _normalize_detection(det: Dict[str, Any]) -> PersonDetection:
        """Coerce a loose dict into the canonical PersonDetection shape."""
        bbox_raw = det.get("bbox", [0.0, 0.0, 0.0, 0.0])
        bbox = [float(v) for v in bbox_raw[:4]]
        while len(bbox) < 4:
            bbox.append(0.0)

        center, width, height, area = PersonDetector._geometry_from_bbox(bbox)

        # Cleaned up duplicate dictionary keys
        return {
            "id": det.get("id"),
            "class": PERSON_CLASS_NAME,
            "confidence": round(float(det.get("confidence", 0.0)), 4),
            "bbox": bbox,
            "center": [float(v) for v in det.get("center", center)],
            "width": round(float(det.get("width", width)), 4),
            "height": round(float(det.get("height", height)), 4),
            "area": round(float(det.get("area", area)), 4),
            "timestamp": float(det.get("timestamp", time.time())),
        }

    def get_persons(self, detections: List[Dict[str, Any]]) -> List[PersonDetection]:
        """Filter a detection list to entries classified as ``person``."""
        persons: List[PersonDetection] = []
        for det in detections:
            if not isinstance(det, dict) or det.get("class") != PERSON_CLASS_NAME:
                continue
            persons.append(self._normalize_detection(det))
        return persons

    def get_people_count(self, detections: List[Dict[str, Any]]) -> int:
        """Return the number of detections in the current frame."""
        return len(self.get_persons(detections))

    def draw_boxes(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw bounding boxes and confidence labels onto the frame."""
        if frame is None or not isinstance(frame, np.ndarray):
            return frame

        prepared = self._prepare_frame(frame)
        if prepared is None:
            return frame

        # Re-use the caller's buffer when it is already a valid contiguous BGR array.
        canvas = frame if prepared is frame else prepared

        for det in detections:
            # Ensure bbox has 4 elements
            bbox = det.get("bbox", [0, 0, 0, 0])
            if len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = (int(v) for v in bbox)
            confidence = det.get("confidence", 0.0)
            label = f"{PERSON_CLASS_NAME} {confidence:.2f}"

            cv2.rectangle(canvas, (x1, y1), (x2, y2), _BOX_COLOR, _BOX_THICKNESS)

            (label_w, label_h), baseline = cv2.getTextSize(
                label, _LABEL_FONT, _LABEL_SCALE, _LABEL_THICKNESS
            )
            label_y1 = max(y1, label_h + baseline + 4)
            
            # Background rectangle for text
            cv2.rectangle(
                canvas,
                (x1, label_y1 - label_h - baseline - 4),
                (x1 + label_w, label_y1),
                _BOX_COLOR,
                cv2.FILLED,
            )
            # Text label
            cv2.putText(
                canvas,
                label,
                (x1, label_y1 - baseline - 2),
                _LABEL_FONT,
                _LABEL_SCALE,
                (0, 0, 0),
                _LABEL_THICKNESS,
                cv2.LINE_AA,
            )

        return canvas


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Standalone smoke test: blank 640x480 BGR frame (no persons expected).
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    try:
        detector = PersonDetector(confidence_threshold=0.5, device="cpu")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Model load failed (expected if yolov8n.pt is missing): {exc}")
        raise SystemExit(1) from exc

    detections_output = detector.detect(blank_frame)
    print("\n--- Detection Results ---")
    print(json.dumps(detections_output, indent=2))

    annotated = detector.draw_boxes(blank_frame.copy(), detections_output)
    print("\n--- Integrity Checks ---")
    print(f"Annotated frame shape: {annotated.shape}, dtype: {annotated.dtype}")
    print(f"Person count via get_persons(): {len(detector.get_persons(detections_output))}")
    print(f"People count via get_people_count(): {detector.get_people_count(detections_output)}")