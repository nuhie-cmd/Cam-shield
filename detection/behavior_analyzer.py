"""
Behavior Analyzer Module for CamShield
Classifies per-person behaviors (idle, walking, running, loitering) by
analyzing the motion history maintained for each tracked ID.

Designed to consume the output of Tracker.update() and produce labeled
detection dicts that downstream modules (event_fusion, predictive_engine)
can act on.
"""

import logging
import math
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Behavior labels
# ---------------------------------------------------------------------------
BEHAVIOR_IDLE = "idle"
BEHAVIOR_WALKING = "walking"
BEHAVIOR_RUNNING = "running"
BEHAVIOR_LOITERING = "loitering"
BEHAVIOR_UNKNOWN = "unknown"

# Type alias
Detection = Dict[str, Any]


class MotionHistory:
    """Circular buffer that records centroid positions and timestamps for one track."""

    def __init__(self, maxlen: int = 30) -> None:
        self._positions: Deque[Tuple[float, float, float]] = deque(maxlen=maxlen)

    def push(self, x: float, y: float, ts: float) -> None:
        self._positions.append((x, y, ts))

    @property
    def positions(self) -> List[Tuple[float, float, float]]:
        return list(self._positions)

    @property
    def length(self) -> int:
        return len(self._positions)

    def total_displacement(self) -> float:
        """Straight-line distance between oldest and newest position."""
        if self.length < 2:
            return 0.0
        x0, y0, _ = self._positions[0]
        x1, y1, _ = self._positions[-1]
        return math.hypot(x1 - x0, y1 - y0)

    def path_length(self) -> float:
        """Sum of step distances — total path travelled."""
        total = 0.0
        pts = self._positions
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            total += math.hypot(dx, dy)
        return total

    def average_speed_px_per_s(self) -> float:
        """Mean speed in pixels/second over the buffered window."""
        if self.length < 2:
            return 0.0
        dt = self._positions[-1][2] - self._positions[0][2]
        if dt <= 0:
            return 0.0
        return self.path_length() / dt

    def time_in_buffer(self) -> float:
        """Duration (seconds) covered by the current buffer."""
        if self.length < 2:
            return 0.0
        return self._positions[-1][2] - self._positions[0][2]


class BehaviorAnalyzer:
    """
    Stateful analyzer that classifies the behavior of each tracked person.

    Call ``analyze(tracked_detections)`` once per frame after
    ``Tracker.update()``.  It returns the same list with a ``"behavior"``
    key added to every detection that carries a valid track ``"id"``.
    """

    def __init__(
        self,
        idle_speed_threshold: float = 5.0,
        walking_speed_threshold: float = 60.0,
        loitering_duration: float = 8.0,
        loitering_displacement_threshold: float = 40.0,
        history_maxlen: int = 30,
    ) -> None:
        """
        Parameters
        ----------
        idle_speed_threshold : float
            Speed (px/s) below which a person is considered idle.
        walking_speed_threshold : float
            Speed (px/s) above idle but below which is classed as walking;
            anything above is running.
        loitering_duration : float
            Minimum seconds a person must be tracked before loitering is checked.
        loitering_displacement_threshold : float
            Maximum net displacement (px) over ``loitering_duration`` seconds
            to be classified as loitering rather than idle/walking.
        history_maxlen : int
            Number of frames kept per track in the motion history buffer.
        """
        self.idle_speed_threshold = idle_speed_threshold
        self.walking_speed_threshold = walking_speed_threshold
        self.loitering_duration = loitering_duration
        self.loitering_displacement_threshold = loitering_displacement_threshold
        self.history_maxlen = history_maxlen

        # track_id -> MotionHistory
        self._histories: Dict[int, MotionHistory] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, detections: Optional[List[Detection]]) -> List[Detection]:
        """
        Classify behaviors for a list of tracked detections.

        Each detection dict is mutated in-place to include:
        - ``"behavior"``        : str label (idle / walking / running / loitering)
        - ``"speed_px_per_s"``  : float average speed over history window
        - ``"loitering_time"``  : float seconds the track has been observed

        Returns the same list for convenience.
        """
        if not detections:
            self._purge_stale_histories(set())
            return detections or []

        active_ids: set = set()

        for det in detections:
            if not isinstance(det, dict):
                continue

            track_id = det.get("id")
            if not isinstance(track_id, int):
                det["behavior"] = BEHAVIOR_UNKNOWN
                det["speed_px_per_s"] = 0.0
                det["loitering_time"] = 0.0
                continue

            active_ids.add(track_id)

            center = det.get("center")
            ts = float(det.get("timestamp", time.time()))

            if not (isinstance(center, (list, tuple)) and len(center) == 2):
                det["behavior"] = BEHAVIOR_UNKNOWN
                det["speed_px_per_s"] = 0.0
                det["loitering_time"] = 0.0
                continue

            history = self._get_or_create_history(track_id)
            history.push(float(center[0]), float(center[1]), ts)

            behavior, speed = self._classify(history)
            det["behavior"] = behavior
            det["speed_px_per_s"] = round(speed, 2)
            det["loitering_time"] = round(history.time_in_buffer(), 2)

        self._purge_stale_histories(active_ids)
        return detections

    def reset(self) -> None:
        """Clear all motion histories (e.g., on camera switch or scene reset)."""
        self._histories.clear()
        logger.info("BehaviorAnalyzer state reset.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_history(self, track_id: int) -> MotionHistory:
        if track_id not in self._histories:
            self._histories[track_id] = MotionHistory(maxlen=self.history_maxlen)
        return self._histories[track_id]

    def _classify(self, history: MotionHistory) -> Tuple[str, float]:
        """Return (behavior_label, speed_px_per_s) for the given history."""
        speed = history.average_speed_px_per_s()
        duration = history.time_in_buffer()
        displacement = history.total_displacement()

        # Loitering: long duration, lots of movement (path) but little net displacement
        if (
            duration >= self.loitering_duration
            and displacement <= self.loitering_displacement_threshold
            and history.path_length() > self.loitering_displacement_threshold * 0.5
        ):
            return BEHAVIOR_LOITERING, speed

        if speed < self.idle_speed_threshold:
            return BEHAVIOR_IDLE, speed
        if speed < self.walking_speed_threshold:
            return BEHAVIOR_WALKING, speed
        return BEHAVIOR_RUNNING, speed

    def _purge_stale_histories(self, active_ids: set) -> None:
        """Remove histories for tracks that are no longer active."""
        stale = [tid for tid in self._histories if tid not in active_ids]
        for tid in stale:
            del self._histories[tid]
            logger.debug(f"Purged behavior history for track ID: {tid}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time as _time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    analyzer = BehaviorAnalyzer(
        idle_speed_threshold=5.0,
        walking_speed_threshold=60.0,
        loitering_duration=4.0,
        loitering_displacement_threshold=30.0,
    )

    print("=== BehaviorAnalyzer Smoke Test ===\n")

    # Simulate a person walking across the frame over 10 frames
    detections = []
    base_ts = _time.time()
    for frame_idx in range(10):
        x = 100.0 + frame_idx * 20.0   # moving right ~20px per frame
        y = 200.0
        ts = base_ts + frame_idx * 0.033  # ~30 fps
        detections.append({
            "id": 1,
            "class": "person",
            "bbox": [x - 30, y - 60, x + 30, y + 60],
            "center": [x, y],
            "confidence": 0.92,
            "timestamp": ts,
        })
        result = analyzer.analyze([detections[-1]])
        print(
            f"Frame {frame_idx:02d} | center=({x:.0f},{y:.0f}) "
            f"| behavior={result[0].get('behavior'):<10} "
            f"| speed={result[0].get('speed_px_per_s'):.1f} px/s"
        )

    print("\n--- Simulating loitering (small displacement, long duration) ---")
    analyzer.reset()
    import math as _math
    for frame_idx in range(50):
        angle = frame_idx * 0.3
        x = 300.0 + 10.0 * _math.cos(angle)  # pacing in tight circle
        y = 300.0 + 10.0 * _math.sin(angle)
        ts = base_ts + frame_idx * 0.2  # slow 5fps
        det = {
            "id": 2,
            "class": "person",
            "bbox": [x - 30, y - 60, x + 30, y + 60],
            "center": [x, y],
            "confidence": 0.88,
            "timestamp": ts,
        }
        result = analyzer.analyze([det])
        if frame_idx % 10 == 9:
            print(
                f"Frame {frame_idx:02d} | center=({x:.1f},{y:.1f}) "
                f"| behavior={result[0].get('behavior'):<10} "
                f"| loitering_time={result[0].get('loitering_time'):.1f}s"
            )

    print("\nDone.")
