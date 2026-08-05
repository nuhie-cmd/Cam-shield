"""
Event Fusion Module for CamShield
Aggregates per-frame detection + behavior data into discrete security events
(e.g. "intrusion", "loitering_alert", "crowd_surge").

Designed to sit between BehaviorAnalyzer and the predictive engine / backend.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------
EVENT_LOITERING = "loitering_alert"
EVENT_RUNNING = "running_alert"
EVENT_CROWD_SURGE = "crowd_surge"
EVENT_INTRUSION = "intrusion"
EVENT_IDLE = "idle_alert"


@dataclass
class SecurityEvent:
    """A discrete security event produced by EventFusion."""

    event_type: str
    track_id: Optional[int]
    timestamp: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "track_id": self.track_id,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class EventFusion:
    """
    Fuses per-frame behavior-annotated detections into higher-level
    security events with de-duplication and cooldown logic.

    Usage
    -----
    >>> fusion = EventFusion()
    >>> events = fusion.process(tracked_and_analyzed_detections)
    """

    def __init__(
        self,
        loitering_confirm_frames: int = 3,
        running_confirm_frames: int = 2,
        crowd_surge_threshold: int = 5,
        event_cooldown: float = 10.0,
    ) -> None:
        """
        Parameters
        ----------
        loitering_confirm_frames : int
            Number of consecutive frames a loitering behavior must be seen
            before an event is emitted.
        running_confirm_frames : int
            Same, for running alerts.
        crowd_surge_threshold : int
            Minimum number of simultaneous tracked persons to trigger a
            crowd surge event.
        event_cooldown : float
            Seconds between repeated events of the same type for the same
            track (prevents alert flooding).
        """
        self.loitering_confirm_frames = loitering_confirm_frames
        self.running_confirm_frames = running_confirm_frames
        self.crowd_surge_threshold = crowd_surge_threshold
        self.event_cooldown = event_cooldown

        # track_id -> behavior -> consecutive frame count
        self._behavior_streak: Dict[int, Dict[str, int]] = {}
        # (track_id, event_type) -> last emission timestamp
        self._last_emitted: Dict[tuple, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, detections: Optional[List[Dict[str, Any]]]) -> List[SecurityEvent]:
        """
        Process a frame's worth of behavior-annotated detections and return
        any security events that should be raised.
        """
        events: List[SecurityEvent] = []
        if not detections:
            return events

        now = time.time()

        for det in detections:
            if not isinstance(det, dict):
                continue

            track_id = det.get("id")
            behavior = det.get("behavior", "unknown")
            confidence = float(det.get("confidence", 0.0))
            speed = float(det.get("speed_px_per_s", 0.0))
            loitering_time = float(det.get("loitering_time", 0.0))

            if not isinstance(track_id, int):
                continue

            # Update behavior streak counter
            streak = self._behavior_streak.setdefault(track_id, {})
            streak[behavior] = streak.get(behavior, 0) + 1
            # Reset streaks for other behaviors
            for b in list(streak.keys()):
                if b != behavior:
                    streak[b] = 0

            # --- Loitering alert ---
            if (
                behavior == "loitering"
                and streak["loitering"] >= self.loitering_confirm_frames
            ):
                ev = self._maybe_emit(
                    event_type=EVENT_LOITERING,
                    track_id=track_id,
                    confidence=min(confidence + 0.05, 1.0),
                    now=now,
                    metadata={
                        "loitering_time_s": loitering_time,
                        "speed_px_per_s": speed,
                    },
                )
                if ev:
                    events.append(ev)

            # --- Running alert ---
            if (
                behavior == "running"
                and streak.get("running", 0) >= self.running_confirm_frames
            ):
                ev = self._maybe_emit(
                    event_type=EVENT_RUNNING,
                    track_id=track_id,
                    confidence=confidence,
                    now=now,
                    metadata={"speed_px_per_s": speed},
                )
                if ev:
                    events.append(ev)

        # --- Crowd surge (frame-level, not per-track) ---
        active_tracks = [
            d for d in detections if isinstance(d, dict) and isinstance(d.get("id"), int)
        ]
        if len(active_tracks) >= self.crowd_surge_threshold:
            ev = self._maybe_emit(
                event_type=EVENT_CROWD_SURGE,
                track_id=None,
                confidence=1.0,
                now=now,
                metadata={"person_count": len(active_tracks)},
            )
            if ev:
                events.append(ev)

        # Log if any events fired
        for ev in events:
            logger.warning(
                f"[EVENT] {ev.event_type} | track={ev.track_id} "
                f"| conf={ev.confidence:.2f} | meta={ev.metadata}"
            )

        return events

    def reset(self) -> None:
        """Clear all internal state (e.g., scene change or camera restart)."""
        self._behavior_streak.clear()
        self._last_emitted.clear()
        logger.info("EventFusion state reset.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_emit(
        self,
        event_type: str,
        track_id: Optional[int],
        confidence: float,
        now: float,
        metadata: Dict[str, Any],
    ) -> Optional[SecurityEvent]:
        """Emit an event only if the cooldown has elapsed."""
        key = (track_id, event_type)
        last = self._last_emitted.get(key, 0.0)
        if now - last < self.event_cooldown:
            return None
        self._last_emitted[key] = now
        return SecurityEvent(
            event_type=event_type,
            track_id=track_id,
            timestamp=now,
            confidence=confidence,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    fusion = EventFusion(
        loitering_confirm_frames=2,
        running_confirm_frames=1,
        crowd_surge_threshold=3,
        event_cooldown=5.0,
    )

    print("=== EventFusion Smoke Test ===\n")

    # Frame 1: one person loitering
    frame1 = [
        {"id": 1, "behavior": "loitering", "confidence": 0.88,
         "speed_px_per_s": 8.0, "loitering_time": 12.0},
    ]
    events = fusion.process(frame1)
    print(f"Frame 1 events: {[e.to_dict() for e in events]}")

    # Frame 2: same person still loitering → should fire
    events = fusion.process(frame1)
    print(f"Frame 2 events: {[e.to_dict() for e in events]}")

    # Frame 3: running person
    frame3 = [{"id": 2, "behavior": "running", "confidence": 0.91, "speed_px_per_s": 120.0, "loitering_time": 0.0}]
    events = fusion.process(frame3)
    print(f"Frame 3 events: {[e.to_dict() for e in events]}")

    # Frame 4: crowd surge (3 people)
    frame4 = [
        {"id": 1, "behavior": "idle", "confidence": 0.80, "speed_px_per_s": 2.0, "loitering_time": 0.0},
        {"id": 2, "behavior": "walking", "confidence": 0.85, "speed_px_per_s": 30.0, "loitering_time": 0.0},
        {"id": 3, "behavior": "walking", "confidence": 0.83, "speed_px_per_s": 25.0, "loitering_time": 0.0},
    ]
    events = fusion.process(frame4)
    print(f"Frame 4 events: {[e.to_dict() for e in events]}")

    print("\nDone.")
