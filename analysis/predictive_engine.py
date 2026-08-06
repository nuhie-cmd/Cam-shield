"""
Predictive Engine Module for CamShield
Assigns a real-time risk score to each tracked person based on fused
security events and behavior history.

Pipeline:
PersonDetector -> Tracker -> BehaviorAnalyzer -> EventFusion -> PredictiveEngine
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Risk Levels
# ---------------------------------------------------------------------

RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"
RISK_CRITICAL = "critical"

_MEDIUM_SCORE = 30.0
_HIGH_SCORE = 60.0
_CRITICAL_SCORE = 85.0

# ---------------------------------------------------------------------
# Score Weights
# ---------------------------------------------------------------------

_EVENT_WEIGHTS: Dict[str, float] = {
    "loitering_alert": 35.0,
    "running_alert": 20.0,
    "crowd_surge": 15.0,
    "intrusion": 50.0,
    "idle_alert": 5.0,
}

_BEHAVIOR_SCORES: Dict[str, float] = {
    "idle": 0.0,
    "walking": 0.5,
    "running": 4.0,
    "loitering": 3.0,
    "unknown": 0.0,
}

_SCORE_DECAY_RATE = 8.0


# ---------------------------------------------------------------------
# Risk Profile
# ---------------------------------------------------------------------

class RiskProfile:
    """Stores risk information for one tracked person."""

    def __init__(self, track_id: int):
        self.track_id = track_id
        self.score = 0.0
        self.risk_level = RISK_LOW
        self.last_updated = time.time()
        self.event_log: List[Tuple[str, float]] = []

    def apply_event(self, event_type: str, weight: float):
        self.score = min(100.0, self.score + weight)
        self.event_log.append((event_type, time.time()))

        if len(self.event_log) > 20:
            self.event_log.pop(0)

        self._update_level()

    def apply_behavior(self, behavior: str):
        self.score = min(
            100.0,
            self.score + _BEHAVIOR_SCORES.get(behavior, 0.0)
        )
        self._update_level()

    def decay(self, now: float):
        elapsed = now - self.last_updated

        self.score = max(
            0.0,
            self.score - elapsed * _SCORE_DECAY_RATE
        )

        self.last_updated = now
        self._update_level()

    def _update_level(self):

        if self.score >= _CRITICAL_SCORE:
            self.risk_level = RISK_CRITICAL

        elif self.score >= _HIGH_SCORE:
            self.risk_level = RISK_HIGH

        elif self.score >= _MEDIUM_SCORE:
            self.risk_level = RISK_MEDIUM

        else:
            self.risk_level = RISK_LOW

    def to_dict(self):

        return {
            "track_id": self.track_id,
            "threat_score": round(self.score, 2),
            "risk_level": self.risk_level,
            "last_updated": self.last_updated,
        }


# ---------------------------------------------------------------------
# Predictive Engine
# ---------------------------------------------------------------------

class PredictiveEngine:

    def __init__(self, profile_ttl: float = 60.0):

        self.profile_ttl = profile_ttl
        self._profiles: Dict[int, RiskProfile] = {}

    def update(
        self,
        detections: Optional[List[Dict[str, Any]]],
        events: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:

        now = time.time()
        active_ids = set()

        # -------------------------------------------------------------
        # Update from behaviors
        # -------------------------------------------------------------

        for det in detections or []:

            if not isinstance(det, dict):
                continue

            track_id = det.get("id")

            if not isinstance(track_id, int):
                continue

            active_ids.add(track_id)

            profile = self._get_or_create_profile(track_id, now)

            profile.decay(now)

            profile.apply_behavior(
                det.get("behavior", "unknown")
            )

            det["threat_score"] = round(profile.score, 2)
            det["risk_level"] = profile.risk_level

        # -------------------------------------------------------------
        # Update from events
        # -------------------------------------------------------------

        for event in events or []:

            if isinstance(event, dict):

                event_type = event.get("event_type")
                track_id = event.get("track_id")

            else:

                event_type = getattr(event, "event_type", None)
                track_id = getattr(event, "track_id", None)

            weight = _EVENT_WEIGHTS.get(event_type, 10.0)

            if isinstance(track_id, int):

                profile = self._get_or_create_profile(track_id, now)

                profile.apply_event(event_type, weight)

                logger.info(
                    f"[RISK] Track {track_id} +{weight} "
                    f"({event_type}) -> {profile.score:.1f}"
                )

            else:

                for tid in active_ids:

                    profile = self._get_or_create_profile(
                        tid,
                        now
                    )

                    profile.apply_event(
                        event_type,
                        weight * 0.5
                    )
                # -------------------------------------------------------------
        # Cleanup stale profiles
        # -------------------------------------------------------------

        self._purge_stale_profiles(active_ids, now)

        report = sorted(
            [profile.to_dict() for profile in self._profiles.values()],
            key=lambda x: x["threat_score"],
            reverse=True,
        )

        return report

    # -----------------------------------------------------------------
    # Public helpers
    # -----------------------------------------------------------------

    def get_high_risk_tracks(self) -> List[Dict[str, Any]]:
        """Return only HIGH and CRITICAL risk tracks."""

        return [
            profile.to_dict()
            for profile in self._profiles.values()
            if profile.risk_level in (RISK_HIGH, RISK_CRITICAL)
        ]

    def reset(self) -> None:
        """Clear all stored profiles."""

        self._profiles.clear()
        logger.info("PredictiveEngine reset.")

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _get_or_create_profile(
        self,
        track_id: int,
        now: float,
    ) -> RiskProfile:

        if track_id not in self._profiles:

            profile = RiskProfile(track_id)
            profile.last_updated = now

            self._profiles[track_id] = profile

        return self._profiles[track_id]

    def _purge_stale_profiles(
        self,
        active_ids: set,
        now: float,
    ) -> None:

        stale_ids = []

        for track_id, profile in self._profiles.items():

            inactive_time = now - profile.last_updated

            if (
                track_id not in active_ids
                and inactive_time > self.profile_ttl
            ):
                stale_ids.append(track_id)

        for track_id in stale_ids:

            del self._profiles[track_id]

            logger.debug(
                f"Removed stale risk profile {track_id}"
            )


# ---------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    import os
    import sys

    sys.path.insert(
        0,
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        ),
    )

    from analysis.event_fusion import SecurityEvent

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    engine = PredictiveEngine()

    print("\n=== Predictive Engine Test ===\n")

    scenarios = [
        ("walking", []),
        ("walking", []),
        ("running", []),
        (
            "loitering",
            [
                SecurityEvent(
                    "loitering_alert",
                    1,
                    time.time(),
                    0.90,
                    {},
                )
            ],
        ),
        (
            "loitering",
            [
                SecurityEvent(
                    "loitering_alert",
                    1,
                    time.time(),
                    0.90,
                    {},
                )
            ],
        ),
    ]

    for frame_no, (behavior, events) in enumerate(scenarios, start=1):

        detections = [
            {
                "id": 1,
                "class": "person",
                "behavior": behavior,
                "confidence": 0.92,
                "speed_px_per_s": (
                    90.0 if behavior == "running" else 20.0
                ),
                "loitering_time": frame_no * 2,
            }
        ]

        report = engine.update(detections, events)

        print(
            f"Frame {frame_no} | "
            f"{behavior:<10} | "
            f"Score = {report[0]['threat_score']:.1f} | "
            f"Level = {report[0]['risk_level']}"
        )

    print("\nHigh Risk Tracks")
    print(engine.get_high_risk_tracks())

    print("\nDone.")            