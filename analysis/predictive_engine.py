"""
Predictive Engine Module for CamShield
Assigns a real-time risk score to each tracked person based on fused
security events and behavior history.

Sits at the top of the pipeline:
  PersonDetector -> Tracker -> BehaviorAnalyzer -> EventFusion -> PredictiveEngine
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risk levels
# ---------------------------------------------------------------------------
RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"
RISK_CRITICAL = "critical"

# Thresholds
_MEDIUM_SCORE = 30.0
_HIGH_SCORE = 60.0
_CRITICAL_SCORE = 85.0

# Event weights (how much each event type raises score)
_EVENT_WEIGHTS: Dict[str, float] = {
    "loitering_alert": 35.0,
    "running_alert": 20.0,
    "crowd_surge": 15.0,
    "intrusion": 50.0,
    "idle_alert": 5.0,
}

# Behavior score contributions per frame
_BEHAVIOR_SCORES: Dict[str, float] = {
    "idle": 0.0,
    "walking": 0.5,
    "running": 4.0,
    "loitering": 3.0,
    "unknown": 0.0,
}

# Score decay per second when no concerning events are observed
_SCORE_DECAY_RATE = 8.0  # points per second


class RiskProfile:
    """Maintains and decays a running risk score for one tracked person."""

    def __init__(self, track_id: int) -> None:
        self.track_id = track_id
        self.score: float = 0.0
        self.risk_level: str = RISK_LOW
        self.last_updated: float = time.time()
        self.event_log: List[Tuple[str, float]] = []  # (event_type, timestamp)

    def apply_event(self, event_type: str, weight: float) -> None:
        self.score = min(100.0, self.score + weight)
        self.event_log.append((event_type, time.time()))
        # Keep only last 20 events
        if len(self.event_log) > 20:
            self.event_log.pop(0)
        self._update_risk_level()

    def apply_behavior(self, behavior: str) -> None:
        delta = _BEHAVIOR_SCORES.get(behavior, 0.0)
        self.score = min(100.0, self.score + delta)
        self._update_risk_level()

    def decay(self, now: float) -> None:
        elapsed = now - self.last_updated
        self.score = max(0.0, self.score - _SCORE_DECAY_RATE * elapsed)
        self.last_updated = now
        self._update_risk_level()

    def _update_risk_level(self) -> None:
        if self.score >= _CRITICAL_SCORE:
            self.risk_level = RISK_CRITICAL
        elif self.score >= _HIGH_SCORE:
            self.risk_level = RISK_HIGH
        elif self.score >= _MEDIUM_SCORE:
            self.risk_level = RISK_MEDIUM
        else:
            self.risk_level = RISK_LOW

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "risk_score": round(self.score, 2),
            "risk_level": self.risk_level,
            "last_updated": self.last_updated,
        }


class PredictiveEngine:
    """
    Aggregates behavior signals and security events into per-person risk
    profiles, emitting actionable threat assessments every frame.

    Usage
    -----
    >>> engine = PredictiveEngine()
    >>> risk_report = engine.update(detections, events)
    """

    def __init__(self, profile_ttl: float = 60.0) -> None:
        """
        Parameters
        ----------
        profile_ttl : float
            Seconds of inactivity after which a risk profile is removed.
        """
        self.profile_ttl = profile_ttl
        self._profiles: Dict[int, RiskProfile] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: Optional[List[Dict[str, Any]]],
        events: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Update risk profiles from the latest detections and security events.

        Parameters
        ----------
        detections : list of detection dicts (with 'id' and 'behavior' keys)
        events : list of SecurityEvent objects from EventFusion (optional)

        Returns
        -------
        List of risk-profile dicts, one per active track, sorted by score desc.
        """
        now = time.time()
        active_ids: set = set()

        # Apply behavior contributions
        for det in detections or []:
            if not isinstance(det, dict):
                continue
            track_id = det.get("id")
            if not isinstance(track_id, int):
                continue
            active_ids.add(track_id)
            profile = self._get_or_create_profile(track_id, now)
            profile.decay(now)
            profile.apply_behavior(det.get("behavior", "unknown"))
            det["risk_score"] = round(profile.score, 2)
            det["risk_level"] = profile.risk_level

        # Apply event contributions
        for event in events or []:
            event_type = getattr(event, "event_type", None) or event.get("event_type")
            track_id = getattr(event, "track_id", None)
            weight = _EVENT_WEIGHTS.get(event_type, 10.0)

            if isinstance(track_id, int):
                profile = self._get_or_create_profile(track_id, now)
                profile.apply_event(event_type, weight)
                logger.info(
                    f"[RISK] Track {track_id} +{weight:.0f} pts "
                    f"from '{event_type}' → score={profile.score:.1f} ({profile.risk_level})"
                )
            else:
                # Scene-level event (e.g. crowd_surge): apply to all active tracks
                for tid in active_ids:
                    profile = self._get_or_create_profile(tid, now)
                    profile.apply_event(event_type, weight * 0.5)

        self._purge_stale_profiles(active_ids, now)

        report = sorted(
            [p.to_dict() for p in self._profiles.values()],
            key=lambda x: x["risk_score"],
            reverse=True,
        )
        return report

    def get_high_risk_tracks(self) -> List[Dict[str, Any]]:
        """Return profiles currently at HIGH or CRITICAL risk."""
        return [
            p.to_dict()
            for p in self._profiles.values()
            if p.risk_level in (RISK_HIGH, RISK_CRITICAL)
        ]

    def reset(self) -> None:
        self._profiles.clear()
        logger.info("PredictiveEngine state reset.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create_profile(self, track_id: int, now: float) -> RiskProfile:
        if track_id not in self._profiles:
            profile = RiskProfile(track_id)
            profile.last_updated = now
            self._profiles[track_id] = profile
        return self._profiles[track_id]

    def _purge_stale_profiles(self, active_ids: set, now: float) -> None:
        stale = [
            tid
            for tid, p in self._profiles.items()
            if tid not in active_ids and (now - p.last_updated) > self.profile_ttl
        ]
        for tid in stale:
            del self._profiles[tid]
            logger.debug(f"Purged risk profile for track ID: {tid}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis.event_fusion import SecurityEvent

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    engine = PredictiveEngine(profile_ttl=30.0)

    print("=== PredictiveEngine Smoke Test ===\n")

    # Simulate 5 frames of escalating behavior
    scenarios = [
        ("walking", []),
        ("walking", []),
        ("running", []),
        ("loitering", [SecurityEvent("loitering_alert", 1, time.time(), 0.9, {})]),
        ("loitering", [SecurityEvent("loitering_alert", 1, time.time(), 0.9, {})]),
    ]

    for i, (behavior, events) in enumerate(scenarios):
        detections = [
            {
                "id": 1,
                "class": "person",
                "behavior": behavior,
                "confidence": 0.9,
                "speed_px_per_s": 80.0 if behavior == "running" else 20.0,
                "loitering_time": i * 2.0,
            }
        ]
        report = engine.update(detections, events)
        r = report[0]
        print(
            f"Frame {i+1} | behavior={behavior:<10} "
            f"| score={r['risk_score']:.1f} | level={r['risk_level']}"
        )

    print("\nHigh-risk tracks:", engine.get_high_risk_tracks())
    print("\nDone.")
