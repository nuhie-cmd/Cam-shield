"""
Tracker Module for CamShield
Provides persistent ID tracking for detected objects using a lightweight
centroid-based Euclidean distance matching algorithm.
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Configure module-level logger
logger = logging.getLogger(__name__)

# Type alias for clarity
Detection = Dict[str, Any]


class Track:
    """
    Represents an active object track maintained in memory.
    """

    def __init__(
        self,
        track_id: int,
        center: List[float],
        bbox: List[float],
        timestamp: float
    ) -> None:
        self.track_id: int = track_id
        self.last_center: List[float] = center
        self.last_bbox: List[float] = bbox
        self.last_timestamp: float = timestamp
        self.age: int = 1
        self.missed_frames: int = 0

    def update(self, center: List[float], bbox: List[float], timestamp: float) -> None:
        self.last_center = center
        self.last_bbox = bbox
        self.last_timestamp = timestamp
        self.age += 1
        self.missed_frames = 0

    def mark_missed(self) -> None:
        self.missed_frames += 1


class Tracker:
    """
    Centroid-based multi-object tracker that maintains stable integer IDs
    for detected persons across consecutive video frames.
    """

    def __init__(
        self,
        max_distance: float = 100.0,
        max_missed_frames: int = 15,
        max_timeout: float = 3.0
    ) -> None:
        self.max_distance: float = max_distance
        self.max_missed_frames: int = max_missed_frames
        self.max_timeout: float = max_timeout

        self._next_id: int = 1
        self.tracks: Dict[int, Track] = {}

    def _get_valid_center(self, det: Detection) -> Optional[List[float]]:
        center = det.get("center")
        if isinstance(center, (list, tuple)) and len(center) == 2:
            try:
                return [float(center[0]), float(center[1])]
            except (ValueError, TypeError):
                pass

        bbox = det.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = (float(v) for v in bbox)
                return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
            except (ValueError, TypeError):
                pass

        return None

    def _create_track(
        self, center: List[float], bbox: List[float], timestamp: float
    ) -> int:
        assigned_id = self._next_id
        self.tracks[assigned_id] = Track(
            track_id=assigned_id,
            center=center,
            bbox=bbox,
            timestamp=timestamp
        )
        self._next_id += 1
        return assigned_id

    def _match_detections(
        self,
        detection_centers: List[List[float]],
        track_ids: List[int]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not track_ids or not detection_centers:
            matched_pairs: List[Tuple[int, int]] = []
            unmatched_tracks = list(range(len(track_ids)))
            unmatched_detections = list(range(len(detection_centers)))
            return matched_pairs, unmatched_tracks, unmatched_detections

        num_tracks = len(track_ids)
        num_dets = len(detection_centers)

        dist_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
        for i, tid in enumerate(track_ids):
            t_center = self.tracks[tid].last_center
            for j, d_center in enumerate(detection_centers):
                dist = math.hypot(t_center[0] - d_center[0], t_center[1] - d_center[1])
                dist_matrix[i, j] = dist

        matched_pairs = []
        used_tracks = set()
        used_dets = set()

        if dist_matrix.size > 0:
            flat_indices = np.argsort(dist_matrix, axis=None)
            for idx in flat_indices:
                i, j = np.unravel_index(idx, dist_matrix.shape)
                if i in used_tracks or j in used_dets:
                    continue
                if dist_matrix[i, j] > self.max_distance:
                    break

                matched_pairs.append((i, j))
                used_tracks.add(i)
                used_dets.add(j)

        unmatched_tracks = [i for i in range(num_tracks) if i not in used_tracks]
        unmatched_detections = [j for j in range(num_dets) if j not in used_dets]

        return matched_pairs, unmatched_tracks, unmatched_detections

    def _remove_stale_tracks(self, current_time: float) -> None:
        stale_ids = []
        for tid, track in self.tracks.items():
            time_elapsed = current_time - track.last_timestamp
            if (
                track.missed_frames > self.max_missed_frames
                or time_elapsed > self.max_timeout
            ):
                stale_ids.append(tid)

        for tid in stale_ids:
            del self.tracks[tid]
            logger.debug(f"Purged stale track ID: {tid}")

    def update(self, detections: Optional[List[Detection]]) -> List[Detection]:
        current_time = time.time()

        if not detections or not isinstance(detections, list):
            for track in self.tracks.values():
                track.mark_missed()
            self._remove_stale_tracks(current_time)
            return [] if detections is None else detections

        valid_indices: List[int] = []
        detection_centers: List[List[float]] = []

        for idx, det in enumerate(detections):
            if not isinstance(det, dict):
                continue
            center = self._get_valid_center(det)
            if center is not None:
                valid_indices.append(idx)
                detection_centers.append(center)

        track_ids = list(self.tracks.keys())

        matched_pairs, unmatched_tracks, unmatched_dets = self._match_detections(
            detection_centers, track_ids
        )

        for track_idx, det_idx in matched_pairs:
            tid = track_ids[track_idx]
            original_idx = valid_indices[det_idx]
            det = detections[original_idx]

            center = detection_centers[det_idx]
            bbox = det.get("bbox", [0.0, 0.0, 0.0, 0.0])
            ts = float(det.get("timestamp", current_time))

            self.tracks[tid].update(center, bbox, ts)
            det["id"] = tid

        for track_idx in unmatched_tracks:
            tid = track_ids[track_idx]
            self.tracks[tid].mark_missed()

        for det_idx in unmatched_dets:
            original_idx = valid_indices[det_idx]
            det = detections[original_idx]

            center = detection_centers[det_idx]
            bbox = det.get("bbox", [0.0, 0.0, 0.0, 0.0])
            ts = float(det.get("timestamp", current_time))

            new_id = self._create_track(center, bbox, ts)
            det["id"] = new_id

        self._remove_stale_tracks(current_time)
        return detections

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1
        logger.info("Tracker state has been reset.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tracker = Tracker()
    sample = [
        {
            "bbox": [100, 100, 200, 200],
            "center": [150, 150],
            "area": 10000,
            "timestamp": time.time(),
            "confidence": 0.95,
            "class": "person",
            "id": None,
        }
    ]
    print("=== Tracker Test ===")
    tracked = tracker.update(sample)
    for person in tracked:
        print(person)
