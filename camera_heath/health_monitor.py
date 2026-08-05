"""
Camera Health Monitor for CamShield
Detects VISUAL tampering directly from the video stream.

Output JSON:
{
    "health_score": float,      # 0.0 (dead) → 1.0 (perfect)
    "blur": bool,               # Laplacian variance drop
    "brightness": float,        # 0.0–1.0 mean pixel brightness
    "freeze": bool,             # SSIM too high for N frames
    "obstruction": bool,        # near-black or near-uniform frame
    "angle_tampered": bool,     # scene structure changed vs baseline
    "color_change": bool        # RGB channel distribution shifted
}

Pipeline position:
    Camera Feed → CameraHealthMonitor → Event Fusion
"""

import logging
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults / tuneable constants
# ---------------------------------------------------------------------------
_BLUR_THRESHOLD          = 80.0     # Laplacian variance below this → blurry
_BRIGHTNESS_LOW          = 0.08     # mean/255 below this → too dark
_BRIGHTNESS_HIGH         = 0.92     # mean/255 above this → overexposed
_FREEZE_SSIM_THRESHOLD   = 0.997    # SSIM above this → frozen
_FREEZE_CONFIRM_FRAMES   = 8        # consecutive frozen frames to confirm
_OBSTRUCTION_STD_THRESH  = 8.0      # pixel std-dev below this → blocked
_ANGLE_SSIM_THRESHOLD    = 0.55     # SSIM vs baseline below this → angle changed
_COLOR_SHIFT_THRESHOLD   = 25.0     # per-channel mean shift vs baseline (0-255)

# Score penalty per active issue (sum subtracted from 1.0)
_ISSUE_PENALTIES = {
    "blur":          0.20,
    "freeze":        0.35,
    "obstruction":   0.40,
    "angle_tampered":0.30,
    "color_change":  0.15,
    "brightness":    0.10,
}


def _ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """Fast single-channel SSIM (no scikit-image dependency)."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    C1, C2 = 6.5025, 58.5225       # (0.01*255)^2, (0.03*255)^2
    mu1, mu2 = cv2.GaussianBlur(a, (11,11), 1.5), cv2.GaussianBlur(b, (11,11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
    s1 = cv2.GaussianBlur(a*a, (11,11), 1.5) - mu1_sq
    s2 = cv2.GaussianBlur(b*b, (11,11), 1.5) - mu2_sq
    s12 = cv2.GaussianBlur(a*b, (11,11), 1.5) - mu1_mu2
    num = (2*mu1_mu2 + C1) * (2*s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    ssim_map = num / (den + 1e-8)
    return float(np.mean(ssim_map))


class CameraHealthMonitor:
    """
    Visual-layer camera health checker.

    Usage
    -----
    >>> monitor = CameraHealthMonitor()
    >>> monitor.set_baseline(first_good_frame)   # optional but recommended
    >>> report = monitor.check(frame)
    >>> print(report)   # dict matching the spec
    """

    def __init__(
        self,
        blur_threshold:           float = _BLUR_THRESHOLD,
        brightness_low:           float = _BRIGHTNESS_LOW,
        brightness_high:          float = _BRIGHTNESS_HIGH,
        freeze_ssim_threshold:    float = _FREEZE_SSIM_THRESHOLD,
        freeze_confirm_frames:    int   = _FREEZE_CONFIRM_FRAMES,
        obstruction_std_threshold:float = _OBSTRUCTION_STD_THRESH,
        angle_ssim_threshold:     float = _ANGLE_SSIM_THRESHOLD,
        color_shift_threshold:    float = _COLOR_SHIFT_THRESHOLD,
    ) -> None:
        self.blur_threshold            = blur_threshold
        self.brightness_low            = brightness_low
        self.brightness_high           = brightness_high
        self.freeze_ssim_threshold     = freeze_ssim_threshold
        self.freeze_confirm_frames     = freeze_confirm_frames
        self.obstruction_std_threshold = obstruction_std_threshold
        self.angle_ssim_threshold      = angle_ssim_threshold
        self.color_shift_threshold     = color_shift_threshold

        # Internal state
        self._prev_gray:        Optional[np.ndarray] = None
        self._baseline_gray:    Optional[np.ndarray] = None
        self._baseline_bgr_means: Optional[np.ndarray] = None   # [B, G, R]
        self._freeze_streak:    int = 0

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    def set_baseline(self, frame: np.ndarray) -> None:
        """
        Store a known-good frame as the reference for angle and colour checks.
        Call this once at startup after the camera stabilises.
        """
        gray = self._to_gray(frame)
        self._baseline_gray = cv2.resize(gray, (320, 240))
        self._baseline_bgr_means = np.array(
            [np.mean(frame[:, :, c]) for c in range(3)], dtype=np.float32
        )
        logger.info("[HEALTH] Baseline frame stored.")

    # ------------------------------------------------------------------
    # Main check
    # ------------------------------------------------------------------

    def check(self, frame: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate one frame and return the camera_health dict.

        Parameters
        ----------
        frame : BGR np.ndarray or None (None = disconnected / read failure)

        Returns
        -------
        dict  matching the CamShield camera_health schema
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return self._disconnected_report()

        gray = self._to_gray(frame)
        small_gray = cv2.resize(gray, (320, 240))

        # -- Individual checks -------------------------------------------
        blur        = self._check_blur(gray)
        brightness  = self._check_brightness(gray)
        freeze      = self._check_freeze(small_gray)
        obstruction = self._check_obstruction(gray)
        angle_tampered = self._check_angle(small_gray)
        color_change   = self._check_color(frame)

        # -- Brightness flag (outside normal window) ----------------------
        brightness_flag = (brightness < self.brightness_low or
                           brightness > self.brightness_high)

        # -- Health score -------------------------------------------------
        active_issues = {
            "blur":           blur,
            "freeze":         freeze,
            "obstruction":    obstruction,
            "angle_tampered": angle_tampered,
            "color_change":   color_change,
            "brightness":     brightness_flag,
        }
        penalty = sum(v for k, v in _ISSUE_PENALTIES.items() if active_issues.get(k, False))
        health_score = round(max(0.0, 1.0 - penalty), 3)

        report: Dict[str, Any] = {
            "health_score":   health_score,
            "blur":           blur,
            "brightness":     round(brightness, 4),
            "freeze":         freeze,
            "obstruction":    obstruction,
            "angle_tampered": angle_tampered,
            "color_change":   color_change,
            "timestamp":      time.time(),
        }

        if health_score < 1.0:
            issues = [k for k, v in active_issues.items() if v]
            logger.warning(f"[HEALTH] issues={issues}  score={health_score}")

        # Update prev frame for next call
        self._prev_gray = small_gray
        return report

    def reset(self) -> None:
        """Clear state on camera reconnect or scene change."""
        self._prev_gray = None
        self._freeze_streak = 0
        logger.info("[HEALTH] State reset.")

    # ------------------------------------------------------------------
    # Individual detectors
    # ------------------------------------------------------------------

    def _check_blur(self, gray: np.ndarray) -> bool:
        """Laplacian variance — low variance = blurry / lens smeared."""
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return bool(variance < self.blur_threshold)

    def _check_brightness(self, gray: np.ndarray) -> float:
        """Mean pixel brightness normalised to 0–1."""
        return float(np.mean(gray)) / 255.0

    def _check_freeze(self, small_gray: np.ndarray) -> bool:
        """SSIM between current and previous frame — near 1.0 = frozen."""
        if self._prev_gray is None or self._prev_gray.shape != small_gray.shape:
            return False
        ssim = _ssim_gray(small_gray, self._prev_gray)
        if ssim >= self.freeze_ssim_threshold:
            self._freeze_streak += 1
        else:
            self._freeze_streak = 0
        return self._freeze_streak >= self.freeze_confirm_frames

    def _check_obstruction(self, gray: np.ndarray) -> bool:
        """
        Near-uniform or near-black frame = lens blocked.
        Uses pixel std-dev: very low std means almost flat image.
        """
        std = float(np.std(gray))
        mean = float(np.mean(gray))
        # Either extremely low std (uniform colour) or very dark
        return std < self.obstruction_std_threshold or mean < 8.0

    def _check_angle(self, small_gray: np.ndarray) -> bool:
        """SSIM vs baseline — scene structure changed = camera moved."""
        if self._baseline_gray is None:
            return False
        if self._baseline_gray.shape != small_gray.shape:
            baseline = cv2.resize(self._baseline_gray, (small_gray.shape[1], small_gray.shape[0]))
        else:
            baseline = self._baseline_gray
        ssim = _ssim_gray(small_gray, baseline)
        return bool(ssim < self.angle_ssim_threshold)

    def _check_color(self, frame: np.ndarray) -> bool:
        """Per-channel mean shift vs baseline — abnormal colour cast."""
        if self._baseline_bgr_means is None:
            return False
        current_means = np.array(
            [np.mean(frame[:, :, c]) for c in range(3)], dtype=np.float32
        )
        shift = np.max(np.abs(current_means - self._baseline_bgr_means))
        return bool(shift > self.color_shift_threshold)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _disconnected_report() -> Dict[str, Any]:
        return {
            "health_score":   0.0,
            "blur":           False,
            "brightness":     0.0,
            "freeze":         False,
            "obstruction":    False,
            "angle_tampered": False,
            "color_change":   False,
            "timestamp":      time.time(),
            "_note":          "disconnected",
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    monitor = CameraHealthMonitor(
        blur_threshold=80.0,
        freeze_confirm_frames=3,
        obstruction_std_threshold=8.0,
        angle_ssim_threshold=0.55,
        color_shift_threshold=25.0,
    )

    # Healthy baseline
    baseline = np.random.randint(40, 180, (480, 640, 3), dtype=np.uint8)
    monitor.set_baseline(baseline)

    print("=== CameraHealthMonitor Smoke Test ===\n")

    scenarios = [
        ("Healthy frame",      np.random.randint(40, 180, (480, 640, 3), dtype=np.uint8)),
        ("Black frame",        np.zeros((480, 640, 3), dtype=np.uint8)),
        ("Overexposed",        np.full((480, 640, 3), 250, dtype=np.uint8)),
        ("Frozen (repeat)",    baseline.copy()),
        ("Frozen (repeat)",    baseline.copy()),
        ("Frozen (repeat)",    baseline.copy()),
        ("Frozen (repeat)",    baseline.copy()),
        ("Color shift",        np.full((480, 640, 3), [40, 30, 200], dtype=np.uint8)),
        ("Blurry (solid)",     np.full((480, 640, 3), 128, dtype=np.uint8)),
        ("Disconnected",       None),
    ]

    for label, frame in scenarios:
        r = monitor.check(frame)
        print(f"[{label:<22}] score={r['health_score']:.2f} | "
              f"blur={r['blur']} freeze={r['freeze']} obstruction={r['obstruction']} "
              f"angle={r['angle_tampered']} color={r['color_change']}")

    print("\nDone.")
