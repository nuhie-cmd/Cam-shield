"""
Camera Integrity Monitor for CamShield
Detects SYSTEM-LEVEL tampering — network, stream, login, and config changes.

Output JSON:
{
    "stream_status":          str,   # CONNECTED | DISCONNECTED | DEGRADED | RECONNECTING
    "login_detected":         bool,  # unauthorised access to camera
    "configuration_changed":  bool,  # resolution/FPS/IP changed
    "stream_disconnected":    bool,  # feed loss flag
    "network_status":         str    # NORMAL | DEGRADED | OFFLINE | HIGH_LATENCY
}

Pipeline position:
    Camera System/Network → CameraIntegrityMonitor → Event Fusion
"""

import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------
STREAM_CONNECTED     = "CONNECTED"
STREAM_DISCONNECTED  = "DISCONNECTED"
STREAM_DEGRADED      = "DEGRADED"
STREAM_RECONNECTING  = "RECONNECTING"

NET_NORMAL           = "NORMAL"
NET_DEGRADED         = "DEGRADED"
NET_OFFLINE          = "OFFLINE"
NET_HIGH_LATENCY     = "HIGH_LATENCY"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
_HIGH_LATENCY_MS     = 150.0    # ping > this → HIGH_LATENCY
_DEGRADED_LOSS_RATE  = 0.10     # packet loss > 10% → DEGRADED
_RECONNECT_WINDOW    = 5.0      # seconds between disconnect and reconnect


@dataclass
class CameraConfig:
    """Snapshot of camera configuration parameters."""
    resolution:  Tuple[int, int] = (1920, 1080)
    fps:         float           = 30.0
    ip_address:  str             = ""
    firmware:    str             = "unknown"
    extra:       Dict[str, Any]  = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolution":  self.resolution,
            "fps":         self.fps,
            "ip_address":  self.ip_address,
            "firmware":    self.firmware,
            **self.extra,
        }


@dataclass
class LoginEvent:
    """A record of a login attempt detected on the camera device."""
    source_ip:   str
    timestamp:   float
    success:     bool
    username:    str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_ip":  self.source_ip,
            "timestamp":  self.timestamp,
            "success":    self.success,
            "username":   self.username,
        }


class CameraIntegrityMonitor:
    """
    System-level camera integrity checker.

    Tracks:
      - Stream connectivity (CONNECTED / DISCONNECTED / DEGRADED / RECONNECTING)
      - Unauthorised login events injected via ``report_login()``
      - Camera configuration drift vs a stored baseline
      - Network reachability and latency via optional ping checks

    Usage
    -----
    >>> monitor = CameraIntegrityMonitor(camera_ip="192.168.1.64")
    >>> monitor.set_baseline_config(CameraConfig(resolution=(1920,1080), fps=30))
    >>>
    >>> # Each frame / poll cycle:
    >>> monitor.update_stream(connected=True, frame_dropped=False)
    >>> report = monitor.check()
    >>> print(report)
    """

    def __init__(
        self,
        camera_ip:             str   = "",
        ping_host:             str   = "8.8.8.8",
        ping_port:             int   = 53,
        ping_timeout:          float = 1.0,
        high_latency_ms:       float = _HIGH_LATENCY_MS,
        packet_loss_threshold: float = _DEGRADED_LOSS_RATE,
        login_whitelist:       Optional[List[str]] = None,
    ) -> None:
        self.camera_ip             = camera_ip
        self.ping_host             = ping_host
        self.ping_port             = ping_port
        self.ping_timeout          = ping_timeout
        self.high_latency_ms       = high_latency_ms
        self.packet_loss_threshold = packet_loss_threshold
        self.login_whitelist: List[str] = login_whitelist or []

        # Stream tracking
        self._connected:         bool  = True
        self._degraded:          bool  = False
        self._reconnecting:      bool  = False
        self._disconnect_time:   Optional[float] = None
        self._consecutive_drops: int   = 0
        self._drop_threshold:    int   = 5      # drops before DEGRADED

        # Config baseline
        self._baseline_config:   Optional[CameraConfig] = None
        self._current_config:    Optional[CameraConfig] = None

        # Login events (injected externally from ONVIF / syslog parser)
        self._login_events:      List[LoginEvent] = []
        self._login_detected:    bool = False

        # Network
        self._last_latency_ms:   float = 0.0
        self._last_net_status:   str   = NET_NORMAL

    # ------------------------------------------------------------------
    # Configuration baseline
    # ------------------------------------------------------------------

    def set_baseline_config(self, config: CameraConfig) -> None:
        """Store the known-good camera configuration to detect future drift."""
        self._baseline_config = config
        self._current_config  = config
        logger.info(f"[INTEGRITY] Baseline config set: {config.to_dict()}")

    def update_config(self, config: CameraConfig) -> None:
        """
        Call this whenever the camera reports its current configuration.
        If it differs from baseline, ``configuration_changed`` will be True.
        """
        self._current_config = config

    # ------------------------------------------------------------------
    # Stream updates
    # ------------------------------------------------------------------

    def update_stream(self, connected: bool, frame_dropped: bool = False) -> None:
        """
        Notify the monitor of the current stream state.

        Parameters
        ----------
        connected     : True if the RTSP/stream read succeeded this cycle.
        frame_dropped : True if the frame was received but was corrupt/incomplete.
        """
        if not connected:
            if self._connected:
                self._disconnect_time = time.time()
                logger.warning("[INTEGRITY] Stream disconnected.")
            self._connected    = False
            self._reconnecting = False
            self._degraded     = False
            self._consecutive_drops = 0
        else:
            if not self._connected:
                elapsed = time.time() - (self._disconnect_time or 0.0)
                if elapsed <= _RECONNECT_WINDOW:
                    self._reconnecting = True
                else:
                    self._reconnecting = False
                self._disconnect_time = None
                logger.info("[INTEGRITY] Stream reconnected.")
            self._connected = True

            if frame_dropped:
                self._consecutive_drops += 1
                if self._consecutive_drops >= self._drop_threshold:
                    self._degraded = True
                    logger.warning(f"[INTEGRITY] Stream DEGRADED ({self._consecutive_drops} drops).")
            else:
                self._consecutive_drops = 0
                self._degraded = False

    # ------------------------------------------------------------------
    # Login injection (ONVIF / syslog parser feeds events here)
    # ------------------------------------------------------------------

    def report_login(
        self,
        source_ip: str,
        success:   bool   = True,
        username:  str    = "",
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Inject a login event detected from camera admin logs or ONVIF events.
        If the source IP is not in the whitelist, ``login_detected`` is set True.
        """
        ts = timestamp or time.time()
        event = LoginEvent(source_ip=source_ip, timestamp=ts,
                           success=success, username=username)
        self._login_events.append(event)

        if source_ip not in self.login_whitelist:
            self._login_detected = True
            logger.warning(
                f"[INTEGRITY] Unauthorised login from {source_ip} "
                f"(user='{username}', success={success})"
            )

    def clear_login_flag(self) -> None:
        """Acknowledge and clear the login_detected flag after handling."""
        self._login_detected = False

    # ------------------------------------------------------------------
    # Network check (TCP ping)
    # ------------------------------------------------------------------

    def check_network(self) -> str:
        """
        Probe network reachability via a TCP connection attempt.
        Returns one of: NORMAL | HIGH_LATENCY | OFFLINE
        """
        try:
            start = time.time()
            with socket.create_connection(
                (self.ping_host, self.ping_port), timeout=self.ping_timeout
            ):
                pass
            latency_ms = (time.time() - start) * 1000.0
            self._last_latency_ms = latency_ms

            if latency_ms > self.high_latency_ms:
                status = NET_HIGH_LATENCY
            else:
                status = NET_NORMAL

        except OSError:
            self._last_latency_ms = -1.0
            status = NET_OFFLINE

        self._last_net_status = status
        return status

    # ------------------------------------------------------------------
    # Main report
    # ------------------------------------------------------------------

    def check(self, run_network_check: bool = False) -> Dict[str, Any]:
        """
        Generate the camera_integrity report dict.

        Parameters
        ----------
        run_network_check : bool
            If True, performs a live TCP ping before generating the report.
            Set False in production loops to avoid blocking; run it in a
            background thread instead and let the cached value be used.

        Returns
        -------
        dict matching the CamShield camera_integrity schema
        """
        if run_network_check:
            self.check_network()

        # Stream status
        if not self._connected:
            stream_status = STREAM_DISCONNECTED
        elif self._reconnecting:
            stream_status = STREAM_RECONNECTING
        elif self._degraded:
            stream_status = STREAM_DEGRADED
        else:
            stream_status = STREAM_CONNECTED

        # Configuration change
        config_changed = self._detect_config_change()

        report: Dict[str, Any] = {
            "stream_status":         stream_status,
            "login_detected":        self._login_detected,
            "configuration_changed": config_changed,
            "stream_disconnected":   not self._connected,
            "network_status":        self._last_net_status,
            "timestamp":             time.time(),
        }

        # Diagnostics (optional extra detail)
        report["_detail"] = {
            "latency_ms":         round(self._last_latency_ms, 1),
            "consecutive_drops":  self._consecutive_drops,
            "login_events_count": len(self._login_events),
        }

        if report["login_detected"] or config_changed or stream_status != STREAM_CONNECTED:
            logger.warning(f"[INTEGRITY] {report}")

        return report

    def reset(self) -> None:
        """Clear all state."""
        self._connected         = True
        self._degraded          = False
        self._reconnecting      = False
        self._disconnect_time   = None
        self._consecutive_drops = 0
        self._login_events.clear()
        self._login_detected    = False
        self._last_net_status   = NET_NORMAL
        logger.info("[INTEGRITY] State reset.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_config_change(self) -> bool:
        """Return True if current config differs from baseline."""
        b = self._baseline_config
        c = self._current_config
        if b is None or c is None:
            return False
        return (
            b.resolution != c.resolution
            or abs(b.fps - c.fps) > 0.5
            or (b.ip_address and b.ip_address != c.ip_address)
            or (b.firmware != "unknown" and b.firmware != c.firmware)
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    monitor = CameraIntegrityMonitor(
        camera_ip="192.168.1.64",
        login_whitelist=["192.168.1.1", "192.168.1.10"],
    )
    monitor.set_baseline_config(CameraConfig(
        resolution=(1920, 1080), fps=30.0,
        ip_address="192.168.1.64", firmware="v2.1.0"
    ))

    print("=== CameraIntegrityMonitor Smoke Test ===\n")

    # Test 1: healthy stream
    monitor.update_stream(connected=True)
    r = monitor.check()
    print(f"[Healthy]          stream={r['stream_status']:<15} "
          f"login={r['login_detected']}  cfg_change={r['configuration_changed']}  "
          f"net={r['network_status']}")

    # Test 2: unauthorised login
    monitor.report_login(source_ip="10.0.0.99", success=True, username="admin")
    r = monitor.check()
    print(f"[Unauth login]     stream={r['stream_status']:<15} "
          f"login={r['login_detected']}  cfg_change={r['configuration_changed']}")
    monitor.clear_login_flag()

    # Test 3: whitelisted login (should not flag)
    monitor.report_login(source_ip="192.168.1.1", success=True, username="admin")
    r = monitor.check()
    print(f"[Whitelisted login]stream={r['stream_status']:<15} "
          f"login={r['login_detected']}")

    # Test 4: config change (resolution dropped)
    monitor.update_config(CameraConfig(
        resolution=(640, 480), fps=15.0,
        ip_address="192.168.1.64", firmware="v2.1.0"
    ))
    r = monitor.check()
    print(f"[Config change]    stream={r['stream_status']:<15} "
          f"cfg_change={r['configuration_changed']}")

    # Test 5: stream disconnected → reconnected
    monitor.update_stream(connected=False)
    r = monitor.check()
    print(f"[Disconnected]     stream={r['stream_status']:<15} "
          f"disconnected={r['stream_disconnected']}")

    monitor.update_stream(connected=True)
    r = monitor.check()
    print(f"[Reconnecting]     stream={r['stream_status']:<15} "
          f"disconnected={r['stream_disconnected']}")

    # Test 6: degraded stream (repeated drops)
    monitor.reset()
    monitor.set_baseline_config(CameraConfig(resolution=(1920,1080), fps=30.0))
    for _ in range(6):
        monitor.update_stream(connected=True, frame_dropped=True)
    r = monitor.check()
    print(f"[Degraded stream]  stream={r['stream_status']:<15} "
          f"drops={r['_detail']['consecutive_drops']}")

    # Test 7: live network check
    print("\n--- Live Network Check ---")
    net = monitor.check_network()
    r = monitor.check()
    print(f"  network_status={r['network_status']}  latency={r['_detail']['latency_ms']}ms")

    print("\nDone.")
