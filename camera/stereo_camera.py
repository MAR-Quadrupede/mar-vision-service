#!/usr/bin/env python3
"""
M.A.R. — 3D Global Shutter Stereo Camera Driver
Supports native side-by-side acquisition, cross-platform backend selection,
hardware parameter configuration, and mock stream generation for development.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import platform
import threading
import time
import cv2
import numpy as np

try:
    from ..config import CameraConfig
except (ImportError, ValueError):
    from config import CameraConfig


@dataclass
class StereoFrame:
    """Represents a synchronized stereo capture."""
    combined: np.ndarray
    left: np.ndarray
    right: np.ndarray
    timestamp: float
    is_mock: bool = False


class StereoCameraDriver:
    """
    High-reliability driver for USB 3.0 / USB 2.0 Stereo Cameras.
    Designed for 3D Global Shutter modules in Side-by-Side (SBS) format.
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.active_index: Optional[int] = None
        self.is_opened = False
        self.is_mock = False
        self._lock = threading.Lock()
        self._mock_step = 0

    def _get_backends(self) -> List[int]:
        """Returns ordered list of OpenCV capture backends for the current OS."""
        system = platform.system()
        if system == "Darwin":
            return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        elif system == "Linux":
            return [cv2.CAP_V4L2, cv2.CAP_ANY]
        else:
            return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    def open(self) -> bool:
        """
        Attempts to open the camera using primary index and fallback candidates.
        Falls back to synthetic mock stream if no physical camera is accessible.
        """
        backends = self._get_backends()
        indices = [self.config.primary_index]
        indices.extend([idx for idx in self.config.fallback_indices if idx != self.config.primary_index])

        print("\n[CAMERA] Inizializzazione 3D Global Shutter Stereo Camera...")

        for index in indices:
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(index, backend)
                    if cap.isOpened():
                        # Read a test frame to ensure stream is valid
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            self.cap = cap
                            self.active_index = index
                            self.is_opened = True
                            self._apply_hardware_settings()
                            return True
                    cap.release()
                except Exception:
                    pass

        print("  ⚠️ Nessuna telecamera fisica disponibile sui dispositivi testati.")
        print("  🔧 Attivazione modalità MOCK SINTETICA per test ed emulazione continua...")
        self.is_mock = True
        self.is_opened = True
        return True

    def _apply_hardware_settings(self):
        """Applies resolution, frame rate and disables auto-tuning for Global Shutter."""
        if not self.cap or not self.is_opened:
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        # Disable autofocus and autoexposure to avoid jitter during robot motion
        if not self.config.auto_focus:
            try:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            except Exception:
                pass

        if not self.config.auto_exposure:
            try:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            except Exception:
                pass

        act_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        act_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        act_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"  ✓ Stereocamera agganciata con successo!")
        print(f"  • Dispositivo: /dev/video{self.active_index} (Index {self.active_index})")
        print(f"  • Risoluzione totale: {act_w}x{act_h} (Risoluzione occhio: {act_w//2}x{act_h})")
        print(f"  • Frame rate: {act_fps:.1f} FPS (Global Shutter mode)")

    def _generate_mock_frame(self) -> StereoFrame:
        """Generates synthetic stereo frame with animated simulated obstacles."""
        self._mock_step += 1
        width = self.config.frame_width
        height = self.config.frame_height
        half_w = width // 2

        now = time.time()
        # Create gradient background
        x = np.linspace(0, 1, half_w, dtype=np.float32)
        bg_left = np.tile((x * 40 + 20).astype(np.uint8), (height, 1))
        bg_left = cv2.cvtColor(bg_left, cv2.COLOR_GRAY2BGR)
        bg_right = bg_left.copy()

        # Simulated moving obstacle (e.g., ball/box) with parallax disparity
        cx = int((np.sin(self._mock_step * 0.05) * 0.35 + 0.5) * half_w)
        cy = int((np.cos(self._mock_step * 0.03) * 0.2 + 0.6) * height)
        radius = int(50 + 15 * np.sin(self._mock_step * 0.04))
        disparity = 24  # Parallax shift between left and right

        # Draw left object
        cv2.circle(bg_left, (cx, cy), radius, (0, 140, 255), -1)
        cv2.putText(bg_left, "SIMULATORE M.A.R. - SX", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw right object with horizontal disparity shift
        cv2.circle(bg_right, (cx - disparity, cy), radius, (0, 140, 255), -1)
        cv2.putText(bg_right, "SIMULATORE M.A.R. - DX", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        combined = np.hstack([bg_left, bg_right])
        time.sleep(0.02)  # Simulate frame timing (~40-50 FPS)

        return StereoFrame(
            combined=combined,
            left=bg_left,
            right=bg_right,
            timestamp=now,
            is_mock=True
        )

    def read(self) -> Optional[StereoFrame]:
        """
        Captures a frame and splits it into left and right views.
        Returns StereoFrame object or None on failure.
        """
        with self._lock:
            if not self.is_opened:
                return None

            if self.is_mock:
                return self._generate_mock_frame()

            ret, combined = self.cap.read()
            if not ret or combined is None or combined.size == 0:
                return None

            h, w = combined.shape[:2]
            mid = w // 2

            left = combined[:, :mid].copy()
            right = combined[:, mid:].copy()

            return StereoFrame(
                combined=combined,
                left=left,
                right=right,
                timestamp=time.time(),
                is_mock=False
            )

    def release(self):
        """Releases the camera hardware."""
        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.is_opened = False
            print("[CAMERA] Dispositivo fotocamera rilasciato.")
