#!/usr/bin/env python3
"""
M.A.R. — 3D Global Shutter Stereo Camera Driver
Supports threaded non-blocking acquisition, hardware parameter configuration,
upside-down mounting 180° rotation, and high-fidelity mock stream generation.
"""

from dataclasses import dataclass
from typing import Optional, List
import platform
import subprocess
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
    Uses dedicated capture thread with Condition variable to eliminate frame lag
    and ensure buffer size 1 with minimal latency.
    """

    def __init__(self, config: CameraConfig, upside_down: bool = True):
        self.config = config
        self.upside_down = upside_down
        self.cap: Optional[cv2.VideoCapture] = None
        self.active_index: Optional[int] = None
        self.is_opened = False
        self.is_mock = False

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._latest_frame: Optional[StereoFrame] = None
        self._mock_step = 0
        self.actual_fps = 0.0
        self._reopen_lock = threading.Lock()
        self._last_frame_mono = 0.0
        self.stalled = False
        self._watchdog: Optional[threading.Thread] = None

        # Optional override for higher FPS (USB bandwidth). Example: 1280x400 @ 30.
        import os
        if os.getenv("MAR_CAM_WIDTH"):
            try:
                self.config.frame_width = int(os.getenv("MAR_CAM_WIDTH", self.config.frame_width))
                self.config.frame_height = int(os.getenv("MAR_CAM_HEIGHT", self.config.frame_height))
                self.config.fps = int(os.getenv("MAR_CAM_FPS", "30"))
            except Exception:
                pass

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
        print("\n[CAMERA] Inizializzazione 3D Global Shutter Stereo Camera...")

        if self._open_device():
            self._start_capture_thread()
            return True

        print("  ⚠️ Nessuna telecamera fisica disponibile sui dispositivi testati.")
        print("  🔧 Attivazione modalità MOCK SINTETICA per test ed emulazione continua...")
        self.is_mock = True
        self.is_opened = True
        self._start_capture_thread()
        return True

    def _configure_capture(self, cap: "cv2.VideoCapture") -> None:
        """MJPG, buffer 1, timeout di lettura: senza timeout V4L2 blocca read() per sempre se l'USB cade."""
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        for prop_name in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
            prop = getattr(cv2, prop_name, None)
            if prop is not None:
                try:
                    cap.set(prop, 2000)
                except Exception:
                    pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)

    def _open_device(self) -> bool:
        """Apre una telecamera fisica. Non avvia thread e non cade nel mock."""
        backends = self._get_backends()
        indices = [self.config.primary_index]
        indices.extend([idx for idx in self.config.fallback_indices if idx != self.config.primary_index])
        for attempt in range(1, 4):
            for index in indices:
                for backend in backends:
                    cap = None
                    try:
                        cap = cv2.VideoCapture(index, backend)
                        if not cap.isOpened():
                            cap.release()
                            continue
                        self._configure_capture(cap)
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            self.cap = cap
                            self.active_index = index
                            self.is_opened = True
                            self.is_mock = False
                            self.stalled = False
                            self._last_frame_mono = time.perf_counter()
                            self._apply_hardware_settings()
                            return True
                        cap.release()
                    except Exception:
                        if cap is not None:
                            try:
                                cap.release()
                            except Exception:
                                pass
            if attempt < 3:
                time.sleep(0.4)
        return False

    def _drop_capture(self) -> None:
        cap = self.cap
        self.cap = None
        self.is_opened = False
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _reopen_device(self) -> bool:
        """Rilascia il device (sblocca un read() appeso) e riprova le /dev/video. Mai il mock."""
        with self._reopen_lock:
            self.stalled = True
            self._drop_capture()
            ok = self._open_device()
            if ok:
                print(f"[CAMERA] riagganciata su /dev/video{self.active_index}")
            else:
                print("[CAMERA] device assente, riprovo")
            return ok

    def _apply_hardware_settings(self):
        """Applies resolution, frame rate and disables auto-tuning for Global Shutter."""
        if not self.cap or not self.is_opened:
            return

        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        if platform.system() == "Linux" and self.active_index is not None:
            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", f"/dev/video{self.active_index}", "-c",
                     "auto_exposure=1,exposure_time_absolute=800,gain=120,gamma=150,white_balance_automatic=1"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2
                )
            except Exception:
                pass

        try:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        except Exception:
            pass

        act_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        act_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        act_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"  ✓ Stereocamera agganciata con successo!")
        print(f"  • Dispositivo: /dev/video{self.active_index} (Index {self.active_index})")
        print(f"  • Risoluzione totale: {act_w}x{act_h} (Risoluzione occhio: {act_w//2}x{act_h})")
        print(f"  • Frame rate: {act_fps:.1f} FPS (Global Shutter mode)")
        print(f"  • Montaggio invertito 180°: {'ATTIVO' if self.upside_down else 'DISATTIVO'}")

    def _start_capture_thread(self):
        """Starts asynchronous acquisition thread and the USB stall watchdog."""
        self._running = True
        self._last_frame_mono = time.perf_counter()
        self._thread = threading.Thread(target=self._capture_worker, name="CameraCaptureThread", daemon=True)
        self._thread.start()
        self._watchdog = threading.Thread(target=self._watchdog_worker, name="CameraWatchdog", daemon=True)
        self._watchdog.start()

    def _generate_mock_frame(self) -> StereoFrame:
        """Generates synthetic stereo frame with animated simulated obstacles."""
        self._mock_step += 1
        width = self.config.frame_width
        height = self.config.frame_height
        half_w = width // 2

        now = time.time()
        x = np.linspace(0, 1, half_w, dtype=np.float32)
        bg_left = np.tile((x * 40 + 20).astype(np.uint8), (height, 1))
        bg_left = cv2.cvtColor(bg_left, cv2.COLOR_GRAY2BGR)
        bg_right = bg_left.copy()

        cx = int((np.sin(self._mock_step * 0.05) * 0.35 + 0.5) * half_w)
        cy = int((np.cos(self._mock_step * 0.03) * 0.2 + 0.6) * height)
        radius = int(50 + 15 * np.sin(self._mock_step * 0.04))
        disparity = 24

        cv2.circle(bg_left, (cx, cy), radius, (0, 140, 255), -1)
        cv2.putText(bg_left, "SIMULATORE M.A.R. - SX", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.circle(bg_right, (cx - disparity, cy), radius, (0, 140, 255), -1)
        cv2.putText(bg_right, "SIMULATORE M.A.R. - DX", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        combined = np.hstack([bg_left, bg_right])
        time.sleep(0.03)

        return StereoFrame(
            combined=combined,
            left=bg_left,
            right=bg_right,
            timestamp=now,
            is_mock=True
        )

    def _capture_worker(self):
        """Background loop acquiring fresh frames with minimal latency."""
        n_frames = 0
        t_fps = time.perf_counter()
        fail_streak = 0

        while self._running:
            if self.is_mock:
                frame = self._generate_mock_frame()
                with self._lock:
                    self._latest_frame = frame
                    self._cond.notify_all()
                continue

            cap = self.cap
            if cap is None or not cap.isOpened():
                self._reopen_device()
                time.sleep(0.5)
                continue

            try:
                ret, combined = cap.read()
            except Exception:
                ret, combined = False, None
            if not ret or combined is None or getattr(combined, "size", 0) == 0:
                fail_streak += 1
                self.stalled = True
                # read() è tornato (timeout o device staccato): riapri, non restare sul vecchio frame
                if fail_streak >= 8:
                    print(f"[CAMERA] {fail_streak} letture vuote, riapro il dispositivo")
                    self._reopen_device()
                    fail_streak = 0
                time.sleep(0.05)
                continue
            fail_streak = 0
            self.stalled = False

            if self.upside_down:
                # 180° rotation flips horizontal and vertical
                combined = cv2.flip(combined, -1)

            h, w = combined.shape[:2]
            hw = w // 2

            left = combined[:, :hw].copy()
            right = combined[:, hw:].copy()

            frame = StereoFrame(
                combined=combined,
                left=left,
                right=right,
                timestamp=time.time(),
                is_mock=False
            )

            n_frames += 1
            now = time.perf_counter()
            self._last_frame_mono = now
            if now - t_fps >= 1.0:
                self.actual_fps = n_frames / (now - t_fps)
                n_frames = 0
                t_fps = now

            with self._lock:
                self._latest_frame = frame
                self._cond.notify_all()

    def _watchdog_worker(self):
        """Se read() resta bloccato dentro V4L2 (USB staccata), release() dall'altro thread lo sblocca."""
        while self._running:
            time.sleep(0.5)
            if self.is_mock or not self._running:
                continue
            last = self._last_frame_mono
            if last <= 0.0:
                continue
            stalled_for = time.perf_counter() - last
            if stalled_for < 3.0:
                continue
            self.stalled = True
            print(f"[CAMERA] nessun frame da {stalled_for:.1f}s (USB?), riaggancio")
            self._reopen_device()
            time.sleep(1.0)

    def read(self, timeout: float = 0.5) -> Optional[StereoFrame]:
        """
        Returns the freshest stereo frame, waiting on condition variable if necessary.
        """
        with self._lock:
            if self._latest_frame is not None:
                frame = self._latest_frame
                self._latest_frame = None
                return frame

            if self._cond.wait(timeout=timeout):
                frame = self._latest_frame
                self._latest_frame = None
                return frame
            return None

    def release(self):
        """Releases the camera hardware and shuts down acquisition thread."""
        self._running = False
        with self._lock:
            self._cond.notify_all()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.is_opened = False
            print("[CAMERA] Dispositivo fotocamera rilasciato.")
