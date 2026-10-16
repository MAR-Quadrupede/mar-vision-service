#!/usr/bin/env python3
"""
M.A.R. — Stereovisione Configuration Module
Defines system configuration dataclasses, parameters, and CLI argument parsing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import argparse
import os
import re
import sys


@dataclass
class CameraConfig:
    """Hardware and acquisition parameters for stereo camera."""
    primary_index: int = 0
    fallback_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    frame_width: int = 2560      # Native side-by-side resolution (1280 per eye)
    frame_height: int = 800      # Native height
    fps: int = 30
    auto_focus: bool = False     # Disabled for Global Shutter stability
    auto_exposure: bool = False  # Disabled for Global Shutter stability
    horizontal_split: bool = True


@dataclass
class DetectorConfig:
    """YOLO detection parameters."""
    model_name_or_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.30
    iou_threshold: float = 0.45
    device: str = ""             # Auto ('cpu', 'cuda', 'mps')
    enable_stereo_sync: bool = True
    local_model_dirs: List[Path] = field(default_factory=lambda: [
        Path(__file__).parent / "models",
        Path(__file__).parent,
        Path(__file__).parent.parent
    ])


@dataclass
class DepthConfig:
    """Disparity map and depth estimation parameters (OpenCV StereoSGBM / StereoBM)."""
    enabled: bool = False
    min_disparity: int = 0
    num_disparities: int = 64     # Must be divisible by 16
    block_size: int = 5           # Optimal window size for robot ELP stereo camera
    baseline_cm: float = 6.0      # Physical distance between left & right lenses
    focal_length_px: float = 1006.5 # Default focal length in pixels (from mtx_L)
    calibration_file: Optional[str] = "parametri_calibrazione_stereo.npz"
    use_wls_filter: bool = True
    enable_spatial_grid: bool = True
    ground_hazard_threshold_cm: float = 40.0 # Trigger STOP_TERRA when lower ground obstacle detected


@dataclass
class NavigationConfig:
    """Spatial risk assessment and obstacle avoidance parameters."""
    danger_zone_width_ratio: float = 0.70  # Central 70% width considered critical path
    safe_distance_cm: float = 100.0
    warning_distance_cm: float = 50.0
    danger_distance_cm: float = 30.0
    emergency_distance_cm: float = 15.0

    # Visual relative area thresholds (percentage of eye frame)
    visual_caution_ratio: float = 0.02
    visual_warning_ratio: float = 0.05
    visual_danger_ratio: float = 0.12
    visual_emergency_ratio: float = 0.25

    ultrasonic_weight: float = 0.80       # Fusing: 80% ultrasonic sensor, 20% visual area


@dataclass
class StreamConfig:
    """MJPEG HTTP Streaming server parameters for dashboard-mar-quadrupede."""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8443
    jpeg_quality: int = 85
    target_fps: int = 30


@dataclass
class ArduinoConfig:
    """Serial communication bridge with Low-Level Controller."""
    enabled: bool = True
    port: Optional[str] = None    # Auto-detect if None
    baudrate: int = 9600          # Legacy 9600 or M.A.R. 115200
    timeout: float = 1.0
    send_mar_protocol: bool = True # Send CMD:FORWARD etc. and fallback single-char


@dataclass
class AppConfig:
    """Global configuration container."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    arduino: ArduinoConfig = field(default_factory=ArduinoConfig)
    headless: bool = False
    interactive_model_select: bool = False


def resolve_model_file(model_spec: str, search_dirs: List[Path]) -> str:
    """
    Resolves local model file or returns model specification string for Ultralytics auto-download.
    """
    if os.path.exists(model_spec):
        return str(Path(model_spec).resolve())

    # Check search dirs
    filename = model_spec if model_spec.endswith(".pt") else f"{model_spec}.pt"
    for d in search_dirs:
        candidate = d / filename
        if candidate.exists():
            return str(candidate.resolve())

    # Return spec (Ultralytics will download)
    return filename


def interactive_model_selector() -> str:
    """
    Interactive CLI menu for YOLO model selection (v5 through v26+).
    """
    min_version = 5
    supported_variants = ['n', 's', 'm', 'l', 'x']
    v_prefix_versions = {5, 8}

    print("\n" + "=" * 62)
    print("           🐾 M.A.R. — SELEZIONE MODELLO YOLO")
    print("=" * 62)
    print("Modelli supportati:")
    print("  • Versioni: YOLOv5, YOLOv8, YOLO11, YOLO26, ...")
    print("  • Varianti: n (nano), s (small), m (medium), l (large), x (extra-large)")
    print("\nVersioni popolari: v8 (consigliata per RPi 5), v11, v26")
    print("Esempi: yolov8n, yolov8s, yolo11n, yolo26x")
    print("Nota: I pesi verranno cercati in ./models o scaricati automaticamente.")
    print("=" * 62)

    while True:
        model_input = input("\nInserisci modello da usare (default: yolov8n): ").strip().lower()
        if not model_input:
            model_input = "yolov8n"

        match = re.match(r'^yolo(v?)(\d+)([nsmlex])$', model_input)
        if not match:
            print("✗ Formato non valido! Usa: yolov8n, yolov8x, yolo11n, yolo26x, ecc.")
            continue

        has_v = bool(match.group(1))
        version = int(match.group(2))
        variant = match.group(3)

        if version < min_version:
            print(f"✗ Versione {version} non supportata. Versione minima: YOLOv{min_version}")
            continue

        expects_v = version in v_prefix_versions
        if expects_v and not has_v:
            print(f"✗ Per YOLO {version} usa la convenzione con 'v': yolov{version}{variant}")
            continue
        if not expects_v and has_v:
            print(f"✗ Per YOLO {version} non usare la 'v': yolo{version}{variant}")
            continue

        if variant not in supported_variants:
            print(f"✗ Variante non riconosciuta ({variant}). Usa una tra: {supported_variants}")
            continue

        prefix = "yolov" if expects_v else "yolo"
        norm_name = f"{prefix}{version}{variant}.pt"
        print(f"✓ Modello confermato: {norm_name}")
        return norm_name


def parse_cli_arguments() -> AppConfig:
    """Parses command line arguments and returns an AppConfig instance."""
    parser = argparse.ArgumentParser(
        description="M.A.R. — Sistema Definitivo Stereovisione 3D, YOLO Edge AI e Streaming",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # General / execution mode
    parser.add_argument("--interactive", action="store_true", help="Avvia menu interattivo per la scelta del modello YOLO")
    parser.add_argument("--headless", action="store_true", help="Esegui senza finestre GUI OpenCV locali")
    parser.add_argument("--no-stream", action="store_true", help="Disabilita il server di streaming HTTP")
    parser.add_argument("--no-arduino", action="store_true", help="Disabilita la connessione seriale ad Arduino")

    # Camera arguments
    parser.add_argument("--camera", type=int, default=0, help="Indice primario della stereocamera USB")
    parser.add_argument("--width", type=int, default=2560, help="Larghezza totale frame side-by-side (1280 per occhio)")
    parser.add_argument("--height", type=int, default=800, help="Altezza del frame")
    parser.add_argument("--fps", type=int, default=30, help="FPS target cattura video")

    # Detector arguments
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Nome o path del modello YOLO (es. yolov8n.pt, yolo11n.pt)")
    parser.add_argument("--conf", type=float, default=0.30, help="Soglia minima di confidenza per le detection")
    parser.add_argument("--iou", type=float, default=0.45, help="Soglia IoU per Non-Maximum Suppression")
    parser.add_argument("--device", type=str, default="", help="Device per inferenza PyTorch ('cpu', 'cuda', 'mps')")

    # Depth arguments
    parser.add_argument("--depth", action="store_true", help="Abilita calcolo mappa di disparità stereoscopica (StereoSGBM)")

    # Server arguments
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Indirizzo IP di ascolto dello stream HTTP")
    parser.add_argument("--port", type=int, default=8443, help="Porta TCP per il server streaming MJPEG (default Dashboard: 8443)")
    parser.add_argument("--quality", type=int, default=85, help="Qualità JPEG di compressione stream (1-100)")

    # Arduino arguments
    parser.add_argument("--serial-port", type=str, default=None, help="Porta seriale Arduino (es. /dev/ttyACM0 o None per auto-detect)")
    parser.add_argument("--baudrate", type=int, default=9600, help="Baudrate seriale Arduino")

    args = parser.parse_args()

    config = AppConfig()
    config.headless = args.headless
    config.interactive_model_select = args.interactive

    # Camera
    config.camera.primary_index = args.camera
    config.camera.frame_width = args.width
    config.camera.frame_height = args.height
    config.camera.fps = args.fps

    # Detector
    if args.interactive:
        selected = interactive_model_selector()
        config.detector.model_name_or_path = selected
    else:
        config.detector.model_name_or_path = args.model

    config.detector.confidence_threshold = args.conf
    config.detector.iou_threshold = args.iou
    config.detector.device = args.device

    # Depth
    config.depth.enabled = args.depth

    # Stream
    config.stream.enabled = not args.no_stream
    config.stream.host = args.host
    config.stream.port = args.port
    config.stream.jpeg_quality = args.quality

    # Arduino
    config.arduino.enabled = not args.no_arduino
    config.arduino.port = args.serial_port
    config.arduino.baudrate = args.baudrate

    # Resolve local model path
    config.detector.model_name_or_path = resolve_model_file(
        config.detector.model_name_or_path,
        config.detector.local_model_dirs
    )

    return config
