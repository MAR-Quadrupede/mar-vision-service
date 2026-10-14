#!/usr/bin/env python3
"""
M.A.R. — Universal YOLO Detector for 3D Stereo Vision
Supports all YOLO variants (v5, v8, v9, v10, v11, v26+; n, s, m, l, x),
synchronized stereo batch inference, bounding box extraction, and overlay HUD.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
import sys
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from ..config import DetectorConfig
except (ImportError, ValueError):
    from config import DetectorConfig


@dataclass
class DetectedObject:
    """Standardized representation of a detected object."""
    label: str
    class_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]          # (cx, cy)
    relative_size: float             # Box area / Frame area (0.0 to 1.0)
    source: str                      # 'LEFT', 'RIGHT', or 'MONO'
    estimated_distance_cm: Optional[float] = None
    in_danger_zone: bool = False
    danger_level: int = 0            # 0=Safe, 1=Caution, 2=Warning, 3=Danger, 4=Emergency
    danger_name: str = "SAFE"


# Generate distinct deterministic colors for 80 COCO classes
np.random.seed(42)
CLASS_COLORS = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)


class YOLODetector:
    """
    Manages loading, inference, and rendering for Ultralytics YOLO models.
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.model = None
        self.model_name = ""
        self._load_model()

    def _load_model(self):
        """Loads local model weights or triggers automatic download."""
        if YOLO is None:
            raise ImportError(
                "La libreria 'ultralytics' non è installata nel sistema. "
                "Esegui: pip install ultralytics"
            )

        model_path = self.config.model_name_or_path
        self.model_name = os.path.basename(model_path).replace('.pt', '').upper()

        print(f"\n[DETECTOR] Caricamento modello YOLO: {self.model_name}")
        print(f"  • Sorgente/File: {model_path}")

        try:
            self.model = YOLO(model_path)
            # Configure default thresholds
            self.model.conf = self.config.confidence_threshold
            self.model.iou = self.config.iou_threshold
            if self.config.device:
                self.model.to(self.config.device)

            print(f"  ✓ Modello {self.model_name} caricato con successo!")
            print(f"  • Task: {getattr(self.model, 'task', 'detect')}")
            print(f"  • Classi riconosciute: {len(self.model.names)}")
        except Exception as e:
            print(f"  ✗ Errore nel caricamento del modello {model_path}: {e}")
            print("  Tentativo di fallback sul modello nano ufficiale yolov8n.pt...")
            try:
                self.model = YOLO("yolov8n.pt")
                self.model_name = "YOLOV8N"
                print("  ✓ Fallback su yolov8n.pt completato.")
            except Exception as e2:
                print(f"  ✗ Errore fatale: {e2}")
                raise

    def detect_single(self, frame: np.ndarray, source: str = "LEFT") -> Tuple[np.ndarray, List[DetectedObject]]:
        """
        Runs object detection on a single camera frame.
        Returns annotated frame and list of DetectedObject.
        """
        h, w = frame.shape[:2]
        frame_area = max(1, w * h)

        results = self.model(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            verbose=False
        )

        detected_objects: List[DetectedObject] = []
        annotated = frame.copy()

        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self.model.names.get(cls_id, f"obj_{cls_id}")

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)
                rel_size = float(area) / frame_area

                obj = DetectedObject(
                    label=label,
                    class_id=cls_id,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    relative_size=rel_size,
                    source=source
                )
                detected_objects.append(obj)

        return annotated, detected_objects

    def detect_stereo(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[DetectedObject], List[DetectedObject]]:
        """
        Runs batch or synchronized inference on left and right frames.
        Returns (annotated_left, annotated_right, left_objects, right_objects).
        """
        # Batch inference in a single call for high throughput
        batch_results = self.model(
            [left_frame, right_frame],
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            verbose=False
        )

        left_ann, left_objs = self._extract_results(left_frame, batch_results[0], "LEFT")
        right_ann, right_objs = self._extract_results(right_frame, batch_results[1], "RIGHT")

        return left_ann, right_ann, left_objs, right_objs

    def _extract_results(
        self,
        frame: np.ndarray,
        result,
        source: str
    ) -> Tuple[np.ndarray, List[DetectedObject]]:
        """Extracts detections from a single result object."""
        h, w = frame.shape[:2]
        frame_area = max(1, w * h)
        annotated = frame.copy()
        objects: List[DetectedObject] = []

        if result is not None and result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self.model.names.get(cls_id, f"obj_{cls_id}")

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)
                rel_size = float(area) / frame_area

                obj = DetectedObject(
                    label=label,
                    class_id=cls_id,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    relative_size=rel_size,
                    source=source
                )
                objects.append(obj)

        return annotated, objects

    def render_overlay(
        self,
        frame: np.ndarray,
        objects: List[DetectedObject],
        zone_left: int,
        zone_right: int,
        eye_label: str = ""
    ) -> np.ndarray:
        """
        Renders bounding boxes, risk badges, danger zone markings and HUD on the frame.
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # Draw danger zone boundary
        cv2.rectangle(canvas, (zone_left, 0), (zone_right, h), (0, 140, 255), 2)
        zone_text = f"ZONA PERICOLO ({eye_label})" if eye_label else "ZONA PERICOLO"
        cv2.putText(canvas, zone_text, (zone_left + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)

        for obj in objects:
            x1, y1, x2, y2 = obj.bbox
            cls_color = [int(c) for c in CLASS_COLORS[obj.class_id % len(CLASS_COLORS)]]

            # Choose color based on combined danger
            if obj.danger_level >= 4:
                box_color = (0, 0, 255)      # Red - Emergency
                border_th = 3
            elif obj.danger_level == 3:
                box_color = (0, 120, 255)    # Orange - Danger
                border_th = 3
            elif obj.danger_level == 2:
                box_color = (0, 220, 255)    # Yellow - Warning
                border_th = 2
            elif obj.danger_level == 1:
                box_color = (0, 255, 180)    # Light Green - Caution
                border_th = 2
            else:
                box_color = cls_color        # Safe
                border_th = 2

            # Bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, border_th)

            # Center target point
            cv2.circle(canvas, obj.center, 4, box_color, -1)

            # Label text
            badge = f"{obj.label} {obj.confidence:.2f}"
            if obj.estimated_distance_cm is not None and obj.estimated_distance_cm > 0:
                badge += f" | {obj.estimated_distance_cm:.0f}cm"
            else:
                badge += f" | {obj.relative_size*100:.1f}%"

            (tw, th), baseline = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_badge = max(y1, th + 10)
            cv2.rectangle(canvas, (x1, y_badge - th - 6), (x1 + tw + 6, y_badge + 2), box_color, -1)
            cv2.putText(canvas, badge, (x1 + 3, y_badge - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return canvas
