#!/usr/bin/env python3
"""
M.A.R. — Localizzazione 3D degli oggetti YOLO tramite la nuvola stereo.

Per ogni bounding box (in coordinate di `rect_left`) si campiona la regione centrale
della nuvola 3D (`points_robot_cm`, frame robot: X destra, Y basso, Z avanti) e si stima:

  • dist_cm      → percentile basso robusto della Z valida (l'oggetto è la cosa più vicina
                   dentro la sua box; il percentile ignora i buchi SGBM e lo sfondo)
  • bearing_deg  → atan2(X, Z) sul mediano dei punti validi (positivo = destra)
                   con fallback pinhole sul centro box se la nuvola è vuota
  • sector       → settore a 45° coerente con il LiDAR (Fronte / Front-DX / Front-SX ...)
  • depth_quality→ frazione di pixel validi nella regione campionata (0..1)

La stereocamera ha molti falsi positivi sulla DISTANZA: qui si segnala la qualità in
modo che il decision engine possa pesarla (W_STEREO) e preferire il LiDAR del settore.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Z_MIN_CM = 15.0
Z_MAX_CM = 600.0
MIN_VALID_PX = 12
DIST_PERCENTILE = 20.0   # percentile basso robusto
CORE_FRACTION = 0.6      # frazione centrale della box campionata (evita i bordi/sfondo)

_SECTOR_BY_INDEX = ("Fronte", "Front-DX", "Destra", "Retro-DX", "Retro", "Retro-SX", "Sinistra", "Front-SX")


def bearing_to_sector(bearing_deg: float) -> str:
    a = (float(bearing_deg) + 360.0) % 360.0
    return _SECTOR_BY_INDEX[int(((a + 22.5) % 360.0) // 45.0)]


def localize_object(
    bbox: Tuple[int, int, int, int],
    points_robot_cm: Optional[np.ndarray],
    points_scale: float,
    frame_w: int,
    frame_h: int,
    focal_rect_px: Optional[float] = None,
) -> Dict[str, Any]:
    """Ritorna dict {dist_cm, bearing_deg, sector, depth_quality, x_cm, z_cm}."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, x2 = max(0, min(x1, x2)), min(frame_w - 1, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(frame_h - 1, max(y1, y2))
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    dist: Optional[float] = None
    bearing: Optional[float] = None
    quality = 0.0
    x_med: Optional[float] = None
    z_med: Optional[float] = None

    if points_robot_cm is not None and points_robot_cm.ndim == 3 and points_scale > 0:
        H, W = points_robot_cm.shape[:2]
        bw, bh = (x2 - x1), (y2 - y1)
        # regione centrale della box (60%) → coordinate nuvola
        cx1 = int((cx - bw * CORE_FRACTION / 2.0) / points_scale)
        cx2 = int((cx + bw * CORE_FRACTION / 2.0) / points_scale) + 1
        cy1 = int((cy - bh * CORE_FRACTION / 2.0) / points_scale)
        cy2 = int((cy + bh * CORE_FRACTION / 2.0) / points_scale) + 1
        cx1, cx2 = max(0, cx1), min(W, max(cx1 + 1, cx2))
        cy1, cy2 = max(0, cy1), min(H, max(cy1 + 1, cy2))
        if cx2 > cx1 and cy2 > cy1:
            roi = points_robot_cm[cy1:cy2, cx1:cx2, :]
            Z = roi[:, :, 2]
            valid = np.isfinite(Z) & (Z > Z_MIN_CM) & (Z < Z_MAX_CM)
            n_valid = int(np.count_nonzero(valid))
            quality = n_valid / float(max(1, Z.size))
            if n_valid >= MIN_VALID_PX:
                zv = Z[valid]
                xv = roi[:, :, 0][valid]
                dist = float(np.percentile(zv, DIST_PERCENTILE))
                z_med = float(np.median(zv))
                x_med = float(np.median(xv))
                bearing = math.degrees(math.atan2(x_med, max(1e-3, z_med)))

    if bearing is None:
        # fallback pinhole: centro box rispetto all'asse ottico
        f = float(focal_rect_px) if focal_rect_px else 0.85 * float(frame_w)  # ~60° HFOV
        bearing = math.degrees(math.atan((cx - frame_w / 2.0) / max(1.0, f)))

    out: Dict[str, Any] = {
        "dist_cm": round(dist, 1) if dist is not None else None,
        "bearing_deg": round(bearing, 1),
        "sector": bearing_to_sector(bearing),
        "depth_quality": round(quality, 2),
        "x_cm": round(x_med, 1) if x_med is not None else None,
        "z_cm": round(z_med, 1) if z_med is not None else None,
    }
    return out


def localize_objects(objects: List[Any], depth_res: Optional[Dict[str, Any]], frame_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
    """Arricchisce una lista di DetectedObject con la geometria 3D. Ritorna la lista di dict pronta per MQTT."""
    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
    pts = depth_res.get("points_robot_cm") if depth_res else None
    scale = float(depth_res.get("points_scale", 2.0)) if depth_res else 2.0
    focal = depth_res.get("focal_rect_px") if depth_res else None
    out: List[Dict[str, Any]] = []
    for o in objects:
        geo = localize_object(o.bbox, pts, scale, frame_w, frame_h, focal)
        try:
            o.estimated_distance_cm = geo["dist_cm"]
        except Exception:
            pass
        out.append({
            "class": o.label,
            "confidence": round(float(o.confidence), 2),
            "bbox": [int(x) for x in o.bbox],
            "dist_cm": geo["dist_cm"],
            "estimated_distance_cm": geo["dist_cm"],
            "bearing_deg": geo["bearing_deg"],
            "sector": geo["sector"],
            "depth_quality": geo["depth_quality"],
            "x_cm": geo["x_cm"],
            "z_cm": geo["z_cm"],
            "relative_size": round(float(getattr(o, "relative_size", 0.0)), 4),
        })
    return out


class DetectionStabilizer:
    """Pubblica un oggetto solo se la stessa classe resta nello stesso settore per `needed` inferenze.

    yolov8n a 256 px sulla stereocamera alterna classi (persona/letto/ombrello) sullo stesso
    fotogramma: una detection isolata non deve entrare in Q_s.
    """

    def __init__(self, needed: int = 2):
        self.needed = max(1, int(needed))
        self._streak: Dict[Tuple[str, str], int] = {}

    def update(self, objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: Dict[Tuple[str, str], int] = {}
        for o in objs:
            key = (str(o.get("class") or ""), str(o.get("sector") or ""))
            seen[key] = self._streak.get(key, 0) + 1
        self._streak = seen
        confirmed: List[Dict[str, Any]] = []
        emitted = set()
        for o in objs:
            key = (str(o.get("class") or ""), str(o.get("sector") or ""))
            if key in emitted or seen[key] < self.needed:
                continue
            emitted.add(key)
            confirmed.append(o)
        return confirmed
