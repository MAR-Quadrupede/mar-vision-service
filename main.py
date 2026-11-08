#!/usr/bin/env python3
"""
M.A.R. — mar-vision-service Microservice Daemon

Realtime architecture (target ≥30 FPS display):
  • Capture + MJPEG overlay loop never waits on YOLO
  • Depth worker (StereoBM) runs in parallel; latest result wins
  • YOLO background thread (on by default, ~2 Hz) with per-object stereo distance/bearing
"""

import sys
import time
import json
import os
import threading
import cv2
import numpy as np

from config import parse_cli_arguments, AppConfig
from camera.stereo_camera import StereoCameraDriver
from camera.stereo_depth import StereoDepthEstimator
from detector.yolo_detector import YOLODetector
from detector.object_localizer import DetectionStabilizer, localize_objects
from server.mjpeg_server import MultiStreamServer
from safety_bands import (
    DIST_CRITICAL_CM,
    DIST_HIGH_CM,
    DIST_MODERATE_CM,
    classify_distance_cm,
)

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
# Keep a wide, clear operator view (full eye FOV). Depth can stay half-res internally.
STREAM_MAX_W = int(os.getenv("MAR_STREAM_MAX_W", "1280"))
# YOLO ON di default (thread separato, non blocca depth/stream). MAR_YOLO_ENABLE=0 per disattivarlo.
YOLO_ENABLE = os.getenv("MAR_YOLO_ENABLE", "1").strip().lower() in ("1", "true", "yes", "on")
YOLO_STRIDE_S = float(os.getenv("MAR_YOLO_PERIOD_S", "0.5"))
YOLO_IMGSZ = int(os.getenv("MAR_YOLO_IMGSZ", "256"))
# Gli oggetti YOLO restano validi per il decision engine per questo tempo dopo l'ultima inferenza
YOLO_OBJ_TTL_S = float(os.getenv("MAR_YOLO_OBJ_TTL_S", "1.5"))
TARGET_STREAM_FPS = float(os.getenv("MAR_TARGET_STREAM_FPS", "30"))


def _prep_stream_frame(frame):
    if frame is None:
        return None
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = frame.shape[:2]
    if w > STREAM_MAX_W:
        nh = max(1, int(h * STREAM_MAX_W / w))
        return cv2.resize(frame, (STREAM_MAX_W, nh), interpolation=cv2.INTER_AREA)
    return frame


def _draw_obstacle_box(frame_bgr, depth_res):
    """Obstacle bbox is computed in full-eye space (w_eye×h_eye); scale to preview."""
    if not depth_res or not depth_res.get("obstacle_found"):
        return frame_bgr
    bbox = depth_res.get("obstacle_bbox")
    if not bbox:
        return frame_bgr
    bx1, by1, bx2, by2 = bbox
    fh, fw = frame_bgr.shape[:2]
    # Canonical eye size from depth estimator (NOT half-res composite)
    ref_w = float(depth_res.get("eye_w") or fw)
    ref_h = float(depth_res.get("eye_h") or fh)
    if ref_w <= 1 or ref_h <= 1:
        return frame_bgr
    sx, sy = fw / ref_w, fh / ref_h
    x1, y1 = int(bx1 * sx), int(by1 * sy)
    x2, y2 = int(bx2 * sx), int(by2 * sy)
    dist = float(depth_res.get("obstacle_dist_cm") or depth_res.get("target_dist_cm") or 0)
    risk = str(depth_res.get("risk_level") or "").upper()
    if risk == "CRITICAL" or dist < DIST_CRITICAL_CM:
        color = (0, 0, 255)
    elif risk == "HIGH" or dist < DIST_HIGH_CM:
        color = (0, 140, 255)
    else:
        color = (0, 220, 0)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
    cv2.drawMarker(frame_bgr, ((x1 + x2) // 2, (y1 + y2) // 2), color, cv2.MARKER_CROSS, 18, 2)
    cv2.putText(
        frame_bgr,
        f"OSTACOLO {dist:.0f}cm",
        (max(8, x1), max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame_bgr


def _overlay_hud(frame_bgr, dist_cm, cmd, risk, fps_stream, fps_depth):
    out = frame_bgr
    color = {
        "CRITICAL": (0, 0, 255),
        "HIGH": (0, 128, 255),
        "MODERATE": (0, 220, 255),
        "LOW": (0, 220, 0),
    }.get(risk, (200, 200, 200))
    lines = [
        f"{cmd} | {risk} | Z={dist_cm:.0f}cm",
        f"stream {fps_stream:.0f}fps | depth {fps_depth:.0f}fps",
        f"bands C<{DIST_CRITICAL_CM:.0f} H<{DIST_HIGH_CM:.0f} M<{DIST_MODERATE_CM:.0f}",
    ]
    y = 28
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        y += 28
    if risk == "CRITICAL":
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
    return out


def _draw_yolo_objects(out, objs_payload, src_shape):
    """Disegna gli oggetti YOLO (box in coordinate rect_left) sul frame operatore, riscalando le box."""
    if not objs_payload or src_shape is None:
        return out
    sh, sw = int(src_shape[0]), int(src_shape[1])
    if sh <= 0 or sw <= 0:
        return out
    H, W = out.shape[:2]
    fx, fy = W / float(sw), H / float(sh)
    for o in objs_payload:
        try:
            x1, y1, x2, y2 = o["bbox"]
        except Exception:
            continue
        x1, x2 = int(x1 * fx), int(x2 * fx)
        y1, y2 = int(y1 * fy), int(y2 * fy)
        d = o.get("dist_cm")
        color = (0, 200, 255)
        if d is not None:
            if d < DIST_CRITICAL_CM:
                color = (0, 0, 255)
            elif d < DIST_HIGH_CM:
                color = (0, 120, 255)
            elif d < DIST_MODERATE_CM:
                color = (0, 220, 255)
            else:
                color = (80, 220, 80)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{o.get('class','?')} {o.get('confidence',0):.2f}"
        label += f" | {d:.0f}cm" if d is not None else " | ?cm"
        label += f" {o.get('bearing_deg',0):+.0f}deg"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        yb = max(y1, th + 8)
        cv2.rectangle(out, (x1, yb - th - 6), (x1 + tw + 6, yb + 2), color, -1)
        cv2.putText(out, label, (x1 + 3, yb - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def run_vision_service(config: AppConfig):
    print("=" * 72)
    print("  🐾 M.A.R. — 3D STEREO VISION & AUTONOMOUS DRIVING SERVICE")
    print("=" * 72)
    print(
        f"  • Safety bands cm: CRITICAL<{DIST_CRITICAL_CM:.0f}  "
        f"HIGH<{DIST_HIGH_CM:.0f}  MODERATE<{DIST_MODERATE_CM:.0f}"
    )
    print(f"  • Target stream FPS: {TARGET_STREAM_FPS:.0f} | YOLO={'ON' if YOLO_ENABLE else 'OFF'}")

    mqtt_client = None
    if mqtt:
        try:
            mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="mar-vision-service")
        except Exception:
            mqtt_client = mqtt.Client(client_id="mar-vision-service")
        mqtt_client.reconnect_delay_set(min_delay=1, max_delay=10)
        mqtt_client.connect_async(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print(f"  ✓ MQTT loop avviato verso {MQTT_BROKER}:{MQTT_PORT} (retry automatico)")

    # Force camera toward 30 FPS if config allows
    try:
        config.camera.fps = max(int(config.camera.fps), 30)
    except Exception:
        pass

    camera = StereoCameraDriver(config.camera, upside_down=True)
    if not camera.open():
        print("✗ Impossibile inizializzare la fotocamera stereo.")
        sys.exit(1)

    depth_estimator = StereoDepthEstimator(config.depth)
    print("  ✓ Pipeline depth realtime ATTIVA")

    detector = None
    if YOLO_ENABLE:
        model_override = os.getenv("MAR_YOLO_MODEL", "").strip()
        if model_override:
            config.detector.model_name_or_path = model_override
        try:
            detector = YOLODetector(config.detector)
            print(f"  ✓ YOLO attivo: {detector.model_name} | imgsz={YOLO_IMGSZ} | periodo={YOLO_STRIDE_S:.2f}s")
        except Exception as e:
            # La visione stereo/safety NON deve morire se il modello manca: prosegui senza oggetti
            print(f"  ⚠️ YOLO non disponibile ({e}): proseguo con sola stereo-depth")
            detector = None
    else:
        print("  ℹ️ YOLO disabilitato (MAR_YOLO_ENABLE=0) per FPS massimi")

    stream_server = MultiStreamServer(
        config.stream,
        command_callback=lambda action, val: depth_estimator.handle_command(action, val),
    )
    stream_server.start()

    # Shared state
    state_lock = threading.Lock()
    latest = {
        "depth_res": None,
        "left_objs": [],
        "fps_depth": 0.0,
        "fps_stream": 0.0,
        "frame_for_yolo": None,
        "left_objs_payload": [],   # oggetti arricchiti (dist_cm, bearing_deg, sector) pronti per MQTT
        "left_objs_ts": 0.0,
        "yolo_fps": 0.0,
    }
    stop_evt = threading.Event()
    depth_queue: list = []
    depth_q_lock = threading.Lock()

    def depth_worker():
        fcount = 0
        t0 = time.time()
        while not stop_evt.is_set():
            pair = None
            with depth_q_lock:
                if depth_queue:
                    pair = depth_queue.pop()  # latest only
                    depth_queue.clear()
            if pair is None:
                time.sleep(0.001)
                continue
            left_raw, right_raw = pair
            try:
                depth_res = depth_estimator.process_stereo_frame(left_raw, right_raw)
            except Exception as e:
                print(f"[DEPTH] errore: {e}")
                time.sleep(0.05)
                continue

            fcount += 1
            now = time.time()
            if now - t0 >= 1.0:
                fps_d = fcount / (now - t0)
                fcount = 0
                t0 = now
            else:
                fps_d = None

            with state_lock:
                latest["depth_res"] = depth_res
                if fps_d is not None:
                    latest["fps_depth"] = fps_d
                latest["frame_for_yolo"] = depth_res.get("rect_left")

            # MQTT as soon as depth is ready (decision latency)
            if mqtt_client and mqtt_client.is_connected():
                with state_lock:
                    objs_payload = (
                        list(latest["left_objs_payload"])
                        if (now - latest["left_objs_ts"]) <= YOLO_OBJ_TTL_S else []
                    )
                dist = float(depth_res["target_dist_cm"])
                payload = {
                    "timestamp": now,
                    "command": depth_res["driving_command"],
                    "reason": depth_res["driving_reason"],
                    "risk_level": depth_res.get("risk_level", classify_distance_cm(dist)),
                    "target_dist_cm": round(dist, 1),
                    "obstacle_near_cm": round(float(depth_res["obstacle_dist_cm"]), 1)
                    if depth_res.get("obstacle_found")
                    else 999.0,
                    "ground_hazard": depth_res.get("ground_hazard", False),
                    "z_basso_cm": round(float(depth_res.get("z_basso_cm", 999.0)), 1),
                    "z_centro_cm": round(float(depth_res.get("z_centro_cm", 999.0)), 1),
                    "fov_blocked": bool(depth_res.get("fov_blocked", False)),
                    "safety_alarm": depth_res["safety_alarm"],
                    "safety_reason": depth_res["safety_reason"],
                    "bands_cm": depth_res.get("bands_cm", {}),
                    "motion": depth_res.get("motion", {}),
                }
                mqtt_client.publish("mar/vision/command", json.dumps(payload))
                if depth_res.get("motion"):
                    mqtt_client.publish("mar/vision/motion", json.dumps(depth_res["motion"]))
                mqtt_client.publish(
                    "mar/vision/detections",
                    json.dumps(
                        {
                            "timestamp": now,
                            "spatial_grid_3d": depth_res.get("spatial_grid_3d", {}),
                            "motion": depth_res.get("motion", {}),
                            "z_basso_cm": payload["z_basso_cm"],
                            "z_centro_cm": payload["z_centro_cm"],
                            "z_alto_cm": round(float(depth_res.get("z_alto_cm", 999.0)), 1),
                            "ground_hazard": payload["ground_hazard"],
                            "target_dist_cm": payload["target_dist_cm"],
                            "risk_level": payload["risk_level"],
                            "objects_count": len(objs_payload),
                            "objects": objs_payload,
                            "objects_age_s": round(now - latest["left_objs_ts"], 2) if latest["left_objs_ts"] else None,
                            "yolo_enabled": detector is not None,
                        }
                    ),
                )
                if payload["ground_hazard"]:
                    mqtt_client.publish(
                        "mar/vision/ground_hazard",
                        json.dumps({"timestamp": now, "hazard": True, "z_basso_cm": payload["z_basso_cm"]}),
                    )
                mqtt_client.publish(
                    "mar/vision/status",
                    json.dumps(
                        {
                            "timestamp": now,
                            "fps_camera": round(camera.actual_fps, 1),
                            "fps_depth": round(latest["fps_depth"], 1),
                            "fps_stream": round(latest["fps_stream"], 1),
                            "risk_level": payload["risk_level"],
                            "fps_yolo": round(latest["yolo_fps"], 2),
                            "objects_count": len(objs_payload),
                            "objects": objs_payload,
                        }
                    ),
                )

    def yolo_worker():
        if detector is None:
            return
        t_last = time.time()
        stabilizer = DetectionStabilizer(needed=2)
        while not stop_evt.is_set():
            frame = None
            depth_snapshot = None
            with state_lock:
                frame = latest.get("frame_for_yolo")
                depth_snapshot = latest.get("depth_res")  # stessa coppia stereo del frame
            if frame is None:
                time.sleep(0.05)
                continue
            try:
                t0 = time.time()
                ann, objs = detector.detect_single(frame, "LEFT", imgsz=YOLO_IMGSZ)
                # Geometria 3D per oggetto: distanza (percentile Z), bearing, settore LiDAR-compatibile
                objs_payload = stabilizer.update(localize_objects(objs, depth_snapshot, frame.shape[:2]))
                dt = time.time() - t0
                with state_lock:
                    latest["left_objs"] = objs
                    latest["left_objs_payload"] = objs_payload
                    latest["left_objs_ts"] = time.time()
                    latest["yolo_ann"] = ann
                    latest["yolo_fps"] = 1.0 / max(1e-3, time.time() - t_last)
                t_last = time.time()
                if objs_payload and (int(t_last) % 5 == 0):
                    desc = ", ".join(
                        f"{o['class']}@{o['dist_cm'] if o['dist_cm'] is not None else '?'}cm/{o['bearing_deg']:+.0f}°"
                        for o in objs_payload[:4]
                    )
                    print(f"[YOLO] {len(objs_payload)} ogg. ({dt*1000:.0f}ms): {desc}")
            except Exception as e:
                print(f"[YOLO] {e}")
            stop_evt.wait(YOLO_STRIDE_S)

    threading.Thread(target=depth_worker, name="depth-worker", daemon=True).start()
    if detector is not None:
        threading.Thread(target=yolo_worker, name="yolo-worker", daemon=True).start()

    print("\n  🚀 PIPELINE VISIONE REALTIME")
    print(f"  📡 Console: http://{config.stream.host}:{config.stream.port}/")
    print("========================================================================\n")

    stream_count = 0
    t_stream = time.time()
    min_dt = 1.0 / max(1.0, TARGET_STREAM_FPS)

    try:
        while not stop_evt.is_set():
            t_loop = time.time()
            stereo_frame = camera.read(timeout=0.5)
            if stereo_frame is None:
                time.sleep(0.002)
                continue

            left_raw = stereo_frame.left
            right_raw = stereo_frame.right

            # Feed depth worker (keep only newest)
            with depth_q_lock:
                depth_queue.append((left_raw, right_raw))
                if len(depth_queue) > 1:
                    depth_queue[:] = depth_queue[-1:]

            with state_lock:
                depth_res = latest.get("depth_res")
                fps_depth = latest.get("fps_depth", 0.0)
                yolo_ann = latest.get("yolo_ann")
                objs_now = (
                    list(latest["left_objs_payload"])
                    if (time.time() - latest["left_objs_ts"]) <= YOLO_OBJ_TTL_S else []
                )

            # Wide clear operator view: ALWAYS full left eye FOV (not half-res rect/autofocus crop)
            preview = left_raw.copy()
            if preview.ndim == 2:
                preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

            if depth_res is not None:
                preview = _draw_obstacle_box(preview, depth_res)
            if objs_now:
                # box YOLO (coordinate rect_left) riscalate sul frame pieno, con distanza stereo e bearing
                src_shape = yolo_ann.shape[:2] if getattr(yolo_ann, "shape", None) is not None else None
                preview = _draw_yolo_objects(preview, objs_now, src_shape)

            dist_cm = float(depth_res["target_dist_cm"]) if depth_res else 999.0
            cmd = depth_res["driving_command"] if depth_res else "…"
            risk = depth_res.get("risk_level", classify_distance_cm(dist_cm)) if depth_res else "LOW"
            preview = _overlay_hud(
                _prep_stream_frame(preview),
                dist_cm,
                cmd,
                risk,
                latest.get("fps_stream", 0.0),
                fps_depth,
            )

            if config.stream.enabled:
                stream_server.update_frame("composite", preview)
                stream_server.update_frame("left", preview)
                # True camera eyes (no HUD) for audit / operator compare
                stream_server.update_frame("raw_left", _prep_stream_frame(left_raw))
                stream_server.update_frame("raw_right", _prep_stream_frame(right_raw))
                stream_server.update_frame("right", _prep_stream_frame(right_raw))
                if depth_res is not None:
                    stream_server.update_frame("map", depth_res.get("map_2d"))
                    stream_server.update_frame("depth", _prep_stream_frame(depth_res.get("composite")))
                diag = (depth_res or {}).get("diag") or {}
                stream_server.update_status(
                    fps=round(latest.get("fps_stream", 0.0), 1),
                    fps_camera=round(camera.actual_fps, 1),
                    fps_depth=round(fps_depth, 1),
                    modo=depth_res["current_mode"] if depth_res else 1,
                    auto_switch=depth_res["auto_switch"] if depth_res else True,
                    offset_x=depth_res["offset_x"] if depth_res else 0,
                    offset_y=depth_res["offset_y"] if depth_res else 0,
                    decisione_guida=cmd,
                    motivo_guida=depth_res["driving_reason"] if depth_res else "",
                    risk_level=risk,
                    ostacolo_centrale_cm=round(dist_cm, 1),
                    ostacolo_vicino_cm=round(float(depth_res["obstacle_dist_cm"]), 1)
                    if depth_res and depth_res.get("obstacle_found")
                    else 999.0,
                    ground_hazard=bool(depth_res.get("ground_hazard", False)) if depth_res else False,
                    allarme_lenti=depth_res["safety_alarm"] if depth_res else False,
                    allarme_motivo=depth_res["safety_reason"] if depth_res else "",
                    model=(detector.model_name if detector else "OFF"),
                    detections_count=len(objs_now),
                    detections=objs_now,
                    fov_blocked=bool((depth_res or {}).get("fov_blocked", False)),
                    motion=(depth_res or {}).get("motion") or {},
                    diag=diag,
                    bands_cm={
                        "critical": DIST_CRITICAL_CM,
                        "high": DIST_HIGH_CM,
                        "moderate": DIST_MODERATE_CM,
                    },
                )

            stream_count += 1
            now = time.time()
            if now - t_stream >= 1.0:
                with state_lock:
                    latest["fps_stream"] = stream_count / (now - t_stream)
                stream_count = 0
                t_stream = now

            # Pace toward target stream FPS (don't spin hotter than needed)
            elapsed = time.time() - t_loop
            if elapsed < min_dt:
                time.sleep(min_dt - elapsed)

    except KeyboardInterrupt:
        print("\nArresto manuale richiesto...")
    finally:
        stop_evt.set()
        print("\n[CLEANUP] Rilascio risorse telecamera e server...")
        stream_server.stop()
        camera.release()
        cv2.destroyAllWindows()


def main():
    config = parse_cli_arguments()
    # Prefer headless + lower jpeg for speed on edge
    config.headless = True
    config.stream.jpeg_quality = min(int(config.stream.jpeg_quality), 65)
    run_vision_service(config)


if __name__ == "__main__":
    main()
