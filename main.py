#!/usr/bin/env python3
"""
M.A.R. — mar-vision-service Microservice Daemon
Synchronized stereo acquisition, fast rectification, epipolar alignment, SGBM 3D corridor depth,
dynamic Auto-Focus obstacle tracking, 2D top-down navigation mapping, YOLO detection,
interactive multi-stream MJPEG HTTP server, and MQTT autonomous driving publisher.
"""

import sys
import time
import json
import os
import cv2
import numpy as np

from config import parse_cli_arguments, AppConfig
from camera.stereo_camera import StereoCameraDriver
from camera.stereo_depth import StereoDepthEstimator
from detector.yolo_detector import YOLODetector
from server.mjpeg_server import MultiStreamServer

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
# YOLO every N depth frames (1=every frame). On RPi default 3 keeps UI fluid.
YOLO_STRIDE = max(1, int(os.getenv("MAR_YOLO_STRIDE", "3")))
# Max width for MJPEG encode (original script used 960)
STREAM_MAX_W = int(os.getenv("MAR_STREAM_MAX_W", "960"))


def _prep_stream_frame(frame):
    """Downscale large frames before JPEG encode to cut encode+Tailscale cost."""
    if frame is None:
        return None
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = frame.shape[:2]
    if w > STREAM_MAX_W:
        nh = int(h * STREAM_MAX_W / w)
        return cv2.resize(frame, (STREAM_MAX_W, nh), interpolation=cv2.INTER_AREA)
    return frame


def run_vision_service(config: AppConfig):
    print("=" * 72)
    print("  🐾 M.A.R. — 3D STEREO VISION & AUTONOMOUS DRIVING SERVICE")
    print("=" * 72)

    # 1. Connessione MQTT
    mqtt_client = None
    if mqtt:
        try:
            mqtt_client = mqtt.Client(client_id="mar-vision-service")
            mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            mqtt_client.loop_start()
            print(f"  ✓ Connesso a MQTT Broker ({MQTT_BROKER}:{MQTT_PORT})")
        except Exception as e:
            print(f"  ⚠️ MQTT non disponibile ({e}). Esecuzione standalone.")

    # 2. Hardware Fotocamera Stereo (con thread non bloccante e rotazione 180°)
    camera = StereoCameraDriver(config.camera, upside_down=True)
    if not camera.open():
        print("✗ Impossibile inizializzare la fotocamera stereo.")
        sys.exit(1)

    # 3. Modulo Disparità 3D, Allineamento Epipolare e Mappa Top-Down
    depth_estimator = StereoDepthEstimator(config.depth)
    print("  ✓ Pipeline 3D SGBM, Auto-Focus Ostacoli & Allineamento Epipolare: ATTIVO")

    # 4. Modello YOLO Edge AI
    detector = YOLODetector(config.detector)

    # 5. Server di Streaming MJPEG & Console Web Interattiva
    stream_server = MultiStreamServer(
        config.stream,
        command_callback=lambda action, val: depth_estimator.handle_command(action, val)
    )
    stream_server.start()

    print("\n  🚀 PIPELINE VISIONE ATTIVA")
    print(f"  📡 Console Web & Streaming: http://{config.stream.host}:{config.stream.port}/")
    print("========================================================================\n")

    fps_depth = 0.0
    frame_count = 0
    t_fps_start = time.time()
    last_mqtt_pub = 0.0
    loop_i = 0
    cached_left_ann = None
    cached_right_ann = None
    cached_left_objs: list = []
    print(f"  • YOLO stride: ogni {YOLO_STRIDE} frame | stream max width: {STREAM_MAX_W}px")

    try:
        while True:
            # 1. Cattura frame stereo sincronizzato
            stereo_frame = camera.read(timeout=0.5)
            if stereo_frame is None:
                time.sleep(0.005)
                continue

            frame_count += 1
            loop_i += 1
            left_raw = stereo_frame.left
            right_raw = stereo_frame.right

            # 2. Pipeline Stereoscopica 3D (Rettificazione, Disparità, Auto-Focus, Mappa, Decisione Guida)
            depth_res = depth_estimator.process_stereo_frame(left_raw, right_raw)

            composite = depth_res["composite"]
            map_2d = depth_res["map_2d"]
            driving_cmd = depth_res["driving_command"]
            driving_reason = depth_res["driving_reason"]
            target_dist = depth_res["target_dist_cm"]
            obs_dist = depth_res["obstacle_dist_cm"] if depth_res["obstacle_found"] else 999.0
            ground_hazard = (target_dist < 45.0)

            # 3. YOLO left-only, throttled — depth/UI stay realtime when you walk in front
            if (loop_i % YOLO_STRIDE) == 1 or cached_left_ann is None:
                left_ann, right_ann, left_objs, right_objs = detector.detect_stereo(
                    depth_res["rect_left"],
                    depth_res["rect_right"]
                )
                cached_left_ann, cached_right_ann = left_ann, right_ann
                cached_left_objs = left_objs
            else:
                left_ann = cached_left_ann
                right_ann = cached_right_ann
                left_objs = cached_left_objs
                right_objs = []
            all_objects = left_objs + right_objs

            # 4. Pubblicazione Decisione di Guida e Rilevamenti su MQTT
            now = time.time()
            if mqtt_client and mqtt_client.is_connected() and (now - last_mqtt_pub > 0.05):
                last_mqtt_pub = now

                # Decisione di marcia immediata per mar-decision-engine
                decision_payload = {
                    "timestamp": now,
                    "command": driving_cmd,
                    "reason": driving_reason,
                    "target_dist_cm": round(float(target_dist), 1),
                    "obstacle_near_cm": round(float(obs_dist), 1),
                    "ground_hazard": depth_res.get("ground_hazard", ground_hazard),
                    "z_basso_cm": round(float(depth_res.get("z_basso_cm", 999.0)), 1),
                    "z_centro_cm": round(float(depth_res.get("z_centro_cm", 999.0)), 1),
                    "safety_alarm": depth_res["safety_alarm"],
                    "safety_reason": depth_res["safety_reason"]
                }
                mqtt_client.publish("mar/vision/command", json.dumps(decision_payload))

                # Griglia 3D, settori spaziali e oggetti per la sensor fusion
                detections_payload = {
                    "timestamp": now,
                    "spatial_grid_3d": depth_res.get("spatial_grid_3d", {}),
                    "z_basso_cm": round(float(depth_res.get("z_basso_cm", 999.0)), 1),
                    "z_centro_cm": round(float(depth_res.get("z_centro_cm", 999.0)), 1),
                    "z_alto_cm": round(float(depth_res.get("z_alto_cm", 999.0)), 1),
                    "ground_hazard": depth_res.get("ground_hazard", ground_hazard),
                    "target_dist_cm": round(float(target_dist), 1),
                    "objects_count": len(all_objects),
                    "objects": [
                        {
                            "class": o.label,
                            "confidence": round(float(o.confidence), 2),
                            "bbox": [int(x) for x in o.bbox]
                        }
                        for o in left_objs
                    ]
                }
                mqtt_client.publish("mar/vision/detections", json.dumps(detections_payload))

                # Allarme ostacolo a terra se presente
                if depth_res.get("ground_hazard", False) or ground_hazard:
                    ground_payload = {
                        "timestamp": now,
                        "hazard": True,
                        "z_basso_cm": round(float(depth_res.get("z_basso_cm", 999.0)), 1)
                    }
                    mqtt_client.publish("mar/vision/ground_hazard", json.dumps(ground_payload))

                # Oggetti YOLO e telemetria
                telemetry_payload = {
                    "timestamp": now,
                    "fps_camera": round(camera.actual_fps, 1),
                    "fps_depth": round(fps_depth, 1),
                    "objects_count": len(all_objects),
                    "objects": [
                        {
                            "class": o.label,
                            "confidence": round(float(o.confidence), 2),
                            "bbox": [int(x) for x in o.bbox]
                        }
                        for o in left_objs
                    ]
                }
                mqtt_client.publish("mar/vision/status", json.dumps(telemetry_payload))

            # 5. Aggiornamento Frame e Telemetria sul Server MJPEG (downscaled JPEG)
            if config.stream.enabled:
                stream_server.update_frame("composite", _prep_stream_frame(composite))
                stream_server.update_frame("map", map_2d)
                stream_server.update_frame("left", _prep_stream_frame(left_ann if left_ann is not None else depth_res["rect_left"]))
                stream_server.update_frame("right", _prep_stream_frame(right_ann if right_ann is not None else depth_res["rect_right"]))
                # Raw full-res only if someone is watching (skip encode cost otherwise)
                if stream_server.active_clients() > 0:
                    stream_server.update_frame("raw_left", _prep_stream_frame(left_raw))
                    stream_server.update_frame("raw_right", _prep_stream_frame(right_raw))

                stream_server.update_status(
                    fps=round(fps_depth, 1),
                    fps_camera=round(camera.actual_fps, 1),
                    fps_depth=round(fps_depth, 1),
                    modo=depth_res["current_mode"],
                    auto_switch=depth_res["auto_switch"],
                    offset_x=depth_res["offset_x"],
                    offset_y=depth_res["offset_y"],
                    decisione_guida=driving_cmd,
                    motivo_guida=driving_reason,
                    ostacolo_centrale_cm=round(float(target_dist), 1),
                    ostacolo_vicino_cm=round(float(obs_dist), 1),
                    ground_hazard=ground_hazard,
                    allarme_lenti=depth_res["safety_alarm"],
                    allarme_motivo=depth_res["safety_reason"],
                    model=detector.model_name,
                    detections_count=len(all_objects)
                )

            # 6. Finestra GUI locale se non headless
            if not config.headless:
                cv2.imshow("M.A.R. Vision Service Monitor", composite)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            # 7. Calcolo FPS
            if time.time() - t_fps_start >= 1.0:
                fps_depth = frame_count / (time.time() - t_fps_start)
                frame_count = 0
                t_fps_start = time.time()

    except KeyboardInterrupt:
        print("\nArresto manuale richiesto...")
    finally:
        print("\n[CLEANUP] Rilascio risorse telecamera e server...")
        stream_server.stop()
        camera.release()
        cv2.destroyAllWindows()


def main():
    config = parse_cli_arguments()
    run_vision_service(config)


if __name__ == "__main__":
    main()
