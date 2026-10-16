#!/usr/bin/env python3
"""
M.A.R. — mar-vision-service Microservice Daemon
Acquires synchronized stereo video, estimates 3D depth, performs YOLO inference,
serves multi-stream MJPEG HTTP feeds, and publishes 3D detections to MQTT.
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


def run_vision_service(config: AppConfig):
    print("=" * 70)
    print("  🐾 M.A.R. — 3D STEREO VISION & EDGE AI SERVICE")
    print("=" * 70)

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

    # 2. Hardware Fotocamera Stereo
    camera = StereoCameraDriver(config.camera)
    if not camera.open():
        print("✗ Impossibile inizializzare la fotocamera stereo.")
        sys.exit(1)

    # 3. Modello YOLO Edge AI
    detector = YOLODetector(config.detector)

    # 4. Modulo Disparità e Profondità Stereo
    depth_estimator = StereoDepthEstimator(config.depth) if config.depth.enabled else None
    if depth_estimator:
        print("  ✓ Calcolo della disparità 3D (StereoSGBM) ABILITATO.")

    # 5. Server di Streaming MJPEG & REST API
    stream_server = MultiStreamServer(config.stream)
    stream_server.start()

    print("\n  🚀 PIPELINE VISIONE ATTIVA")
    print("  📡 Stream MJPEG disponibile sulla porta impostata (default 8080/8081)")
    print("======================================================================\n")

    fps = 0.0
    frame_count = 0
    t_fps_start = time.time()
    last_mqtt_pub = 0.0

    try:
        while True:
            t_loop_start = time.time()

            # Cattura frame stereo sincronizzato
            stereo_frame = camera.read()
            if stereo_frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            left_raw = stereo_frame.left
            right_raw = stereo_frame.right
            h_eye, w_eye = left_raw.shape[:2]

            # Inferenza YOLO
            left_ann, right_ann, left_objs, right_objs = detector.detect_stereo(left_raw, right_raw)

            # Stima disparità stereo & Griglia Spaziale 3D a 9 Zone
            color_disp = None
            grid_3d = {}
            depth_metrics = {
                "ground_hazard": False,
                "critical_front_cm": 999.0,
                "z_basso": 999.0,
                "z_centro": 999.0,
                "z_alto": 999.0
            }

            if depth_estimator:
                grid_3d, color_disp, depth_metrics = depth_estimator.compute_spatial_grid_3d(left_raw, right_raw)
                disp_map = depth_estimator.compute_disparity(left_raw, right_raw, apply_clahe=False)
                for obj in left_objs:
                    obj.estimated_distance_cm = depth_estimator.estimate_bbox_distance_cm(disp_map, obj.bbox)

            all_objects = left_objs + right_objs

            # Pubblicazione rilevamenti 3D su MQTT per mar-decision-engine
            now = time.time()
            if mqtt_client and mqtt_client.is_connected() and (now - last_mqtt_pub > 0.05):
                last_mqtt_pub = now
                detected_payload = {
                    "timestamp": now,
                    "count": len(all_objects),
                    "objects": [
                        {
                            "class": o.class_name,
                            "confidence": round(float(o.confidence), 2),
                            "bbox": [int(x) for x in o.bbox],
                            "distance_cm": round(float(o.estimated_distance_cm), 1) if o.estimated_distance_cm else None,
                            "sector": "CENTER" if (w_eye * 0.33 <= o.center_x <= w_eye * 0.66) else ("LEFT" if o.center_x < w_eye * 0.33 else "RIGHT")
                        }
                        for o in left_objs
                    ],
                    "spatial_grid_3d": grid_3d,
                    "ground_hazard": depth_metrics["ground_hazard"],
                    "z_basso_cm": depth_metrics["z_basso"],
                    "z_centro_cm": depth_metrics["z_centro"]
                }
                mqtt_client.publish("mar/vision/detections", json.dumps(detected_payload))

                # Se rilevato ostacolo basso a terra non visibile dal LiDAR, invia alert specifico
                if depth_metrics["ground_hazard"]:
                    mqtt_client.publish("mar/vision/ground_hazard", json.dumps({
                        "timestamp": now,
                        "z_basso_cm": depth_metrics["z_basso"],
                        "hazard": "TERRAIN_OR_LOW_OBSTACLE"
                    }))

            # Aggiornamento frame sul server MJPEG
            if config.stream.enabled:
                stream_server.update_frame("left", left_ann)
                stream_server.update_frame("right", right_ann)
                stream_server.update_frame("raw_left", left_raw)
                stream_server.update_frame("raw_right", right_raw)
                if color_disp is not None:
                    stream_server.update_frame("depth", color_disp)

                stream_server.update_status(
                    fps=round(fps, 1),
                    model=detector.model_name,
                    detections_count=len(all_objects),
                    active_command="STANDALONE_VISION",
                    action_type="VISION_ACTIVE",
                    ultrasonic_distance_cm=-1,
                    risk_level="INFO"
                )

            # Finestra GUI desktop se non headless
            if not config.headless:
                views = np.hstack([left_ann, right_ann])
                cv2.imshow("M.A.R. Vision Service Monitor", views)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            # Calcolo FPS
            if time.time() - t_fps_start >= 1.0:
                fps = frame_count / (time.time() - t_fps_start)
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
