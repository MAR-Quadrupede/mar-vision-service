#!/usr/bin/env python3
"""
M.A.R. — High-Performance Multi-Stream MJPEG & Telemetry Server
Exposes processed & raw stereo streams, depth maps, and REST status API
for direct consumption by dashboard-mar-quadrupede (Next.js / WebSocket) and web browsers.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Any
import json
import threading
import time
import cv2
import numpy as np

try:
    from ..config import StreamConfig
except (ImportError, ValueError):
    from config import StreamConfig


class MultiStreamServer:
    """
    Multi-threaded MJPEG and HTTP API Server.
    Provides standard M.A.R. endpoints with full CORS headers.
    """

    def __init__(self, config: StreamConfig):
        self.config = config
        self._frames: Dict[str, Optional[bytes]] = {
            "left": None,
            "right": None,
            "raw_left": None,
            "raw_right": None,
            "depth": None
        }
        self._status_data: Dict[str, Any] = {
            "status": "ONLINE",
            "fps": 0.0,
            "model": "YOLOv8",
            "detections_count": 0,
            "active_command": "STOP",
            "ultrasonic_distance_cm": -1.0,
            "risk_level": "SAFE"
        }
        self._lock = threading.Lock()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def _make_handler(self):
        server_ref = self

        class StreamRequestHandler(BaseHTTPRequestHandler):
            def do_OPTIONS(self):
                """Handles CORS preflight requests."""
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "*")
                self.end_headers()

            def do_HEAD(self):
                """Handles HEAD requests for stream health checks."""
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
                if self.path in ("/", "/index.html"):
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                elif self.path == "/api/status":
                    self.send_header("Content-Type", "application/json")
                else:
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()

            def do_GET(self):
                # Health / status JSON API
                if self.path == "/api/status":
                    with server_ref._lock:
                        payload = json.dumps(server_ref._status_data).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                # Interactive HTML5 test dashboard
                if self.path in ("/", "/index.html"):
                    self._serve_test_page()
                    return

                # Endpoint mapping
                endpoint_map = {
                    "/stream_left.mjpg": "left",
                    "/stream_right.mjpg": "right",
                    "/stream_raw_left.mjpg": "raw_left",
                    "/stream_raw_right.mjpg": "raw_right",
                    "/stream_depth.mjpg": "depth"
                }

                if self.path not in endpoint_map:
                    self.send_error(404, f"Endpoint non trovato: {self.path}")
                    return

                stream_key = endpoint_map[self.path]
                self._stream_mjpeg(stream_key)

            def _stream_mjpeg(self, key: str):
                """Streams multipart/x-mixed-replace JPEG frames."""
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache, private, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                try:
                    while server_ref._running:
                        with server_ref._lock:
                            frame_bytes = server_ref._frames.get(key)

                        if frame_bytes is None:
                            time.sleep(0.03)
                            continue

                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                        time.sleep(0.033)  # ~30 FPS throttling
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _serve_test_page(self):
                """Serves modern dark-themed HTML UI showing all video streams."""
                html = """<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>M.A.R. — Stereovisione & Edge AI Monitor</title>
    <style>
        :root {
            --bg-base: #0f1117;
            --bg-card: #181b24;
            --border: #292d3e;
            --accent: #3b82f6;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-base);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }
        .header h1 { margin: 0; font-size: 1.5rem; display: flex; align-items: center; gap: 8px; }
        .badge {
            background: #1e3a8a;
            color: #bfdbfe;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        }
        .card-header {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 600;
            font-size: 0.95rem;
        }
        .card-img-container {
            position: relative;
            background: #000;
            min-height: 280px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .card-img-container img {
            width: 100%;
            height: auto;
            display: block;
        }
        .endpoints-box {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            margin-top: 24px;
        }
        .endpoints-box h3 { margin-top: 0; color: var(--accent); }
        .endpoint-item {
            font-family: monospace;
            background: #11141c;
            padding: 8px 12px;
            border-radius: 6px;
            margin: 6px 0;
            display: inline-block;
            margin-right: 12px;
        }
        a { color: var(--accent); text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🐾 M.A.R. — Stereovisione 3D & AI Edge</h1>
        <span class="badge">SERVER ATTIVO (Porta 8443)</span>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-header">
                <span>📹 Occhio Sinistro (LEFT + YOLO)</span>
                <a href="/stream_left.mjpg" target="_blank">Apri Flusso</a>
            </div>
            <div class="card-img-container">
                <img src="/stream_left.mjpg" alt="Left Stream">
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <span>📹 Occhio Destro (RIGHT + YOLO)</span>
                <a href="/stream_right.mjpg" target="_blank">Apri Flusso</a>
            </div>
            <div class="card-img-container">
                <img src="/stream_right.mjpg" alt="Right Stream">
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <span>📷 Stream RAW Sinistro</span>
                <a href="/stream_raw_left.mjpg" target="_blank">Apri Flusso</a>
            </div>
            <div class="card-img-container">
                <img src="/stream_raw_left.mjpg" alt="Raw Left">
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <span>📷 Stream RAW Destro</span>
                <a href="/stream_raw_right.mjpg" target="_blank">Apri Flusso</a>
            </div>
            <div class="card-img-container">
                <img src="/stream_raw_right.mjpg" alt="Raw Right">
            </div>
        </div>
    </div>

    <div class="endpoints-box">
        <h3>🔗 Endpoints per Dashboard M.A.R. (Next.js)</h3>
        <div class="endpoint-item">/stream_left.mjpg</div>
        <div class="endpoint-item">/stream_right.mjpg</div>
        <div class="endpoint-item">/stream_raw_left.mjpg</div>
        <div class="endpoint-item">/stream_raw_right.mjpg</div>
        <div class="endpoint-item"><a href="/api/status" target="_blank">/api/status (JSON)</a></div>
    </div>
</body>
</html>"""
                payload = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format, *args):
                # Suppress noisy HTTP GET logging during live streaming
                return

        return StreamRequestHandler

    def start(self) -> bool:
        """Starts HTTP server in a background daemon thread."""
        if not self.config.enabled:
            print("[SERVER] Server di streaming disabilitato da configurazione.")
            return False

        if self._running:
            return True

        try:
            self._httpd = ThreadingHTTPServer(
                (self.config.host, self.config.port),
                self._make_handler()
            )
            self._httpd.daemon_threads = True
            self._running = True
            self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
            self._thread.start()

            print(f"\n[SERVER] Server MJPEG & REST avviato su http://{self.config.host}:{self.config.port}")
            print(f"  • Dashboard SX + YOLO: http://localhost:{self.config.port}/stream_left.mjpg")
            print(f"  • Dashboard DX + YOLO: http://localhost:{self.config.port}/stream_right.mjpg")
            print(f"  • Web Monitor:         http://localhost:{self.config.port}/")
            return True
        except Exception as e:
            print(f"[SERVER] ✗ Errore avvio server su {self.config.host}:{self.config.port}: {e}")
            self._running = False
            return False

    def update_frame(self, key: str, frame: np.ndarray):
        """Encodes frame as JPEG and stores in memory buffer."""
        if not self._running or frame is None:
            return

        ok, encoded = cv2.imencode(
            '.jpg',
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality]
        )
        if not ok:
            return

        with self._lock:
            self._frames[key] = encoded.tobytes()

    def update_status(self, **kwargs):
        """Updates internal telemetry dictionary."""
        with self._lock:
            self._status_data.update(kwargs)

    def stop(self):
        """Stops the streaming server gracefully."""
        if not self._running:
            return

        self._running = False
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        print("[SERVER] Server streaming arrestato.")
