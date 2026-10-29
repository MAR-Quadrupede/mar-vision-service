#!/usr/bin/env python3
"""
M.A.R. — High-Performance Multi-Stream MJPEG & Telemetry Server
Exposes processed & raw stereo streams, depth maps, 2D top-down navigation map,
interactive camera calibration endpoints, and telemetry for dashboard & browsers.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Any, Callable
from urllib.parse import urlparse, parse_qs
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

    def __init__(self, config: StreamConfig, command_callback: Optional[Callable[[str, Any], None]] = None):
        self.config = config
        self.command_callback = command_callback
        self._frames: Dict[str, Optional[bytes]] = {
            "left": None,
            "right": None,
            "raw_left": None,
            "raw_right": None,
            "depth": None,
            "composite": None,
            "map": None
        }
        self._status_data: Dict[str, Any] = {
            "status": "ONLINE",
            "fps": 0.0,
            "fps_camera": 0.0,
            "fps_depth": 0.0,
            "modo": 0,
            "auto_switch": True,
            "offset_x": 0,
            "offset_y": 0,
            "decisione_guida": "STOP",
            "motivo_guida": "Inizializzazione",
            "ostacolo_centrale_cm": 999.0,
            "ostacolo_vicino_cm": 999.0,
            "ground_hazard": False,
            "allarme_lenti": False,
            "allarme_motivo": "OK",
            "model": "YOLOv8",
            "detections_count": 0
        }
        self._active_stream_clients = 0
        self._lock = threading.Lock()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def _make_handler(self):
        server_ref = self

        class StreamRequestHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

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
                elif self.path in ("/api/status", "/stato"):
                    self.send_header("Content-Type", "application/json")
                else:
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path

                # Status APIs
                if path in ("/api/status", "/stato"):
                    with server_ref._lock:
                        payload = json.dumps(server_ref._status_data).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                # Interactive Commands API (/cmd?azione=... or ?modo=... etc)
                if path == "/cmd":
                    params = parse_qs(parsed.query)
                    if server_ref.command_callback:
                        for key, val_list in params.items():
                            val = val_list[0] if val_list else ""
                            server_ref.command_callback(key, val)

                    resp = json.dumps({"status": "OK", "received": params}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp)
                    return

                # Interactive Web UI
                if path in ("/", "/index.html"):
                    self._serve_test_page()
                    return

                # Single JPEG snapshots
                snapshot_map = {
                    "/snapshot.jpg": "composite",
                    "/snapshot_map.jpg": "map",
                    "/snapshot_left.jpg": "left",
                    "/snapshot_right.jpg": "right",
                    "/snapshot_depth.jpg": "depth"
                }
                if path in snapshot_map:
                    self._serve_snapshot(snapshot_map[path])
                    return

                # MJPEG video streams
                endpoint_map = {
                    "/stream": "composite",
                    "/stream_map.mjpg": "map",
                    "/stream_left.mjpg": "left",
                    "/stream_right.mjpg": "right",
                    "/stream_raw_left.mjpg": "raw_left",
                    "/stream_raw_right.mjpg": "raw_right",
                    "/stream_depth.mjpg": "depth"
                }

                if path in endpoint_map:
                    self._stream_mjpeg(endpoint_map[path])
                    return

                self.send_error(404, f"Endpoint non trovato: {path}")

            def _serve_snapshot(self, key: str):
                """Serves single instant JPEG snapshot with no-cache headers."""
                with server_ref._lock:
                    frame_bytes = server_ref._frames.get(key)
                    if frame_bytes is None and key == "composite":
                        frame_bytes = server_ref._frames.get("left")

                if frame_bytes is None:
                    self.send_error(503, "Frame non ancora disponibile")
                    return

                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame_bytes)))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(frame_bytes)

            def _stream_mjpeg(self, key: str):
                """Streams multipart/x-mixed-replace JPEG frames."""
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache, private, no-store, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                with server_ref._lock:
                    server_ref._active_stream_clients += 1

                try:
                    while server_ref._running:
                        with server_ref._lock:
                            frame_bytes = server_ref._frames.get(key)
                            if frame_bytes is None and key == "composite":
                                frame_bytes = server_ref._frames.get("left")

                        if frame_bytes is None:
                            time.sleep(0.02)
                            continue

                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                        time.sleep(0.033)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    with server_ref._lock:
                        server_ref._active_stream_clients = max(0, server_ref._active_stream_clients - 1)

            def _serve_test_page(self):
                """Serves modern dark-themed interactive UI matching the 8080 design."""
                html = """<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>M.A.R. — Stereovisione 3D & AI Edge Monitor</title>
    <style>
        :root {
            --bg-base: #0f1117;
            --bg-card: #181b24;
            --border: #292d3e;
            --accent: #3b82f6;
            --accent-green: #10b981;
            --accent-yellow: #f59e0b;
            --accent-red: #ef4444;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-base);
            color: var(--text-main);
            margin: 0;
            padding: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 14px;
            margin-bottom: 20px;
        }
        .header h1 { margin: 0; font-size: 1.4rem; display: flex; align-items: center; gap: 10px; }
        .pill {
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .pill-go { background: #064e3b; color: #6ee7b7; border: 1px solid #059669; }
        .pill-warn { background: #78350f; color: #fde68a; border: 1px solid #d97706; }
        .pill-stop { background: #7f1d1d; color: #fca5a5; border: 1px solid #dc2626; }

        .main-layout {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }
        @media (max-width: 1024px) {
            .main-layout { grid-template-columns: 1fr; }
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 16px rgba(0,0,0,0.5);
            margin-bottom: 20px;
        }
        .card-header {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 600;
        }
        .video-box {
            position: relative;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 360px;
        }
        .video-box img {
            width: 100%;
            height: auto;
            display: block;
        }

        .controls-card {
            padding: 16px;
        }
        .btn-group {
            display: flex;
            gap: 8px;
            margin-bottom: 14px;
            flex-wrap: wrap;
        }
        button {
            background: #252a38;
            color: var(--text-main);
            border: 1px solid var(--border);
            padding: 8px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.85rem;
            transition: all 0.15s ease;
        }
        button:hover {
            background: var(--accent);
            color: #fff;
            border-color: var(--accent);
        }
        button.active {
            background: var(--accent);
            color: #fff;
            border-color: var(--accent);
        }
        button.active-auto {
            background: var(--accent-green);
            color: #fff;
            border-color: var(--accent-green);
        }

        .stats-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            margin-top: 10px;
        }
        .stats-table td {
            padding: 8px 10px;
            border-bottom: 1px solid #202433;
        }
        .stats-table tr:last-child td { border-bottom: none; }
        .label-col { color: var(--text-muted); width: 45%; }
        .val-col { font-weight: 600; font-family: monospace; }

        .alarm-banner {
            background: #450a0a;
            color: #fca5a5;
            border: 1px solid #ef4444;
            padding: 10px 14px;
            border-radius: 8px;
            margin-bottom: 14px;
            display: none;
            font-weight: 600;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🐾 M.A.R. — Stereovisione 3D & Guida Autonoma</h1>
        <div id="decision-pill" class="pill pill-stop">STOP</div>
    </div>

    <div id="alarm-box" class="alarm-banner">⚠️ ALLARME ATTIVO: Lenti o Sensore non affidabili!</div>

    <div class="main-layout">
        <!-- Colonna Principale: Video Live -->
        <div>
            <div class="card">
                <div class="card-header">
                    <span>📹 Flusso Live Principale (<span id="mode-label">Modo 0: Stereo Side-by-Side</span>)</span>
                    <a href="/stream" target="_blank" style="color:var(--accent); text-decoration:none;">Fullscreen</a>
                </div>
                <div class="video-box">
                    <img id="main-stream" src="/stream" alt="Video Stream">
                </div>
            </div>

            <!-- Controlli Interattivi -->
            <div class="card controls-card">
                <div style="font-weight:600; margin-bottom:10px;">Modalità Visualizzazione & Disparità</div>
                <div class="btn-group">
                    <button id="btn-m0" onclick="sendCmd('modo=0')">Modo 0: Stereo SBS</button>
                    <button id="btn-m1" onclick="sendCmd('modo=1')">Modo 1: Disparità Raw</button>
                    <button id="btn-m2" onclick="sendCmd('modo=2')">Modo 2: Disparità Jet</button>
                    <button id="btn-m3" onclick="sendCmd('modo=3')">Modo 3: Ostacolo Auto-Focus</button>
                    <button id="btn-auto" onclick="sendCmd('auto=toggle')">Auto 0/3: ON</button>
                </div>

                <div style="font-weight:600; margin-top:14px; margin-bottom:10px;">Allineamento Asse Ottico (Calibrazione Hardware)</div>
                <div class="btn-group">
                    <button onclick="sendCmd('dx=-1')">DX -1</button>
                    <button onclick="sendCmd('dx=1')">DX +1</button>
                    <button onclick="sendCmd('dy=-1')">DY -1</button>
                    <button onclick="sendCmd('dy=1')">DY +1</button>
                    <button onclick="sendCmd('salva=1')" style="background:#065f46; border-color:#059669;">Salva Offset</button>
                    <button onclick="sendCmd('reset=1')" style="background:#831843; border-color:#be185d;">Reset 0,0</button>
                </div>
            </div>
        </div>

        <!-- Colonna Laterale: Mappa Top-Down 2D e Telemetria -->
        <div>
            <div class="card">
                <div class="card-header">
                    <span>🗺️ Mappa Top-Down Navigazione (2D)</span>
                    <a href="/stream_map.mjpg" target="_blank" style="color:var(--accent); text-decoration:none;">Apri</a>
                </div>
                <div class="video-box" style="min-height:300px;">
                    <img src="/stream_map.mjpg" alt="Mappa Navigazione">
                </div>
            </div>

            <div class="card" style="padding:16px;">
                <div style="font-weight:600; border-bottom:1px solid var(--border); padding-bottom:8px; margin-bottom:8px;">
                    📊 Telemetria in Tempo Reale
                </div>
                <table class="stats-table">
                    <tr><td class="label-col">FPS Telecamera:</td><td class="val-col" id="val-fps-cam">0.0</td></tr>
                    <tr><td class="label-col">FPS Pipeline 3D:</td><td class="val-col" id="val-fps-depth">0.0</td></tr>
                    <tr><td class="label-col">Decisione Guida:</td><td class="val-col" id="val-decision">STOP</td></tr>
                    <tr><td class="label-col">Motivo Guida:</td><td class="val-col" id="val-reason">OK</td></tr>
                    <tr><td class="label-col">Ostacolo Centro:</td><td class="val-col" id="val-obs-center">--- cm</td></tr>
                    <tr><td class="label-col">Ostacolo Più Vicino:</td><td class="val-col" id="val-obs-near">--- cm</td></tr>
                    <tr><td class="label-col">Rischio Terreno Basso:</td><td class="val-col" id="val-ground">NO</td></tr>
                    <tr><td class="label-col">Offset DX / DY:</td><td class="val-col" id="val-offsets">0 / 0</td></tr>
                    <tr><td class="label-col">Integrità Lenti:</td><td class="val-col" id="val-lenses">OK</td></tr>
                </table>
            </div>
        </div>
    </div>

    <script>
        function sendCmd(param) {
            fetch('/cmd?' + param)
                .then(r => r.json())
                .then(() => aggiornaStato());
        }

        function aggiornaStato() {
            fetch('/stato')
                .then(r => r.json())
                .then(d => {
                    document.getElementById('val-fps-cam').innerText = (d.fps_camera || d.fps || 0).toFixed(1);
                    document.getElementById('val-fps-depth').innerText = (d.fps_depth || 0).toFixed(1);
                    document.getElementById('val-decision').innerText = d.decisione_guida || 'STOP';
                    document.getElementById('val-reason').innerText = d.motivo_guida || 'OK';

                    const c_cm = (d.ostacolo_centrale_cm < 900) ? d.ostacolo_centrale_cm.toFixed(1) + ' cm' : 'Libero';
                    const n_cm = (d.ostacolo_vicino_cm < 900) ? d.ostacolo_vicino_cm.toFixed(1) + ' cm' : 'Nessuno';
                    document.getElementById('val-obs-center').innerText = c_cm;
                    document.getElementById('val-obs-near').innerText = n_cm;
                    document.getElementById('val-ground').innerText = d.ground_hazard ? '⚠️ ALLERTA' : 'NO';
                    document.getElementById('val-offsets').innerText = (d.offset_x || 0) + ' / ' + (d.offset_y || 0);
                    document.getElementById('val-lenses').innerText = d.allarme_lenti ? ('⚠️ ' + d.allarme_motivo) : 'OK';

                    const pill = document.getElementById('decision-pill');
                    pill.innerText = d.decisione_guida || 'STOP';
                    pill.className = 'pill ' + (d.decisione_guida === 'AVANTI' ? 'pill-go' : (d.decisione_guida === 'FRENA' ? 'pill-warn' : 'pill-stop'));

                    const alarmBox = document.getElementById('alarm-box');
                    if (d.allarme_lenti) {
                        alarmBox.innerText = '⚠️ ALLARME SICUREZZA: ' + d.allarme_motivo;
                        alarmBox.style.display = 'block';
                    } else {
                        alarmBox.style.display = 'none';
                    }

                    // Mode labels
                    const modeNames = ["Modo 0: Stereo SBS", "Modo 1: Disparità Raw", "Modo 2: Disparità Jet", "Modo 3: Ostacolo Auto-Focus"];
                    document.getElementById('mode-label').innerText = modeNames[d.modo] || ('Modo ' + d.modo);

                    // Mode buttons active state
                    for (let i = 0; i <= 3; i++) {
                        const btn = document.getElementById('btn-m' + i);
                        if (btn) btn.className = (d.modo === i) ? 'active' : '';
                    }
                    const btnAuto = document.getElementById('btn-auto');
                    if (btnAuto) {
                        btnAuto.innerText = 'Auto 0/3: ' + (d.auto_switch ? 'ON' : 'OFF');
                        btnAuto.className = d.auto_switch ? 'active-auto' : '';
                    }
                })
                .catch(() => {});
        }

        setInterval(aggiornaStato, 500);
        aggiornaStato();
    </script>
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
            print(f"  • Console Web & Guida: http://localhost:{self.config.port}/")
            print(f"  • Stream Composito:    http://localhost:{self.config.port}/stream")
            print(f"  • Mappa 2D Top-Down:   http://localhost:{self.config.port}/stream_map.mjpg")
            print(f"  • Telemetria JSON:     http://localhost:{self.config.port}/stato")
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

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        print("[SERVER] Server streaming arrestato.")
