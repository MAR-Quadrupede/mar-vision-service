# Microservizio: `mar-vision-service`

**3D Stereo Perception, Disparity Depth & Edge AI Inference**  
*Parte dell'architettura a microservizi M.A.R. (Multi-legged Autonomous Robot)*

---

## Ruolo nell'Architettura a Microservizi

`mar-vision-service` è il microservizio a maggiore intensità computazionale del robot. Gira su Raspberry Pi 5 ed è responsabile della **comprensione visiva e spaziale 3D**:
1. **Acquisizione Stereo Global Shutter Sincronizzata**:
   - Acquisizione simultanea dei frame sinistro e destro via USB 3.0 / USB 2.0 (telecamera ELP).
   - Nessuna distorsione rolling-shutter durante i movimenti rapidi o le andature a trotto.
2. **Stima di Profondità 3D (Stereo Depth) & Calibrazione Hardware**:
   - Calibrazione intrinseca ed estrinseca reale caricata dal file `parametri_calibrazione_stereo.npz`.
   - Pre-elaborazione del contrasto con CLAHE per adattarsi a condizioni di luce dinamiche.
   - Calcolo della mappa di disparità tramite algoritmo `StereoSGBM` ed edge-preserving smoothing con filtro WLS.
3. **Mappatura Spaziale 3D a 9 Zone & Rilevamento Dislivelli**:
   - Segmentazione metrica della scena in 9 zone (`Alto`, `Centro`, `Basso` x `SX`, `Centro`, `DX`) con filtraggio statistico dei percentili 15°-85°.
   - Rilevamento di ostacoli bassi a terra, scalini o buche ($Z_{\text{basso}} < 40\text{ cm}$) invisibili al LiDAR 2D orizzontale.
4. **Edge AI Object Detection (YOLO)**:
   - Riconoscimento semantico in tempo reale di ostacoli (*persone, animali, scatole, ostacoli rigidi*).
   - Supporto modelli `yolov8n`, `yolo11n` e formati ONNX/NCNN per massimizzare il throughput sui 4 core Cortex-A76 del Raspberry Pi 5.
5. **Streaming Video Multi-feed**:
   - Server HTTP MJPEG per fornire flussi video diretti alla `mar-dashboard` (`/stream/left`, `/stream/right`, `/stream/depth`).
6. **Pubblicazione Rilevamenti 3D su MQTT**:
   - Pubblica le classi riconosciute, le coordinate 3D metriche e la griglia spaziale sul topic `mar/vision/detections`.
   - Notifiche immediate di pericolo a terra sul topic `mar/vision/ground_hazard`.

---

## Interfacce, Endpoint HTTP e Topic MQTT

### Endpoint HTTP Streaming
* `http://<IP_RPI>:8080/stream/left`: Stream occhio sinistro con bounding box
* `http://<IP_RPI>:8080/stream/right`: Stream occhio destro annotato
* `http://<IP_RPI>:8080/stream/depth`: Mappa di disparità a falsi colori con overlay griglia 3D
* `http://<IP_RPI>:8080/api/status`: JSON con telemetria FPS, modello attivo, conteggio ostacoli

### Topic MQTT
* **`mar/vision/detections`** (Pub):
  ```json
  {
    "timestamp": 1728940000.12,
    "count": 1,
    "objects": [
      {
        "class": "person",
        "confidence": 0.88,
        "bbox": [120, 80, 240, 320],
        "distance_cm": 115.4,
        "sector": "CENTER"
      }
    ],
    "spatial_grid_3d": {
      "Centro_Centro": [0.0, 0.0, 115.4],
      "Basso_Centro": [0.0, -15.2, 85.0]
    },
    "ground_hazard": false,
    "z_basso_cm": 85.0,
    "z_centro_cm": 115.4
  }
  ```
* **`mar/vision/ground_hazard`** (Pub):
  ```json
  {
    "timestamp": 1728940000.15,
    "z_basso_cm": 35.0,
    "hazard": "TERRAIN_OR_LOW_OBSTACLE"
  }
  ```

---

## Struttura del Repository

```text
mar-vision-service/
├── README.md                          # Questo documento
├── requirements.txt                   # Dipendenze Python generiche
├── requirements-raspberrypi.txt       # Dipendenze ottimizzate per Raspberry Pi 5
├── config.py                          # Configurazione CLI e parametri stereo/AI
├── main.py                            # Demone del microservizio
├── camera/                            # Driver acquisizione stereo, calibrazione npz e griglia 3D
├── detector/                          # Wrapper ultralytics per inferenza YOLO
├── models/                            # Pesi neurali (yolov8n.pt, yolo11n.pt, ecc.)
├── server/                            # Server HTTP multi-stream MJPEG
├── scripts/                           # Script di benchmark modelli e utility
└── docs/                              # Documentazione calibrazione ed endpoint
```

---

## Avvio del Microservizio

### 1. Installazione Dipendenze
```bash
pip install -r requirements.txt
```

### 2. Esecuzione
```bash
# Avvio standard con profondità e streaming abilitati
python main.py --depth

# Modalità Headless su Raspberry Pi 5 (senza display grafico locale)
python main.py --headless --depth --model models/yolov8n.pt
```
