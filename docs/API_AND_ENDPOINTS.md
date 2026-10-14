# 🌐 Specifiche API & Streaming Endpoints — M.A.R.

Il server di streaming integrato espone i flussi video MJPEG e gli endpoint REST utilizzati da **`dashboard-mar-quadrupede`** (Next.js) e da qualsiasi client HTTP.

Porta predefinita: **`8443`** (configurabile con `--port`).

---

## 📹 Flussi Video MJPEG (`multipart/x-mixed-replace`)

Tutti i flussi video sono conformi allo standard MJPEG e includono gli header `Cache-Control: no-cache` e `Access-Control-Allow-Origin: *`.

| Endpoint | Descrizione | Utilizzo nella Dashboard |
| :--- | :--- | :--- |
| `GET /stream_left.mjpg` | Stream occhio sinistro con annotazioni YOLO, bounding box e zona di pericolo | `VideoGrid.tsx` (Camera Sinistra) |
| `GET /stream_right.mjpg` | Stream occhio destro con annotazioni YOLO, bounding box e zona di pericolo | `VideoGrid.tsx` (Camera Destra) |
| `GET /stream_raw_left.mjpg` | Stream occhio sinistro RAW ad alta velocità (senza bounding box o elaborazioni) | Diagnostica e bassa latenza |
| `GET /stream_raw_right.mjpg` | Stream occhio destro RAW ad alta velocità (senza bounding box o elaborazioni) | Diagnostica e calibrazione |
| `GET /stream_depth.mjpg` | Mappa di disparità a falsi colori (colormapped Turbo/Jet) generata da StereoSGBM | Visualizzazione tridimensionale |

---

## 📊 REST API di Telemetria

### `GET /api/status`

Restituisce un payload JSON contenente la telemetria in tempo reale del sottosistema visivo e dei sensori.

#### Esempio di Risposta JSON:

```json
{
  "status": "ONLINE",
  "fps": 28.4,
  "model": "YOLOV8N",
  "detections_count": 2,
  "active_command": "CMD:FORWARD,50",
  "action_type": "FORWARD",
  "ultrasonic_distance_cm": 118.5,
  "risk_level": "SAFE"
}
```

#### Campi del Payload:

- **`status`** (`string`): Stato del server (`ONLINE`, `DISCONNECTED`).
- **`fps`** (`float`): Frame rate effettivo del ciclo di acquisizione ed inferenza.
- **`model`** (`string`): Modello YOLO attualmente caricato in memoria (es. `YOLOV8N`, `YOLO11M`).
- **`detections_count`** (`integer`): Numero totale di oggetti attualmente tracciati nei frame.
- **`active_command`** (`string`): Ultimo comando cinematica inviato ad Arduino (es. `CMD:FORWARD,50`, `CMD:STOP`).
- **`action_type`** (`string`): Categoria azione (`FORWARD`, `STOP`, `TURN_LEFT`, `TURN_RIGHT`, `SLOW`).
- **`ultrasonic_distance_cm`** (`float`): Distanza rilevata dal sensore ultrasuoni HC-SR04 (-1 se non disponibile).
- **`risk_level`** (`string`): Livello di allerta computato (`SAFE`, `CAUTION`, `WARNING`, `DANGER`, `EMERGENCY`).

---

## 🖥️ Interfaccia Web di Monitoraggio

### `GET /`

Accedendo con qualsiasi browser a `http://localhost:8443/` viene servita una dashboard web nativa in HTML5 / CSS3 scuro, che mostra in una griglia responsive:
- Il flusso live annotato sinistro e destro
- I due flussi raw di supporto
- La lista degli endpoint per facile copia/incolla
- Stato del server e connettività
