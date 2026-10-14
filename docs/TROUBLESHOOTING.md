# 🛠️ Risoluzione Problemi (Troubleshooting) — M.A.R.

Guida alla risoluzione dei problemi più comuni relativi a fotocamere stereo, permessi hardware, modelli YOLO e prestazioni.

---

## 📷 1. Problemi con la Stereocamera

### Sintomo: La telecamera non viene rilevata o fallisce l'apertura

#### Su macOS:
1. **Permessi di Sicurezza e Privacy**:
   - Vai in *Impostazioni di Sistema* ➔ *Privacy e Sicurezza* ➔ *Fotocamera*.
   - Verifica che l'applicazione Terminale (o l'IDE in uso) sia abilitata.
2. **Conflitto di Connessione**:
   - Assicurati che altre applicazioni (FaceTime, Zoom, OBS, browser) non stiano occupando il dispositivo video.
3. **Verifica Riconoscimento USB**:
   ```bash
   system_profiler SPCameraDataType
   ```

#### Su Linux / Raspberry Pi:
1. **Verifica Nodi Video**:
   ```bash
   ls -la /dev/video*
   v4l2-ctl --list-devices
   ```
2. **Permessi Gruppo Video**:
   ```bash
   sudo usermod -a -G video $USER
   ```
3. **Riavvia demone udev**:
   ```bash
   sudo udevadm control --reload-rules && sudo udevadm trigger
   ```

> [!NOTE]
> Se nessuna telecamera fisica è collegata, il driver attiva automaticamente la **modalità simulatore sintetico (mock)**: puoi così testare il server web, la grafica e le API anche senza hardware collegato!

---

## 🔌 2. Problemi Seriale con Arduino

### Sintomo: Errore "Permission denied" su `/dev/ttyACM0` o `/dev/ttyUSB0`

Aggiungi il tuo utente al gruppo `dialout`:
```bash
sudo usermod -a -G dialout $USER
```
Effettua il log out e rientra nella sessione affinché i permessi diventino attivi.

### Sintomo: Porta seriale non trovata automaticamente
Specifica esplicitamente la porta con il flag `--serial-port`:
```bash
python3 main.py --serial-port /dev/cu.usbmodem14101
```

Oppure disabilita temporaneamente la seriale se esegui solo test visivi:
```bash
python3 main.py --no-arduino
```

---

## ⚡ 3. Ottimizzazione Prestazioni e Frame Rate (FPS)

### Sintomo: FPS bassi su Raspberry Pi 5 o 4
1. **Scelta del Modello YOLO**:
   - Usa **`yolov8n.pt`** (Nano): è il modello raccomandato con l'ottimo compromesso velocità/accuratezza per dispositivi embedded ARM (~15-30 FPS).
   - Evita modelli `x` (extra-large come `yolo26x.pt` o `yolov8x.pt`) in tempo reale su CPU, in quanto pensati per GPU workstation dedicate.
2. **Riduci Risoluzione di Cattura**:
   ```bash
   python3 main.py --width 1280 --height 400
   ```
3. **Esegui in Modalità Headless**:
   L'apertura di finestre grafiche X11/Wayland con `cv2.imshow` consuma cicli CPU. Sui robot in produzione avvia sempre in headless:
   ```bash
   python3 main.py --headless
   ```
   I flussi visivi rimarranno comunque visibili a pieno framerate via rete su `http://<IP_ROBOT>:8443/`.

---

## 🌐 4. Connessione con la Dashboard Next.js

### Sintomo: I video non compaiono in `dashboard-mar-quadrupede`
1. Assicurati che lo script di stereovisione sia in esecuzione sulla porta `8443`.
2. Apri `http://localhost:8443/` nel browser: dovresti vedere la griglia di monitoraggio con i 4 stream.
3. Se accedi da un altro PC/tablet via rete locale o Tailscale:
   - Apri `dashboard-mar-quadrupede/components/VideoGrid.tsx` o verifica che punti all'IP corretto del robot.
   - Tutti gli endpoint includono `Access-Control-Allow-Origin: *` per evitare blocchi CORS.
