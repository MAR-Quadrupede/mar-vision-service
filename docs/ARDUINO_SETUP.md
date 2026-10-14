# 🔌 Configurazione Hardware e Firmware Arduino — M.A.R.

Guida per il collegamento hardware tra la scheda **Arduino (Mega / Uno / Nano)**, il driver motori **L298N**, il sensore ad ultrasuoni **HC-SR04** e il canale seriale USB con il **Raspberry Pi 5**.

---

## 📌 Schema Cablaggio Pin

### 1. Driver Motori DC (L298N)

| Funzione Hardware | Pin Arduino | Descrizione |
| :--- | :--- | :--- |
| **IN1** (Motore Sinistro) | **D8** | Direzione avanti motore SX |
| **IN2** (Motore Sinistro) | **D7** | Direzione indietro motore SX |
| **ENA** (PWM Motore SX) | **D11** | Velocità motore SX (segnale analogico PWM 0-255) |
| **IN3** (Motore Destro) | **D9** | Direzione avanti motore DX |
| **IN4** (Motore Destro) | **D10** | Direzione indietro motore DX |
| **ENB** (PWM Motore DX) | **D12** | Velocità motore DX (segnale analogico PWM 0-255) |
| **GND** | **GND** | Massa comune con alimentazione batterie motori |

### 2. Sensore a Ultrasuoni HC-SR04

Il firmware supporta sia la modalità a **singolo pin** (TRIGGER ed ECHO ponticellati insieme) sia la modalità standard a **due pin**:

| Pin Sensore HC-SR04 | Connessione Arduino | Note |
| :--- | :--- | :--- |
| **VCC** | **+5V** | Alimentazione 5V da Arduino |
| **GND** | **GND** | Massa |
| **TRIG** | **D52** (o **D7**) | Impulso di attivazione ultrasuoni |
| **ECHO** | **D52** (o pin dedicato) | Lettura eco di ritorno |

---

## 💻 Caricamento del Firmware

1. Collega la scheda Arduino al computer via cavo USB.
2. Apri l'Arduino IDE.
3. Seleziona la tua scheda: `Tools -> Board -> Arduino Mega 2560` (o Uno/Nano).
4. Seleziona la porta seriale corretta (`/dev/cu.usbmodem*` su macOS o `/dev/ttyACM*` su Linux).
5. Apri lo sketch:
   [`firmware/mar_stereovision_controller/mar_stereovision_controller.ino`](file:///Users/gianluca/Desktop/m.a.r./stereovisione/firmware/mar_stereovision_controller/mar_stereovision_controller.ino).
6. Clicca su **Carica (Upload)**.

---

## 📡 Protocollo di Comunicazione Seriale

La comunicazione avviene a **9600** o **115200** baud delimitata da caratteri di ritorno a capo (`\n`).

### Comandi Downlink (RPi 5 ➔ Arduino)

| Comando M.A.R. | Comando Legacy | Azione |
| :--- | :---: | :--- |
| `CMD:STOP` | `S` | Arresto immediato e spegnimento uscite PWM motori |
| `CMD:FORWARD,<0-100>` | `F` | Marcia avanti alla velocità percentuale specificata |
| `CMD:BACKWARD,<0-100>` | `B` | Retromarcia alla velocità percentuale specificata |
| `CMD:TURN_L,<0-100>` | `L` | Rotazione sul posto verso sinistra |
| `CMD:TURN_R,<0-100>` | `R` | Rotazione sul posto verso destra |
| `CMD:STATUS` | `?` | Richiesta telemetria immediata |

### Telemetria Uplink (Arduino ➔ RPi 5)

Ad ogni ciclo di campionamento (10-15 Hz), Arduino trasmette sulla seriale:
```text
ULTRASONIC:34.2cm
TLM:STATUS=OK,SPEED=160,US=34.2
```

Se la distanza scende sotto la soglia di sicurezza critica ($< 15\text{ cm}$):
```text
EMERGENCY_STOP:14.2cm
```

---

## 🛡️ Meccanismi di Sicurezza Hardware Integrati

1. **Watchdog Heartbeat (1000 ms)**:
   Se per oltre 1 secondo Arduino non riceve alcun comando dalla seriale (crash di Python, cavo USB disconnesso o blocco del software), il firmware entra autonomamente in arresto e azzera le uscite PWM dei motori.
2. **Stop Fisico da Ultrasuoni**:
   Se il sensore rileva un ostacolo imminente ($< 15\text{ cm}$), Arduino blocca i motori ignorando comandi di avanzamento.
