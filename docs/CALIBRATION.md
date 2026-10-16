# Guida alla Calibrazione Stereoscopica e Mappa di Profondità

La stima della distanza 3D $(X, Y, Z)$ si basa sulla triangolazione epipolare tra i due sensori della stereocamera 3D Global Shutter.

---

## Principio Fisico della Visione Stereoscopica

Dati due sensori ottici paralleli separati da una distanza nota $B$ (baseline), un punto nello spazio $P$ viene proiettato nei piani immagine a coordinate orizzontali $x_L$ e $x_R$:

$$\text{Disparità } d = x_L - x_R$$

La distanza reale $Z$ dall'asse focale della camera al punto $P$ si ottiene tramite la formula:

$$Z = \frac{f \cdot B}{d}$$

Dove:
- $Z$: Distanza in centimetri o metri lungo l'asse ottico.
- $f$: Lunghezza focale effettiva della lente in pixel ($f = \frac{F_{\text{mm}} \cdot W_{\text{px}}}{\text{SensorWidth}_{\text{mm}}}$).
- $B$: Baseline (distanza tra i centri delle due lenti, tipicamente **$6.0\text{ cm}$** per la camera in uso).
- $d$: Disparità in pixel (spostamento orizzontale dell'oggetto tra occhio sinistro e destro).

---

##️ Parametri Ottimali per StereoSGBM

Il modulo [`stereo_depth.py`](file:///Users/gianluca/Desktop/m.a.r./stereovisione/camera/stereo_depth.py) adotta l'algoritmo **Semi-Global Block Matching**:

| Parametro | Valore Default | Significato e Regolazione |
| :--- | :--- | :--- |
| `minDisparity` | `0` | Disparità minima considerata (0 per oggetti a infinito). |
| `numDisparities` | `64` | Range di ricerca della disparità. Deve essere multiplo di 16. Aumentare (es. 96 o 128) per misurare oggetti molto vicini ($<30\text{ cm}$). |
| `blockSize` | `9` | Dimensione della finestra di matching (numero dispari: 5, 7, 9 o 11). Valori più grandi producono mappe più uniformi e riducono il rumore. |
| `P1` | $8 \cdot 3 \cdot \text{blockSize}^2$ | Penalità per piccole variazioni di disparità ($\Delta d = 1$). |
| `P2` | $32 \cdot 3 \cdot \text{blockSize}^2$ | Penalità per grandi discontinuità di disparità. |
| `disp12MaxDiff` | `1` | Tolleranza di consistenza tra matching sinistra-destra e destra-sinistra. |
| `uniquenessRatio` | `10` | Margine di confidenza percentuale minimo rispetto al secondo miglior match. |

---

## Stima della Distanza degli Oggetti YOLO

Invece di calcolare la distanza punto per punto su tutto il frame (computazionalmente oneroso per il processore ARM del Raspberry Pi 5):
1. Viene calcolata la disparità $d$.
2. Per ciascun bounding box rilevato da YOLO:
   - Viene estratto il **ROI centrale (50% centrale della box)**, escludendo i bordi per non includere lo sfondo.
   - Viene calcolata la **mediana statistica** dei valori di disparità validi ($d > 1$).
   - Viene applicata la formula $Z = \frac{f \cdot B}{d_{\text{mediana}}}$.
   - Il valore calcolato viene annotato sul frame e inviato all'engine decisionale.
