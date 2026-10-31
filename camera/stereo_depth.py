#!/usr/bin/env python3
"""
M.A.R. — Advanced Stereo Depth, Obstacle Isolation, Auto-Focus & Safety Module
Incorporates fast fixed-point rectification, epipolar vertical alignment,
density & blob filtering, dynamic obstacle auto-focus, safety lenses health checks,
top-down navigation mapping, and corridor collision driving decisions.
"""

from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import os
import json
import time
import platform
import cv2
import numpy as np

try:
    from ..config import DepthConfig
    from ..safety_bands import (
        RiskDecisionStabilizer,
        DIST_CRITICAL_CM,
        DIST_HIGH_CM,
        DIST_MODERATE_CM,
    )
except (ImportError, ValueError):
    from config import DepthConfig
    from safety_bands import (
        RiskDecisionStabilizer,
        DIST_CRITICAL_CM,
        DIST_HIGH_CM,
        DIST_MODERATE_CM,
    )


# Parametri geometrici e soglie rover M.A.R.
# Su ARM (RPi) allineati a stereovisione_robot.4py.py: SGBM_DIV=4 + RECT_DIV=2
# (~10× più leggero di DIV=2 a piena rettificazione) così la UI non resta a 0.5 FPS.
IS_ARM = any(tag in platform.machine().lower() for tag in ("arm", "aarch64"))

FRAME_W_ATTESO, FRAME_H_ATTESO = 2560, 800
SGBM_DIV = int(os.getenv("MAR_SGBM_DIV", "4" if IS_ARM else "2"))
RECT_DIV = int(os.getenv("MAR_RECT_DIV", "2" if IS_ARM else "1"))
NUM_DISPARITIES = int(os.getenv("MAR_NUM_DISPARITIES", "96" if SGBM_DIV >= 4 else "192"))
SGBM_BLOCK = 5 if SGBM_DIV >= 4 else 7
VS = 1.0 / RECT_DIV
STEP_SUB = max(1, 4 // max(1, SGBM_DIV))
# Soglia gradiente Sobel anti-falsi ostacoli su pareti lisce e pavimenti (1.5 per non scartare pareti indoor reali)
TEX_MIN = float(os.getenv("MAR_TEX_MIN", "1.5"))

# Limiti corridoio rover (camera montata sottosopra)
ALTEZZA_MIN_ROBOT = -25.0  # cm
ALTEZZA_MAX_ROBOT = 25.0   # cm
MAX_DIST_CM = 300.0        # cm
SOGLIA_DISTANZA_CM = 200.0 # 2.0m soglia auto-switch Modo 0 -> Modo 3
ISTERESI_CM = 15.0
DENSITA_MIN = 0.5
MIN_AREA_BLOB = 60         # Componente connessa minima (anti-rumore)

# Mappa navigazione top-down
MAP_SIZE = 450
RISOLUZIONE_CM = 1.0

# Sicurezza lenti & watchdog
SIC_BUIO_MEDIA = 15.0
SIC_SATURO_MEDIA = 245.0
SIC_UNIFORME_STD = 5.0
SIC_UNIFORME_GRAD = 3.0
SIC_ASIMMETRIA_LUMINOSITA = 55.0
SIC_ASIMMETRIA_TEXTURE = 4.5
SIC_FRAME_CONGELATI = 45
SIC_MIN_PUNTI_VALIDI = 0.005
SIC_FRAME_ATTIVAZIONE = 3
SIC_FRAME_RIENTRO = 6
SIC_TIMEOUT_FRAME_S = 1.5


def mappe_veloci(mx, my):
    """Converte le mappe float in formato fixed-point CV_16SC2 per remapping ad altissime prestazioni."""
    return cv2.convertMaps(mx, my, cv2.CV_16SC2)


def filtra_densita(mask: np.ndarray, k: int = 5, dens_min: float = DENSITA_MIN) -> np.ndarray:
    """Elimina i punti isolati di rumore mantenendo solo aggregati densi."""
    dens = cv2.blur(mask.astype(np.float32), (k, k))
    return mask & (dens > dens_min)


def isola_blob_compatto(mask: np.ndarray) -> Optional[np.ndarray]:
    """Isola la componente connessa più grande; None se non presente o inferiore all'area minima."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    kmax = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[kmax, cv2.CC_STAT_AREA] < MIN_AREA_BLOB:
        return None
    return lab == kmax


def stima_dy_epipolare(grigio_L: np.ndarray, grigio_R: np.ndarray) -> Optional[float]:
    """Stima lo sfasamento verticale tra L e R rettificate tramite feature matching ORB."""
    clahe = cv2.createCLAHE(3.0, (8, 8))
    orb = cv2.ORB_create(3000, fastThreshold=5)
    k1, d1 = orb.detectAndCompute(clahe.apply(grigio_L), None)
    k2, d2 = orb.detectAndCompute(clahe.apply(grigio_R), None)
    if d1 is None or d2 is None:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(m) < 60:
        return None
    dy = np.array([k2[x.trainIdx].pt[1] - k1[x.queryIdx].pt[1] for x in m])
    dx = np.array([k1[x.queryIdx].pt[0] - k2[x.trainIdx].pt[0] for x in m])
    ok = (np.abs(dy) < 80) & (dx > -5) & (dx < 400)
    dy = dy[ok]
    if len(dy) < 60:
        return None
    hist, edges = np.histogram(dy, bins=np.arange(-80, 81, 2))
    picco = edges[hist.argmax()] + 1
    vicini = np.abs(dy - picco) < 4
    if vicini.mean() < 0.40:
        return None
    return -float(np.median(dy[vicini]))


class StereoDepthEstimator:
    """
    Motore completo di stereovisione, rettificazione ad alte prestazioni,
    stima 3D, isolamento ostacoli, auto-focus dinamico e diagnostica di sicurezza.
    """

    def __init__(self, config: DepthConfig):
        self.config = config
        self.has_calibration = False
        self.w_eye = 1280
        self.h_eye = 800

        # Offset calibrati (da calibrazione_offset.json)
        self.offset_x = -14
        self.offset_y = 28
        self.offset_file: Optional[Path] = None
        self._load_offset_config()

        # Parametri e mappe di rettificazione
        self.map_L_1 = None
        self.map_L_2 = None
        self.map_R_1 = None
        self.map_R_2 = None
        self.Q = np.eye(4, dtype=np.float64)
        self.Q_veloce = np.eye(4, dtype=np.float64)
        # Sfasamento verticale epipolare: default calibrato su rover -28.0px (coerente con calibrazione_offset.json)
        self.dy_epi = float(os.getenv("MAR_EPI_DY", "-28.0"))
        self.M_epi = np.float32([[1, 0, 0], [0, 1, self.dy_epi / SGBM_DIV]])
        self._dy_samples: List[float] = []
        self._dy_calibrated: bool = False

        # Carica calibrazione NPZ e calcola mappe veloci
        self._load_calibration()

        # Matcher stereoscopico: StereoSGBM 3-WAY calibrato di Ferraresso (affidabile e veloce su 320x200)
        use_bm = os.getenv("MAR_STEREO_BM", "0").strip() in ("1", "true", "yes")
        if use_bm:
            bm_block = 15 if SGBM_DIV >= 4 else 11
            self.stereo = cv2.StereoBM_create(
                numDisparities=NUM_DISPARITIES,
                blockSize=bm_block,
            )
            self.stereo.setPreFilterCap(31)
            self.stereo.setTextureThreshold(int(os.getenv("MAR_BM_TEX_THR", "3")))
            self.stereo.setUniquenessRatio(int(os.getenv("MAR_BM_UNIQUENESS", "7")))
            self.stereo.setSpeckleWindowSize(50)
            self.stereo.setSpeckleRange(2)
            print(f"  ✓ [DEPTH] Matcher=StereoBM block={bm_block} (fast path)")
        else:
            self.stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=NUM_DISPARITIES,
                blockSize=SGBM_BLOCK,
                P1=8 * 3 * SGBM_BLOCK ** 2,
                P2=32 * 3 * SGBM_BLOCK ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=15,
                speckleWindowSize=100,
                speckleRange=2,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            )
            print(f"  ✓ [DEPTH] Matcher=StereoSGBM 3-WAY (disp={NUM_DISPARITIES}, block={SGBM_BLOCK}, dy={self.dy_epi:+.1f}px)")

        # Stato modalità video (0=Mix fisso, 1=Nitida, 2=Depth, 3=Auto-Focus)
        self.modalita_vista = 3
        self.auto_switch_attivo = True
        self.offset_autofocus_smooth = float(self.offset_x)

        # Sicurezza e watchdog
        self.gs_prec = None
        self.frame_identici = 0
        self.sic_cattivi = 0
        self.sic_buoni = 0
        self.allarme_sic = False
        self.motivo_allarme = ""
        self.comando_prec = "STOP"
        self.t_ultimo_frame = time.time()
        # Decisione ponderata: niente LOW↔CRITICAL a ogni frame
        self.risk_stabilizer = RiskDecisionStabilizer()
        self._last_diag: Dict[str, Any] = {}
        self._dist_hist: List[float] = []
        self._dist_hist_max = int(os.getenv("MAR_DIST_MEDIAN_N", "7"))

    def _load_offset_config(self):
        """Carica calibrazione_offset.json se disponibile."""
        candidate_paths = [
            Path(__file__).parent / "calibrazione_offset.json",
            Path.cwd() / "calibrazione_offset.json",
            Path("/home/dietpi/stereovisione/calibrazione_offset.json")
        ]
        for p in candidate_paths:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.offset_x = int(data.get("offset_x", -14))
                    self.offset_y = int(data.get("offset_y", 28))
                    self.offset_file = p
                    print(f"  ✓ [DEPTH] Offset caricati da {p.name}: X={self.offset_x:+d}, Y={self.offset_y:+d}")
                    return
                except Exception as e:
                    print(f"  ⚠️ [DEPTH] Errore lettura offset ({e}).")
                    break

        self.offset_file = Path(__file__).parent / "calibrazione_offset.json"

    def salva_offset(self, ox: int, oy: int) -> bool:
        """Salva gli offset correnti su calibrazione_offset.json."""
        self.offset_x = ox
        self.offset_y = oy
        try:
            target = self.offset_file or (Path(__file__).parent / "calibrazione_offset.json")
            with open(target, "w", encoding="utf-8") as f:
                json.dump({"offset_x": int(ox), "offset_y": int(oy)}, f, indent=4)
            print(f"  ✓ [DEPTH] Offset salvati con successo in {target}: X={ox:+d}, Y={oy:+d}")
            return True
        except Exception as e:
            print(f"  ⚠️ [DEPTH] Errore salvataggio offset: {e}")
            return False

    def _load_calibration(self):
        """Carica parametri_calibrazione_stereo.npz e precalcola le mappe fixed-point."""
        candidate_paths = [
            Path(self.config.calibration_file) if self.config.calibration_file else None,
            Path(__file__).parent / "parametri_calibrazione_stereo.npz",
            Path.cwd() / "parametri_calibrazione_stereo.npz",
            Path("/home/dietpi/stereovisione/parametri_calibrazione_stereo.npz")
        ]

        calib_file = None
        for p in candidate_paths:
            if p and p.exists():
                calib_file = p
                break

        if not calib_file:
            print("  ⚠️ [DEPTH] File calibrazione stereo non trovato. Funzionamento SGBM sintetico.")
            return

        try:
            d = np.load(str(calib_file))
            mtx_L, dist_L = d["mtx_L"], d["dist_L"]
            mtx_R, dist_R = d["mtx_R"], d["dist_R"]
            R, T = d["R"], d["T"]

            img_size = (self.w_eye, self.h_eye)
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                mtx_L, dist_L, mtx_R, dist_R, img_size, R, T,
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
            )

            map_L_x, map_L_y = cv2.initUndistortRectifyMap(mtx_L, dist_L, R1, P1, img_size, cv2.CV_32FC1)
            map_R_x, map_R_y = cv2.initUndistortRectifyMap(mtx_R, dist_R, R2, P2, img_size, cv2.CV_32FC1)

            # Su ARM le mappe vengono sottocampionate (RECT_DIV) come nello script originale
            if RECT_DIV > 1:
                dim = (img_size[0] // RECT_DIV, img_size[1] // RECT_DIV)
                map_L_x = (cv2.resize(map_L_x, dim, interpolation=cv2.INTER_AREA) + 0.5) / RECT_DIV - 0.5
                map_L_y = (cv2.resize(map_L_y, dim, interpolation=cv2.INTER_AREA) + 0.5) / RECT_DIV - 0.5
                map_R_x = (cv2.resize(map_R_x, dim, interpolation=cv2.INTER_AREA) + 0.5) / RECT_DIV - 0.5
                map_R_y = (cv2.resize(map_R_y, dim, interpolation=cv2.INTER_AREA) + 0.5) / RECT_DIV - 0.5

            self.map_L_1, self.map_L_2 = mappe_veloci(map_L_x, map_L_y)
            self.map_R_1, self.map_R_2 = mappe_veloci(map_R_x, map_R_y)
            self.Q = Q
            # Disparità calcolata a ~eye/SGBM_DIV (poi eventuale STEP_SUB).
            # Q va scalato sullo STESSO fattore dei pixel di lavoro — NON su RECT_DIV.
            # Prima: scale=RECT_DIV*STEP_SUB e disp*SGBM_DIV → Z/X inconsistenti (0.5m↔1.7m).
            scale_q = float(SGBM_DIV) * float(max(1, STEP_SUB))
            self.Q_veloce = Q.copy()
            self.Q_veloce[0, 3] /= scale_q  # cx
            self.Q_veloce[1, 3] /= scale_q  # cy
            self.Q_veloce[2, 3] /= scale_q  # f  (disparity resta in pixel della griglia ridotta)
            self.has_calibration = True
            self._q_scale = scale_q

            f_px = abs(Q[2, 3])
            baseline_cm = (1.0 / abs(Q[3, 2])) / 10.0 if abs(Q[3, 2]) > 1e-6 else 6.0
            print(
                f"  ✓ [DEPTH] Calibrazione {calib_file.name}: Focale={f_px:.1f}px, Base={baseline_cm:.1f}cm | "
                f"ARM={IS_ARM} RECT_DIV={RECT_DIV} SGBM_DIV={SGBM_DIV} Q_scale={scale_q:.0f} numDisp={NUM_DISPARITIES}"
            )

        except Exception as e:
            print(f"  ⚠️ [DEPTH] Errore inizializzazione matrici di rettificazione: {e}")

    def calibra_allineamento_verticale(self, frame_sinistro: np.ndarray, frame_destro: np.ndarray):
        """Esegue stima epipolare del dy tra i due occhi."""
        if not self.has_calibration:
            return
        g_l = cv2.cvtColor(cv2.remap(frame_sinistro, self.map_L_1, self.map_L_2, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
        g_r = cv2.cvtColor(cv2.remap(frame_destro, self.map_R_1, self.map_R_2, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
        v = stima_dy_epipolare(g_l, g_r)
        if v is not None:
            self.dy_epi = float(v)
            self.M_epi = np.float32([[1, 0, 0], [0, 1, self.dy_epi / SGBM_DIV]])
            print(f"  ✓ [DEPTH] Sfasamento verticale epipolare stimato: {self.dy_epi:+.1f}px")

    def valuta_sicurezza_lenti(self, gL: np.ndarray, gR: np.ndarray) -> Optional[str]:
        """Controlla se una o entrambe le lenti sono coperte, saturate o spente."""
        hh, ww = gL.shape[:2]
        r0, r1, c0, c1 = hh // 4, 3 * hh // 4, ww // 4, 3 * ww // 4
        misure = []
        for g in (gL, gR):
            roi = g[r0:r1, c0:c1]
            media, dev = cv2.meanStdDev(roi)
            grad = float(np.abs(cv2.Sobel(roi, cv2.CV_16S, 1, 0, ksize=3)).mean())
            misure.append((float(media[0][0]), float(dev[0][0]), grad))

        for i, (m, sd, gr) in enumerate(misure):
            nome = f"LENTE {i + 1}"
            if m < 6.0:
                return f"{nome} oscurata (immagine completamente nera)"
            if m > 250.0 and sd < 3.0:
                return f"{nome} saturata (immagine bianca)"
            # Allarme coperta solo se deviazione standard microscopica (oggetto piatto premuto sulla lente)
            if sd < 2.0 and gr < 0.8 and m > 30.0:
                return f"{nome} coperta da ostacolo cieco"

        (m1, _, g1), (m2, _, g2) = misure
        if abs(m1 - m2) > 65.0:
            return "luminosita' asimmetrica tra le lenti (una coperta?)"
        if max(g1, g2) > 8.0 and max(g1, g2) / max(min(g1, g2), 0.5) > 6.0:
            return "una lente vede meno dettaglio dell'altra"
        return None

    def process_stereo_frame(
        self,
        left_raw: np.ndarray,
        right_raw: np.ndarray
    ) -> Dict[str, Any]:
        """
        Elabora la coppia stereo completa:
        - Rettificazione con mappe veloci
        - Diagnostica sicurezza lenti e watchdog
        - Calcolo disparità SGBM con filtro texture Sobel
        - Ricostruzione 3D nel corridoio utile
        - Rilevamento ostacolo critico (filtro densità + componenti connesse)
        - Auto-Focus dinamico (inseguimento offset -disp)
        - Mappa top-down e decisione marcia (AVANTI / FRENA / STOP)
        - Rendering viste (Modo 0, 1, 2, 3)
        """
        h, w = left_raw.shape[:2]
        self.t_ultimo_frame = time.time()

        # 1. Separazione, conversione scala di grigi e rettificazione
        g_L = cv2.cvtColor(left_raw, cv2.COLOR_BGR2GRAY)
        g_R = cv2.cvtColor(right_raw, cv2.COLOR_BGR2GRAY)

        # Pre-sottocampiona il grezzo se RECT_DIV>1 (mappe già scalate)
        if RECT_DIV > 1 and not self.has_calibration:
            g_L = cv2.resize(g_L, (w // RECT_DIV, h // RECT_DIV), interpolation=cv2.INTER_AREA)
            g_R = cv2.resize(g_R, (w // RECT_DIV, h // RECT_DIV), interpolation=cv2.INTER_AREA)

        if self.has_calibration:
            rect_L = cv2.remap(g_L, self.map_L_1, self.map_L_2, cv2.INTER_LINEAR)
            rect_R = cv2.remap(g_R, self.map_R_1, self.map_R_2, cv2.INTER_LINEAR)
        else:
            rect_L, rect_R = g_L, g_R

        # 2. Input SGBM: scala RECT_DIV/SGBM_DIV rispetto alla rettificata (come script originale)
        _f = float(RECT_DIV) / float(SGBM_DIV)
        if abs(_f - 1.0) > 1e-3:
            gs_L = cv2.resize(rect_L, None, fx=_f, fy=_f, interpolation=cv2.INTER_AREA)
            gs_R = cv2.resize(rect_R, None, fx=_f, fy=_f, interpolation=cv2.INTER_AREA)
        else:
            gs_L, gs_R = rect_L, rect_R

        # Auto-allineamento verticale epipolare nei primi frame (come stereovisione_robot.4py.py)
        if not self._dy_calibrated and len(self._dy_samples) < 12 and self.has_calibration:
            v_est = stima_dy_epipolare(rect_L, rect_R)
            if v_est is not None:
                self._dy_samples.append(float(v_est))
                if len(self._dy_samples) >= 3 and (max(self._dy_samples) - min(self._dy_samples)) < 6.0:
                    self.dy_epi = float(np.median(self._dy_samples))
                    self.M_epi = np.float32([[1, 0, 0], [0, 1, self.dy_epi / SGBM_DIV]])
                    self._dy_calibrated = True
                    print(f"  ✓ [DEPTH] Sfasamento verticale stimato e sincronizzato: {self.dy_epi:+.1f}px")

        if self.has_calibration and self.M_epi is not None:
            gs_R = cv2.warpAffine(gs_R, self.M_epi, (gs_R.shape[1], gs_R.shape[0]))

        # Disparità (BM o SGBM). OpenCV restituisce fixed-point ×16.
        disp_small = self.stereo.compute(gs_L, gs_R).astype(np.float32) / 16.0
        # Elimina picco di saturazione / bezel ai bordi
        disp_small[disp_small >= (NUM_DISPARITIES - 4.0)] = -1.0

        # Texture filter: scarta regioni senza gradiente per evitare falsi ostacoli
        gx = cv2.Sobel(gs_L, cv2.CV_16S, 1, 0, ksize=3)
        tex = cv2.blur(np.abs(gx).astype(np.float32), (5, 5))
        hh_s, ww_s = tex.shape[:2]
        disp_small[tex < TEX_MIN] = -1.0

        # FOV bloccato = foglio/mano che tappa TUTTA l'inquadratura (quasi uniforme).
        # NON usare solo il centro: un corridoio scuro / porta senza texture dava falsi CRITICAL.
        c0, c1 = int(ww_s * 0.25), int(ww_s * 0.75)
        r0, r1 = int(hh_s * 0.28), int(hh_s * 0.85)
        center_disp = disp_small[r0:r1, c0:c1]
        center_tex = tex[r0:r1, c0:c1]
        invalid_ratio = 1.0 - (np.count_nonzero(center_disp > 0) / max(1, center_disp.size))
        full_tex = float(tex.mean())
        full_std = float(gs_L.std())
        full_mean = float(gs_L.mean())
        # Blocco FOV reale (mano/cartone premuto contro la camera)
        fov_blocked = bool(
            invalid_ratio > 0.95
            and float(center_tex.mean()) < 1.0
            and full_std < 4.0
            and (full_mean > 40.0)
        )

        # 3. Diagnostica sicurezza lenti
        motivo_sic = self.valuta_sicurezza_lenti(gs_L, gs_R)
        if motivo_sic is None and np.count_nonzero(disp_small > 0) < SIC_MIN_PUNTI_VALIDI * disp_small.size:
            motivo_sic = "visione cieca: disparita' 3D insufficiente"

        if self.gs_prec is not None and self.gs_prec.shape == gs_L.shape and not cv2.absdiff(gs_L, self.gs_prec).any():
            self.frame_identici += 1
        else:
            self.frame_identici = 0
        self.gs_prec = gs_L.copy()

        if motivo_sic is None and self.frame_identici >= SIC_FRAME_CONGELATI:
            motivo_sic = "camera ferma: frame identici"

        if motivo_sic:
            self.sic_cattivi += 1
            self.sic_buoni = 0
        else:
            self.sic_buoni += 1
            self.sic_cattivi = 0

        if not self.allarme_sic and self.sic_cattivi >= SIC_FRAME_ATTIVAZIONE:
            self.allarme_sic = True
            self.motivo_allarme = motivo_sic
        elif self.allarme_sic and motivo_sic:
            self.motivo_allarme = motivo_sic
        elif self.allarme_sic and self.sic_buoni >= SIC_FRAME_RIENTRO:
            self.allarme_sic = False
            self.motivo_allarme = ""

        # 4. Ripenetrazione spaziale 3D nel corridoio utile
        # Disparità scalata in pixel pieni (allineata a Q_veloce / 4)
        disparity_veloce = disp_small[::STEP_SUB, ::STEP_SUB].astype(np.float32) * float(SGBM_DIV)
        Q_veloce = self.Q.copy()
        Q_veloce[0, 3] /= 4.0
        Q_veloce[1, 3] /= 4.0
        punti_3d = cv2.reprojectImageTo3D(disparity_veloce, Q_veloce)
        punti_3d_cm = punti_3d / 10.0
        H_vel, W_vel = disparity_veloce.shape[:2]

        # Griglia Spaziale 3D a 9 Zone (Alto, Centro, Basso x SX, Centro, DX)
        spatial_grid_3d: Dict[str, Tuple[float, float, float]] = {}
        h_step, w_step = max(1, H_vel // 3), max(1, W_vel // 3)
        nomi_riga = ["Alto", "Centro", "Basso"]
        nomi_col = ["SX", "Centro", "DX"]

        # Trasformazione nel frame del robot (rotazione 180° per montaggio a testa in giù)
        punti_robot_cm = np.zeros_like(punti_3d_cm)
        punti_robot_cm[:, :, 0] = -np.flip(np.flip(punti_3d_cm[:, :, 0], 0), 1)
        punti_robot_cm[:, :, 1] = -np.flip(np.flip(punti_3d_cm[:, :, 1], 0), 1)
        punti_robot_cm[:, :, 2] = np.flip(np.flip(punti_3d_cm[:, :, 2], 0), 1)

        for r in range(3):
            for c in range(3):
                nome_settore = f"{nomi_riga[r]}_{nomi_col[c]}"
                roi_Z = punti_robot_cm[r * h_step:(r + 1) * h_step, c * w_step:(c + 1) * w_step, 2]
                roi_X = punti_robot_cm[r * h_step:(r + 1) * h_step, c * w_step:(c + 1) * w_step, 0]
                roi_Y = punti_robot_cm[r * h_step:(r + 1) * h_step, c * w_step:(c + 1) * w_step, 1]

                valid_grid = (roi_Z > 20.0) & (roi_Z <= 450.0) & np.isfinite(roi_Z)
                if np.count_nonzero(valid_grid) > 12:
                    z_vals = roi_Z[valid_grid]
                    q_l, q_h = np.percentile(z_vals, [15, 85])
                    filt_grid = (roi_Z >= q_l) & (roi_Z <= q_h) & valid_grid
                    if np.count_nonzero(filt_grid) > 0:
                        m_z = float(np.median(roi_Z[filt_grid]))
                        m_x = float(np.median(roi_X[filt_grid]))
                        m_y = float(np.median(roi_Y[filt_grid]))
                        spatial_grid_3d[nome_settore] = (round(m_x, 1), round(m_y, 1), round(m_z, 1))
                if nome_settore not in spatial_grid_3d:
                    spatial_grid_3d[nome_settore] = (0.0, 0.0, 999.0)

        z_basso_cm = spatial_grid_3d.get("Basso_Centro", (0.0, 0.0, 999.0))[2]
        z_centro_cm = spatial_grid_3d.get("Centro_Centro", (0.0, 0.0, 999.0))[2]
        z_alto_cm = spatial_grid_3d.get("Alto_Centro", (0.0, 0.0, 999.0))[2]
        ground_hazard = bool(z_basso_cm < 40.0)

        # La telecamera è fisicamente invertita di 180°:
        # Le righe 0 .. 0.28*H_vel corrispondono al basamento/tavolo del rover.
        r_utile_min = int(H_vel * 0.28)
        r_utile_max = int(H_vel * 0.85)

        # Corridoio nel FRAME ROBOT (non nel cloud grezzo non-flippato)
        roi_robot = punti_robot_cm[r_utile_min:r_utile_max, :, :]
        mask_corridoio = (
            (roi_robot[:, :, 2] > 25.0) &
            (roi_robot[:, :, 2] < 220.0) &
            (np.abs(roi_robot[:, :, 0]) < 35.0) &
            (roi_robot[:, :, 1] > ALTEZZA_MIN_ROBOT) &
            (roi_robot[:, :, 1] < ALTEZZA_MAX_ROBOT) &
            np.isfinite(roi_robot[:, :, 2]) &
            (disparity_veloce[r_utile_min:r_utile_max, :] > 0.5)
        )
        mask_corridoio = filtra_densita(mask_corridoio)

        ostacolo_trovato = False
        dist_ostacolo_cm = 999.0
        bbox_ostacolo = None
        offset_target = float(self.offset_x)

        if np.sum(mask_corridoio) > 15:
            z_ostacoli_corridoio = roi_robot[:, :, 2][mask_corridoio]
            z_min_ostacolo = float(np.percentile(z_ostacoli_corridoio, 15))

            mask_target = mask_corridoio & (roi_robot[:, :, 2] <= (z_min_ostacolo + 35.0))
            blob = isola_blob_compatto(mask_target)
            if blob is not None:
                sub_rows, cols_obs = np.where(blob)
                rows_obs = sub_rows + r_utile_min

                if len(rows_obs) >= 12:
                    ostacolo_trovato = True
                    dist_ostacolo_cm = float(np.median(roi_robot[:, :, 2][blob]))
                    disp_mediana_obs = float(np.median(disparity_veloce[r_utile_min:r_utile_max, :][blob]))

                    # BBox nel frame eye (robot upright = già flip camera): scala griglia → eye
                    scale_px = float(SGBM_DIV) * float(max(1, STEP_SUB))
                    r_min, r_max = int(np.min(rows_obs)), int(np.max(rows_obs))
                    c_min, c_max = int(np.min(cols_obs)), int(np.max(cols_obs))
                    bx1 = max(0, int(c_min * scale_px))
                    bx2 = min(self.w_eye - 1, int((c_max + 1) * scale_px))
                    by1 = max(0, int(r_min * scale_px))
                    by2 = min(self.h_eye - 1, int((r_max + 1) * scale_px))
                    bbox_ostacolo = (bx1, by1, bx2, by2)

                    # Autofocus stereoscopico esatto: sposta R di -disp_mediana_obs
                    offset_target = float(np.clip(-disp_mediana_obs, -120.0, 40.0))

        # Stima parete frontale (frame robot, ROI centrale)
        c_roi_inizio = int(W_vel * 0.30)
        c_roi_fine = int(W_vel * 0.70)
        roi_centrale_z = punti_robot_cm[r_utile_min:r_utile_max, c_roi_inizio:c_roi_fine, 2]
        valide_centrale = roi_centrale_z[
            (roi_centrale_z > 30.0) & (roi_centrale_z < MAX_DIST_CM) & np.isfinite(roi_centrale_z)
        ]

        if len(valide_centrale) > 20:
            # 25° percentile: reattivo agli ostacoli, meno rumore del min assoluto
            distanza_fronte_cm = float(np.percentile(valide_centrale, 25))
        elif ostacolo_trovato:
            distanza_fronte_cm = dist_ostacolo_cm
        else:
            distanza_fronte_cm = 230.0

        # FOV ostruito vero (foglio): solo se NON abbiamo già una distanza stereo affidabile.
        # Altrimenti un false-positive FOV bloccava CRITICAL a 30cm su stanze normali.
        has_trustworthy_z = bool(
            ostacolo_trovato and 15.0 < float(dist_ostacolo_cm) < 220.0
        ) or bool(25.0 < float(distanza_fronte_cm) < 220.0 and len(valide_centrale) > 15)
        fov_for_decision = bool(fov_blocked) and not has_trustworthy_z
        if fov_for_decision:
            ostacolo_trovato = True

        # Filtro passa-basso sull'autofocus
        self.offset_autofocus_smooth = 0.70 * self.offset_autofocus_smooth + 0.30 * offset_target
        offset_autofocus_x = int(round(self.offset_autofocus_smooth))

        # Distanza grezza: preferisci blob ostacolo; altrimenti fronte.
        # Mediana su N frame → niente salti 50↔230 che tengono HIGH bloccato.
        distanza_instant = float(dist_ostacolo_cm if ostacolo_trovato else distanza_fronte_cm)
        # Non avvelenare lo storico con il sentinel "vuoto" (230) se poco prima c'era un ostacolo
        if distanza_instant >= 220.0 and any(d < 150.0 for d in self._dist_hist[-5:]):
            pass
        else:
            self._dist_hist.append(distanza_instant)
            if len(self._dist_hist) > self._dist_hist_max:
                self._dist_hist = self._dist_hist[-self._dist_hist_max:]
        distanza_raw = float(np.median(np.asarray(self._dist_hist, dtype=np.float64))) if self._dist_hist else distanza_instant
        distanza_per_switch = float(distanza_raw)
        if self.auto_switch_attivo:
            if distanza_per_switch < SOGLIA_DISTANZA_CM:
                self.modalita_vista = 3
            elif distanza_per_switch > (SOGLIA_DISTANZA_CM + ISTERESI_CM):
                self.modalita_vista = 0

        # 5. Mappa di navigazione top-down 2D
        elenco_punti = punti_3d_cm.reshape(-1, 3)
        X = elenco_punti[:, 0]
        Y = elenco_punti[:, 1]
        Z = elenco_punti[:, 2]

        maschera_ostacoli = (
            (Z > 10.0) & (Z < MAX_DIST_CM) &
            (Y > ALTEZZA_MIN_ROBOT) & (Y < ALTEZZA_MAX_ROBOT) &
            np.isfinite(Z) & (disparity_veloce.reshape(-1) > 0.0)
        )

        X_ostacoli = X[maschera_ostacoli]
        Z_ostacoli = Z[maschera_ostacoli]

        mappa_navigazione = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
        robot_x_pixel = MAP_SIZE // 2
        robot_z_pixel = MAP_SIZE - 15

        # Disegna griglia metrica e corridoio rover
        cv2.circle(mappa_navigazione, (robot_x_pixel, robot_z_pixel), 6, (0, 255, 0), -1)
        for dist_cerchio_cm in (50, 100, 150, 200):
            rz = int(robot_z_pixel - (dist_cerchio_cm / RISOLUZIONE_CM))
            if 0 <= rz < MAP_SIZE:
                cv2.line(mappa_navigazione, (10, rz), (MAP_SIZE - 10, rz), (40, 40, 40), 1)
                cv2.putText(mappa_navigazione, f"{dist_cerchio_cm}cm", (12, rz - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

        # Traccia corridoio guida (larghezza +/- 40cm)
        x_corr_sx = int(robot_x_pixel - 40 / RISOLUZIONE_CM)
        x_corr_dx = int(robot_x_pixel + 40 / RISOLUZIONE_CM)
        cv2.line(mappa_navigazione, (x_corr_sx, 20), (x_corr_sx, robot_z_pixel), (0, 70, 0), 1)
        cv2.line(mappa_navigazione, (x_corr_dx, 20), (x_corr_dx, robot_z_pixel), (0, 70, 0), 1)

        if len(Z_ostacoli) > 0:
            mappa_x_pixel = (X_ostacoli / RISOLUZIONE_CM + (MAP_SIZE // 2)).astype(np.int32)
            mappa_z_pixel = (robot_z_pixel - (Z_ostacoli / RISOLUZIONE_CM)).astype(np.int32)

            validi_a_schermo = (
                (mappa_x_pixel >= 0) & (mappa_x_pixel < MAP_SIZE) &
                (mappa_z_pixel >= 0) & (mappa_z_pixel < MAP_SIZE)
            )
            mappa_navigazione[mappa_z_pixel[validi_a_schermo], mappa_x_pixel[validi_a_schermo]] = [0, 0, 255]

        # 6. Logica di guida — distanza frontale + decisione ponderata (EMA/isteresi/dwell)
        ostacoli_centrali = (X_ostacoli > -20.0) & (X_ostacoli < 20.0) & (Z_ostacoli < DIST_MODERATE_CM)
        punti_pericolo = int(np.sum(ostacoli_centrali))

        comando, motivo_cmd, risk_level, dist_decision = self.risk_stabilizer.update(
            float(distanza_raw),
            fov_blocked=bool(fov_for_decision),
            safety_alarm=bool(self.allarme_sic),
            safety_reason=self.motivo_allarme or "",
        )
        # Distanza pubblicata = EMA stabile (non il grezzo che oscilla)
        distanza_per_switch = float(dist_decision)
        self.comando_prec = comando
        self._last_diag = {
            "fov_blocked_raw": bool(fov_blocked),
            "fov_for_decision": bool(fov_for_decision),
            "invalid_ratio": round(float(invalid_ratio), 3),
            "full_tex": round(full_tex, 2),
            "full_std": round(full_std, 2),
            "dist_raw_cm": round(float(distanza_raw), 1),
            "has_trustworthy_z": bool(has_trustworthy_z),
        }

        # 7. Rendering delle modalità video (raddrizzate 180° per correggere il montaggio)
        flip_L = cv2.flip(rect_L, -1)
        flip_R = cv2.flip(rect_R, -1)

        if self.modalita_vista == 0:
            # Modo 0: Mix Stereo 50/50 fisso
            if self.offset_x != 0 or self.offset_y != 0:
                M = np.float32([[1, 0, self.offset_x], [0, 1, self.offset_y]])
                shifted_R = cv2.warpAffine(flip_R, M, (flip_R.shape[1], flip_R.shape[0]))
            else:
                shifted_R = flip_R
            blend = cv2.addWeighted(flip_L, 0.5, shifted_R, 0.5, 0)
            composite_view = cv2.cvtColor(blend, cv2.COLOR_GRAY2BGR)
            titolo_modo = "Mix Stereo 50/50 (Fisso)"

        elif self.modalita_vista == 1:
            # Modo 1: Nitida FPV monoculare principale (path più veloce)
            composite_view = cv2.cvtColor(flip_L, cv2.COLOR_GRAY2BGR)
            titolo_modo = "Camera Principale (Nitida FPV)"

        elif self.modalita_vista == 2:
            # Modo 2: Mappa di profondità a colori
            disp_full = cv2.resize(disp_small * float(SGBM_DIV), (self.w_eye, self.h_eye), interpolation=cv2.INTER_NEAREST)
            disp_norm = np.clip(disp_full * (255.0 / (NUM_DISPARITIES * SGBM_DIV)), 0, 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
            depth_color[disp_full <= 0] = 0
            composite_view = cv2.flip(depth_color, -1)
            titolo_modo = "Mappa Profondita' (3D Heatmap)"

        else:
            # Modo 3: Auto-Focus Ostacolo
            if offset_autofocus_x != 0 or self.offset_y != 0:
                M = np.float32([[1, 0, offset_autofocus_x], [0, 1, self.offset_y]])
                shifted_R = cv2.warpAffine(flip_R, M, (flip_R.shape[1], flip_R.shape[0]))
            else:
                shifted_R = flip_R
            blend = cv2.addWeighted(flip_L, 0.5, shifted_R, 0.5, 0)
            composite_view = cv2.cvtColor(blend, cv2.COLOR_GRAY2BGR)
            titolo_modo = "Auto-Focus Ostacolo (Modo 3)"

            if ostacolo_trovato and bbox_ostacolo is not None:
                bx1, by1, bx2, by2 = bbox_ostacolo
                colore_box = (0, 0, 255) if dist_ostacolo_cm < 120.0 else ((0, 165, 255) if dist_ostacolo_cm < 180.0 else (0, 255, 0))
                cv2.rectangle(composite_view, (bx1, by1), (bx2, by2), colore_box, 2)
                mcx = (bx1 + bx2) // 2
                mcy = (by1 + by2) // 2
                cv2.drawMarker(composite_view, (mcx, mcy), colore_box, cv2.MARKER_CROSS, 24, 2)
                label = f"OSTACOLO: {dist_ostacolo_cm:.0f}cm [AF: {offset_autofocus_x:+d}px]"
                cv2.putText(composite_view, label, (max(10, bx1), max(35, by1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, colore_box, 2)
            else:
                mcx, mcy = composite_view.shape[1] // 2, composite_view.shape[0] // 2
                cv2.drawMarker(composite_view, (mcx, mcy), (0, 255, 100), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(composite_view, f"PARETE: {distanza_fronte_cm:.0f}cm [Fuoco {offset_autofocus_x:+d}px]",
                            (max(10, mcx - 140), mcy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2)

        # OSD Testo e allarme sicurezza
        colore_osd = (0, 0, 255) if distanza_per_switch < SOGLIA_DISTANZA_CM else (0, 255, 0)
        auto_tag = "[AUTO]" if self.auto_switch_attivo else "[MANUALE]"
        cv2.putText(
            composite_view,
            f"Dist: {distanza_per_switch:.0f}cm | {titolo_modo} {auto_tag} | Offset X:{self.offset_x} Y:{self.offset_y}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            colore_osd,
            2
        )

        if self.allarme_sic:
            vh, vw = composite_view.shape[:2]
            cv2.rectangle(composite_view, (0, 0), (vw - 1, vh - 1), (0, 0, 255), 6)
            txt_stop = f"STOP - {self.motivo_allarme}"
            cv2.rectangle(composite_view, (0, vh // 2 - 25), (vw - 1, vh // 2 + 25), (0, 0, 160), -1)
            cv2.putText(composite_view, txt_stop, (max(20, (vw - len(txt_stop) * 14) // 2), vh // 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return {
            "rect_left": flip_L,
            "rect_right": flip_R,
            "composite": composite_view,
            "map_2d": mappa_navigazione,
            "eye_w": int(self.w_eye),
            "eye_h": int(self.h_eye),
            "obstacle_found": ostacolo_trovato,
            "obstacle_dist_cm": dist_ostacolo_cm,
            "target_dist_cm": distanza_per_switch,
            "spatial_grid_3d": spatial_grid_3d,
            "z_basso_cm": z_basso_cm,
            "z_centro_cm": z_centro_cm,
            "z_alto_cm": z_alto_cm,
            "ground_hazard": ground_hazard,
            "obstacle_bbox": bbox_ostacolo,
            "autofocus_offset_x": offset_autofocus_x,
            "current_mode": self.modalita_vista,
            "auto_switch": self.auto_switch_attivo,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "driving_command": comando,
            "driving_reason": motivo_cmd,
            "risk_level": risk_level,
            "fov_blocked": bool(fov_for_decision),
            "fov_blocked_raw": bool(fov_blocked),
            "safety_alarm": self.allarme_sic,
            "safety_reason": self.motivo_allarme,
            "danger_points": punti_pericolo,
            "diag": dict(self._last_diag),
            "bands_cm": {
                "critical": DIST_CRITICAL_CM,
                "high": DIST_HIGH_CM,
                "moderate": DIST_MODERATE_CM,
            },
        }

    def set_mode(self, mode: int):
        """Sets display mode 0, 1, 2, or 3."""
        if 0 <= mode <= 3:
            self.modalita_vista = mode
            print(f"[DEPTH] Modalità impostata su {mode}")

    def toggle_auto(self, force_val: Optional[bool] = None):
        """Toggles or sets automatic switching between Modo 0 and Modo 3."""
        if force_val is not None:
            self.auto_switch_attivo = force_val
        else:
            self.auto_switch_attivo = not self.auto_switch_attivo
        print(f"[DEPTH] Auto-Switch: {self.auto_switch_attivo}")

    def adjust_offset(self, dx: int = 0, dy: int = 0):
        """Adjusts horizontal and vertical optical calibration offsets."""
        self.offset_x += dx
        self.offset_y += dy
        self.M_epi[0, 2] = float(self.offset_x)
        self.M_epi[1, 2] = float(self.offset_y)
        print(f"[DEPTH] Offset aggiornato: X={self.offset_x}, Y={self.offset_y}")

    def save_current_offset(self) -> bool:
        """Saves current offsets to disk."""
        return self.salva_offset(self.offset_x, self.offset_y)

    def reset_offset(self):
        """Resets offsets to 0, 0."""
        self.offset_x = 0
        self.offset_y = 0
        self.M_epi[0, 2] = 0.0
        self.M_epi[1, 2] = 0.0
        print("[DEPTH] Offset resettato a 0, 0")

    def handle_command(self, action: str, value: Any = ""):
        """Dispatches commands from Web UI or REST API."""
        action = action.lower()
        if action == "modo":
            try:
                self.set_mode(int(value))
            except ValueError:
                pass
        elif action == "auto":
            if str(value).lower() in ("true", "1", "on"):
                self.toggle_auto(True)
            elif str(value).lower() in ("false", "0", "off"):
                self.toggle_auto(False)
            else:
                self.toggle_auto()
        elif action == "dx":
            try:
                self.adjust_offset(dx=int(value))
            except ValueError:
                pass
        elif action == "dy":
            try:
                self.adjust_offset(dy=int(value))
            except ValueError:
                pass
        elif action == "salva":
            self.save_current_offset()
        elif action == "reset":
            self.reset_offset()

