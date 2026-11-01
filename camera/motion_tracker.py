#!/usr/bin/env python3
"""
M.A.R. — 3D Spatial Motion & Approach Tracker
Tracks 3D positions over time to estimate:
  - Approach / Departure velocity (dZ/dt in cm/s)
  - Lateral motion drift (dX/dt in cm/s)
  - Time-To-Collision (TTC in seconds)
  - Full spatial scene overview (clearance left, center, right)
  - Recommended reactive evasion direction
"""

from __future__ import annotations
import time
from typing import Dict, Tuple, Optional, List
import numpy as np


class ObjectMotionTracker:
    """
    Stima il vettore di movimento 3D degli ostacoli nel corridoio del rover,
    il tempo stimato all'impatto (TTC) e la visione d'insieme della scena spaziale.
    """

    def __init__(self, history_len: int = 8, ema_alpha: float = 0.40):
        self.history_len = history_len
        self.ema_alpha = ema_alpha

        # Storico campioni: (timestamp, x_cm, y_cm, z_cm)
        self.history: List[Tuple[float, float, float, float]] = []

        # Velocità stimate (filtro EMA) in cm/s
        self.vz_ema: float = 0.0
        self.vx_ema: float = 0.0
        self.last_update_ts: float = 0.0

        # Stato corrente
        self.motion_state: str = "STABILE"
        self.ttc_sec: float = 999.0
        self.is_approaching: bool = False

    def update(
        self,
        target_z_cm: float,
        target_x_cm: float = 0.0,
        target_y_cm: float = 0.0,
        spatial_grid: Optional[Dict[str, Tuple[float, float, float]]] = None,
        now: Optional[float] = None
    ) -> Dict[str, object]:
        """
        Aggiorna il tracciamento con una nuova misura (x, y, z) a timestamp corrente.
        Restituisce un dizionario completo di telemetria del movimento e visione d'insieme.
        """
        t = time.time() if now is None else float(now)

        # Pulizia storico se ci sono buchi temporali superiori a 1.2s
        if self.last_update_ts > 0 and (t - self.last_update_ts) > 1.2:
            self.history.clear()
            self.vz_ema = 0.0
            self.vx_ema = 0.0

        self.last_update_ts = t

        # Aggiungi campione valido
        if 10.0 <= target_z_cm <= 450.0:
            self.history.append((t, float(target_x_cm), float(target_y_cm), float(target_z_cm)))
            if len(self.history) > self.history_len:
                self.history = self.history[-self.history_len:]

        # Calcolo velocità se abbiamo almeno 2 campioni
        vz_inst = 0.0
        vx_inst = 0.0
        if len(self.history) >= 2:
            t_old, x_old, y_old, z_old = self.history[0]
            t_cur, x_cur, y_cur, z_cur = self.history[-1]
            dt = t_cur - t_old

            if dt >= 0.08:
                # vz < 0 significa che z sta diminuendo (l'oggetto si sta AVVICINANDO)
                vz_inst = (z_cur - z_old) / dt
                vx_inst = (x_cur - x_old) / dt

                self.vz_ema = (self.ema_alpha * vz_inst) + ((1.0 - self.ema_alpha) * self.vz_ema)
                self.vx_ema = (self.ema_alpha * vx_inst) + ((1.0 - self.ema_alpha) * self.vx_ema)

        # Time-To-Collision (TTC): se vz_ema < -8 cm/s (avvicinamento significativo)
        if self.vz_ema < -8.0 and target_z_cm > 15.0:
            self.ttc_sec = max(0.1, round(target_z_cm / abs(self.vz_ema), 2))
            self.is_approaching = True
        else:
            self.ttc_sec = 999.0
            self.is_approaching = False

        # Classificazione stato di movimento
        if self.vz_ema < -35.0:
            self.motion_state = "AVVICINAMENTO RAPIDO"
        elif self.vz_ema < -10.0:
            self.motion_state = "AVVICINAMENTO"
        elif self.vz_ema > 30.0:
            self.motion_state = "ALLONTANAMENTO RAPIDO"
        elif self.vz_ema > 10.0:
            self.motion_state = "ALLONTANAMENTO"
        elif abs(self.vx_ema) > 15.0:
            self.motion_state = "DEVIAZIONE_DX" if self.vx_ema > 0 else "DEVIAZIONE_SX"
        else:
            self.motion_state = "STABILE"

        # Visione d'insieme dallo Spatial Grid 3D (SX vs CENTRO vs DX)
        clearance_sx = 999.0
        clearance_cnt = 999.0
        clearance_dx = 999.0
        suggerimento_fuga = "AVANTI"

        if spatial_grid:
            # Estrai distanze Z per colonne SX, Centro, DX
            pts_sx = [spatial_grid[k][2] for k in spatial_grid if k.endswith("_SX") and 15.0 < spatial_grid[k][2] < 900.0]
            pts_cnt = [spatial_grid[k][2] for k in spatial_grid if k.endswith("_Centro") and 15.0 < spatial_grid[k][2] < 900.0]
            pts_dx = [spatial_grid[k][2] for k in spatial_grid if k.endswith("_DX") and 15.0 < spatial_grid[k][2] < 900.0]

            clearance_sx = round(float(np.median(pts_sx)), 1) if pts_sx else 999.0
            clearance_cnt = round(float(np.median(pts_cnt)), 1) if pts_cnt else target_z_cm
            clearance_dx = round(float(np.median(pts_dx)), 1) if pts_dx else 999.0

            # Se il centro ha un ostacolo (< 90cm)
            if clearance_cnt < 90.0 or target_z_cm < 90.0:
                if clearance_dx >= clearance_sx + 15.0 and clearance_dx > 80.0:
                    suggerimento_fuga = "DEVIA_DESTRA"
                elif clearance_sx >= clearance_dx + 15.0 and clearance_sx > 80.0:
                    suggerimento_fuga = "DEVIA_SINISTRA"
                elif clearance_cnt < 30.0 or self.ttc_sec < 1.2:
                    suggerimento_fuga = "STOP_EMERGENZA"
                else:
                    suggerimento_fuga = "FRENA"

        return {
            "motion_state": self.motion_state,
            "velocity_z_cms": round(self.vz_ema, 1),
            "velocity_x_cms": round(self.vx_ema, 1),
            "is_approaching": self.is_approaching,
            "ttc_sec": self.ttc_sec,
            "clearance_sx_cm": clearance_sx,
            "clearance_center_cm": clearance_cnt,
            "clearance_dx_cm": clearance_dx,
            "evasion_recommendation": suggerimento_fuga,
            "target_z_cm": round(target_z_cm, 1),
            "timestamp": t
        }
