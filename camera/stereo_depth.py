#!/usr/bin/env python3
"""
M.A.R. — Stereo Disparity, 3D Depth & 9-Zone Spatial Grid Module
Calculates stereo disparity maps (StereoSGBM + WLS Filter) using calibrated
camera matrices and projects disparity into a 9-Zone 3D Spatial Grid (X, Y, Z in cm)
with low-profile ground obstacle detection (STOP_TERRA).
"""

from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import os
import cv2
import numpy as np

try:
    from ..config import DepthConfig
except (ImportError, ValueError):
    from config import DepthConfig


class StereoDepthEstimator:
    """
    Computes real-time stereo disparity, metric depth, and a 9-zone 3D spatial grid.
    Incorporates calibrated focal length and baseline from physical robot testing.
    """

    def __init__(self, config: DepthConfig):
        self.config = config
        self.focale_px = config.focal_length_px
        self.cx = 320.0
        self.cy = 240.0
        self.baseline_cm = config.baseline_cm
        self.has_calibration = False

        # Load physical calibration parameters if available
        self._load_calibration()

        # CLAHE contrast enhancement for robust feature matching under varying light
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Stereo SGBM Matcher
        self.stereo_L = self._create_matcher()

        # WLS Disparity Filter (if cv2.ximgproc is available)
        self.use_wls = False
        self.wls_filter = None
        self.stereo_R = None
        if config.use_wls_filter:
            self._init_wls_filter()

    def _load_calibration(self):
        """Loads calibration file (parametri_calibrazione_stereo.npz)."""
        candidate_paths = [
            Path(self.config.calibration_file) if self.config.calibration_file else None,
            Path(__file__).parent / "parametri_calibrazione_stereo.npz",
            Path(__file__).parent.parent / "parametri_calibrazione_stereo.npz",
            Path.cwd() / "parametri_calibrazione_stereo.npz"
        ]

        for p in candidate_paths:
            if p and p.exists():
                try:
                    calib_data = np.load(str(p))
                    if 'mtx_L' in calib_data:
                        self.focale_px = float(calib_data['mtx_L'][0, 0])
                        self.cx = float(calib_data['mtx_L'][0, 2])
                        self.cy = float(calib_data['mtx_L'][1, 2])
                        self.has_calibration = True
                        print(f"  ✓ [DEPTH] Calibrazione caricata da {p.name}: Focale={self.focale_px:.1f}px, cx={self.cx:.1f}, cy={self.cy:.1f}")
                        return
                except Exception as e:
                    print(f"  ⚠️ [DEPTH] Errore lettura calibrazione ({e}): fallback parametri predefiniti.")
                    break

        print(f"  ℹ️ [DEPTH] Calibrazione stereo non trovata. Uso focale configurata: {self.focale_px:.1f}px")

    def _create_matcher(self):
        """Creates tuned StereoSGBM disparity matcher for robot navigation."""
        num_disp = max(16, (self.config.num_disparities // 16) * 16)
        block_size = self.config.block_size
        if block_size % 2 == 0:
            block_size += 1

        matcher = cv2.StereoSGBM_create(
            minDisparity=self.config.min_disparity,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=50,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        return matcher

    def _init_wls_filter(self):
        """Initializes OpenCV ximgproc WLS disparity filter for edge-preserving smoothing."""
        try:
            if hasattr(cv2, 'ximgproc'):
                self.stereo_R = cv2.ximgproc.createRightMatcher(self.stereo_L)
                if hasattr(cv2.ximgproc, 'createDisparityWLSFilter'):
                    self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo_L)
                else:
                    self.wls_filter = cv2.ximgproc.createDisparityFilter(self.stereo_L)

                self.wls_filter.setLambda(8000.0)
                self.wls_filter.setSigmaColor(1.5)
                self.use_wls = True
                print("  ✓ [DEPTH] Filtro Disparità WLS abilitato con successo.")
        except Exception as e:
            self.use_wls = False
            print(f"  ℹ️ [DEPTH] WLS Filter non disponibile ({e}). Uso StereoSGBM standard.")

    def compute_disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray, apply_clahe: bool = True) -> np.ndarray:
        """
        Computes disparity map from left and right images.
        Applies CLAHE and optional WLS filtering.
        """
        left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

        if apply_clahe:
            left_gray = self.clahe.apply(left_gray)
            right_gray = self.clahe.apply(right_gray)

        disp_L = self.stereo_L.compute(left_gray, right_gray)

        if self.use_wls and self.stereo_R and self.wls_filter:
            disp_R = self.stereo_R.compute(right_gray, left_gray)
            try:
                filtered_disp = self.wls_filter.filter(disp_L, left_gray, disparity_map_right=disp_R)
            except TypeError:
                filtered_disp = self.wls_filter.filter(disp_L, left_gray, disparity_right=disp_R)
            disp_float = filtered_disp.astype(np.float32) / 16.0
        else:
            disp_float = disp_L.astype(np.float32) / 16.0

        disp_float[disp_float < 0] = 0
        return disp_float

    def compute_spatial_grid_3d(
        self,
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        proc_width: int = 320,
        proc_height: int = 240
    ) -> Tuple[Dict[str, Tuple[float, float, float]], np.ndarray, Dict[str, Any]]:
        """
        Performs downscaled fast 3D disparity computation and segments the scene into a 9-Zone Grid:
          [ Alto_SX    | Alto_Centro    | Alto_DX    ]
          [ Centro_SX  | Centro_Centro  | Centro_DX  ]
          [ Basso_SX   | Basso_Centro   | Basso_DX   ]

        Returns:
          - grid_3d: {zone_name: (X_cm, Y_cm, Z_cm)}
          - depth_colormap: colorized depth visualization with zone overlays
          - metrics: {"ground_hazard": bool, "critical_front_cm": float, "z_basso": float, "z_centro": float}
        """
        # Downscale for real-time edge processing on RPi5
        h_orig, w_orig = left_bgr.shape[:2]
        scale_x = proc_width / float(w_orig)
        scale_y = proc_height / float(h_orig)

        small_left = cv2.resize(left_bgr, (proc_width, proc_height))
        small_right = cv2.resize(right_bgr, (proc_width, proc_height))

        # Compute disparity on resized frames
        disparity = self.compute_disparity(small_left, small_right, apply_clahe=True)

        h, w = disparity.shape
        f_px = self.focale_px * scale_x
        cx_scaled = self.cx * scale_x
        cy_scaled = self.cy * scale_y

        h_step, w_step = h // 3, w // 3
        nomi_riga = ["Alto", "Centro", "Basso"]
        nomi_col = ["SX", "Centro", "DX"]

        disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

        grid_3d: Dict[str, Tuple[float, float, float]] = {}

        for r in range(3):
            for c in range(3):
                nome_settore = f"{nomi_riga[r]}_{nomi_col[c]}"
                roi_disp = disparity[r * h_step:(r + 1) * h_step, c * w_step:(c + 1) * w_step]
                valid = roi_disp[roi_disp > 1.0]

                if len(valid) > 15:
                    q_low, q_high = np.percentile(valid, 15), np.percentile(valid, 85)
                    filt = valid[(valid >= q_low) & (valid <= q_high)]
                    if len(filt) > 0:
                        med_d = float(np.median(filt))
                        if med_d > 0.1:
                            Z = (f_px * self.baseline_cm) / med_d
                            u_center = (c * w_step) + (w_step / 2.0)
                            v_center = (r * h_step) + (h_step / 2.0)
                            X = ((u_center - cx_scaled) * Z) / f_px
                            Y = ((v_center - cy_scaled) * Z) / f_px

                            if 10.0 <= Z <= 500.0:
                                grid_3d[nome_settore] = (round(X, 1), round(Y, 1), round(Z, 1))

                                # Draw bounding box and label on colormap
                                pt1 = (c * w_step, r * h_step)
                                pt2 = ((c + 1) * w_step, (r + 1) * h_step)
                                cv2.rectangle(depth_colormap, pt1, pt2, (200, 200, 200), 1)
                                cv2.putText(
                                    depth_colormap,
                                    f"{Z:.0f}cm",
                                    (pt1[0] + 5, pt1[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45,
                                    (255, 255, 255),
                                    1
                                )

        # Ground hazard & critical distance evaluation
        z_basso = grid_3d.get("Basso_Centro", (0, 0, 999.0))[2]
        z_centro = grid_3d.get("Centro_Centro", (0, 0, 999.0))[2]
        z_alto = grid_3d.get("Alto_Centro", (0, 0, 999.0))[2]

        ground_hazard = z_basso < self.config.ground_hazard_threshold_cm
        critical_front_cm = min(z_basso, z_centro)

        # Scale colormap back to original eye dimensions if needed
        depth_view = cv2.resize(depth_colormap, (w_orig, h_orig))

        metrics = {
            "ground_hazard": ground_hazard,
            "critical_front_cm": round(critical_front_cm, 1),
            "z_basso": round(z_basso, 1),
            "z_centro": round(z_centro, 1),
            "z_alto": round(z_alto, 1)
        }

        return grid_3d, depth_view, metrics

    def get_color_disparity_map(self, disp_float: np.ndarray) -> np.ndarray:
        """Generates an 8-bit RGB colorized depth visualizer using Jet colormap."""
        max_d = max(1.0, float(self.config.num_disparities))
        norm_disp = np.clip(disp_float / max_d * 255.0, 0, 255).astype(np.uint8)
        color_map = cv2.applyColorMap(norm_disp, cv2.COLORMAP_JET)
        color_map[norm_disp == 0] = 0
        return color_map

    def estimate_bbox_distance_cm(self, disp_float: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[float]:
        """Estimates metric distance (cm) for an object enclosed by (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = bbox
        h, w = disp_float.shape[:2]

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        rw = max(2, (x2 - x1) // 4)
        rh = max(2, (y2 - y1) // 4)

        roi = disp_float[cy - rh:cy + rh, cx - rw:cx + rw]
        valid_disp = roi[roi > 1.0]

        if len(valid_disp) < 5:
            return None

        med_disp = float(np.median(valid_disp))
        if med_disp <= 0.5:
            return None

        distance_cm = (self.focale_px * self.baseline_cm) / med_disp
        return float(np.clip(distance_cm, 10.0, 1500.0))
