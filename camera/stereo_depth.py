#!/usr/bin/env python3
"""
M.A.R. — Stereo Disparity & 3D Depth Estimation Module
Calculates stereo disparity maps (StereoSGBM / StereoBM) and converts
pixel bounding boxes into metric depth estimations (Z in meters).
"""

from typing import Optional, Tuple
import cv2
import numpy as np

try:
    from ..config import DepthConfig
except (ImportError, ValueError):
    from config import DepthConfig


class StereoDepthEstimator:
    """
    Computes real-time stereo disparity and 3D depth maps from synchronized stereo pairs.
    Formula: Z = (f * B) / disparity
      - f: focal length in pixels
      - B: baseline distance between left and right lenses (in cm or meters)
    """

    def __init__(self, config: DepthConfig):
        self.config = config
        self.matcher = self._create_matcher()

    def _create_matcher(self):
        """Creates tuned StereoSGBM disparity matcher for outdoor & indoor robot environments."""
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
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        return matcher

    def compute_disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        """
        Computes 16-bit raw disparity map from left and right images.
        """
        left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

        disp16 = self.matcher.compute(left_gray, right_gray)
        # Disparity values are scaled by 16 by OpenCV
        disp_float = disp16.astype(np.float32) / 16.0
        # Filter out invalid / negative disparity
        disp_float[disp_float < 0] = 0
        return disp_float

    def get_color_disparity_map(self, disp_float: np.ndarray) -> np.ndarray:
        """
        Generates an 8-bit RGB colorized depth visualizer using Jet colormap.
        """
        max_d = max(1.0, float(self.config.num_disparities))
        norm_disp = np.clip(disp_float / max_d * 255.0, 0, 255).astype(np.uint8)
        color_map = cv2.applyColorMap(norm_disp, cv2.COLORMAP_TURBO)
        # Set invalid areas (zero disparity) to black
        color_map[norm_disp == 0] = 0
        return color_map

    def estimate_bbox_distance_cm(self, disp_float: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[float]:
        """
        Estimates the metric distance (cm) of an object enclosed by a bounding box (x1, y1, x2, y2).
        Uses the median disparity in the central 50% ROI of the bounding box to avoid edge noise.
        """
        x1, y1, x2, y2 = bbox
        h, w = disp_float.shape[:2]

        # Clamp coordinates
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        # Sample central 50% region of the box
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

        # Z = (f * B) / d
        distance_cm = (self.config.focal_length_px * self.config.baseline_cm) / med_disp
        # Clamp to realistic robotics range (10 cm to 1500 cm)
        return float(np.clip(distance_cm, 10.0, 1500.0))
