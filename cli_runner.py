"""
VARI Spectroscopic Engine - Core Processing Module
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict
import time

# Configure professional telemetry logging
logger = logging.getLogger("VARI_Telematics")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RadiometricCalibrationError(Exception):
    """Raised when optical sensor white balance fails physical constraints."""
    pass


class VARIPipelineError(Exception):
    """Raised when the input array violates processing requirements."""
    pass


class VARIEngine:
    """
    Radiometric VARI Processing Engine
    """
    
    def __init__(self, blur_kernel: int = 5, colormap: int = cv2.COLORMAP_VIRIDIS):
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.colormap = colormap
        
        self.wb_r = 1.0
        self.wb_g = 1.0
        self.wb_b = 1.0
        self.last_process_time = 0.0
        
    def calibrate_white_balance(self, 
                                image: np.ndarray,
                                roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, float]:
        if image is None or image.size == 0:
            logger.critical("Sensor buffer empty. Cannot perform radiometric calibration.")
            raise RadiometricCalibrationError("Null image array received from payload.")
            
        if image.dtype != np.uint8:
            logger.warning(f"Expected 8-bit sensor depth, received {image.dtype}. Normalization may fail.")

        if roi is None:
            h, w = image.shape[:2]
            cx, cy = w // 2, h // 2
            roi_w, roi_h = w // 10, h // 10
            roi = (cx - roi_w // 2, cy - roi_h // 2, roi_w, roi_h)
        
        x, y, rw, rh = roi
        reference_patch = image[y:y+rh, x:x+rw]
        
        if reference_patch.size == 0:
            raise RadiometricCalibrationError("Calibration ROI is outside image bounds.")

        mean_b, mean_g, mean_r = cv2.mean(reference_patch)[:3]
        
        self.wb_r = mean_g / (mean_r + 1e-6)
        self.wb_g = 1.0
        self.wb_b = mean_g / (mean_b + 1e-6)
        
        # Sanity check the calculated factors to detect severe sensor anomalies
        if not (0.1 < self.wb_r < 10.0) or not (0.1 < self.wb_b < 10.0):
            logger.error(f"Calibration factors outside physical bounds: R:{self.wb_r:.2f}, B:{self.wb_b:.2f}")
            raise RadiometricCalibrationError("Severe optical anomaly detected in calibration ROI.")
            
        logger.info(f"Radiometric baseline established. R-gain: {self.wb_r:.3f}, B-gain: {self.wb_b:.3f}")
        
        return {'wb_r': self.wb_r, 'wb_g': self.wb_g, 'wb_b': self.wb_b, 'roi': roi}
    
    def _apply_white_balance(self, image_float: np.ndarray) -> np.ndarray:
        balanced = image_float.copy()
        balanced[:, :, 0] *= self.wb_b
        balanced[:, :, 1] *= self.wb_g
        balanced[:, :, 2] *= self.wb_r
        return np.clip(balanced, 0.0, 1.0)
    
    def calculate_vari(self, 
                      image: np.ndarray,
                      apply_white_balance: bool = True,
                      return_raw: bool = False) -> np.ndarray:
        
        if image is None:
            raise VARIPipelineError("Received NoneType instead of image array.")
            
        t_start = time.perf_counter()
        
        # Linearize sRGB to Linear RGB
        image_float = image.astype(np.float32) / 255.0
        linear_rgb = np.where(image_float <= 0.04045, 
                              image_float / 12.92, 
                              np.power((image_float + 0.055) / 1.055, 2.4))
        
        if apply_white_balance and (self.wb_r != 1.0 or self.wb_b != 1.0):
            linear_rgb = self._apply_white_balance(linear_rgb)
        
        if self.blur_kernel > 1:
            linear_rgb = cv2.GaussianBlur(linear_rgb, (self.blur_kernel, self.blur_kernel), 0)
        
        B = linear_rgb[:, :, 0]
        G = linear_rgb[:, :, 1]
        R = linear_rgb[:, :, 2]
        
        epsilon = 1e-6
        vari = np.clip((G - R) / (G + R - B + epsilon), -1.0, 1.0)
        vari = np.nan_to_num(vari, nan=-1.0, posinf=-1.0, neginf=-1.0)
        
        self.last_process_time = (time.perf_counter() - t_start) * 1000
        logger.debug(f"Hardware execution time: {self.last_process_time:.2f} ms")
        
        if return_raw:
            return vari
        return ((vari + 1.0) / 2.0 * 255).astype(np.uint8)
    
    def generate_stress_map(self, image: np.ndarray, apply_white_balance: bool = True) -> np.ndarray:
        vari_raw = self.calculate_vari(image, apply_white_balance=apply_white_balance, return_raw=True)
        veg_mask = vari_raw >= 0.0
        
        vari_vis = ((vari_raw + 1.0) / 2.0 * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(vari_vis, self.colormap)
        heatmap[~veg_mask] = [0, 0, 0]
        
        return heatmap

    def analyze_stress_statistics(self, image: np.ndarray, apply_white_balance: bool = True) -> Dict:
        vari_raw = self.calculate_vari(image, apply_white_balance, return_raw=True)
        total_pixels = vari_raw.size
        
        healthy_mask = vari_raw > 0.2
        stressed_mask = (vari_raw >= 0.0) & (vari_raw <= 0.2)
        nonveg_mask = vari_raw < 0.0
        
        veg_vari = vari_raw[vari_raw >= 0.0]
        
        return {
            'mean_vari': float(np.mean(veg_vari)) if veg_vari.size > 0 else 0.0,
            'std_vari': float(np.std(veg_vari)) if veg_vari.size > 0 else 0.0,
            'healthy_pct': float(np.sum(healthy_mask) / total_pixels * 100),
            'stressed_pct': float(np.sum(stressed_mask) / total_pixels * 100),
            'nonveg_pct': float(np.sum(nonveg_mask) / total_pixels * 100),
            'execution_ms': self.last_process_time
        }