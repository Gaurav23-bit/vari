""""
VARI Spectroscopic Engine - Precision Agriculture Computer Vision
==================================================================

Author: Computer Vision Engineer (Remote Sensing Specialist)
Purpose: Radiometrically-calibrated Visible Atmospherically Resistant Index (VARI)
         calculation for pre-visual plant stress detection from smartphone RGB imagery.

Scientific Foundation:
---------------------
VARI is designed to be minimally sensitive to atmospheric effects and maximizes
sensitivity to vegetation while minimizing sensitivity to soil brightness.

Formula: VARI = (Green - Red) / (Green + Red - Blue)
Range: -1.0 to 1.0
- VARI > 0.2  : Healthy vegetation (high chlorophyll)
- VARI 0-0.2  : Stressed vegetation
- VARI < 0    : Soil/non-vegetation

Key Engineering Features:
-------------------------
1. 32-bit float precision to prevent overflow/underflow
2. White balance normalization for sensor calibration
3. Noise reduction via Gaussian filtering
4. Vectorized NumPy operations for <50ms processing @ 1080p
5. Modular architecture for real-time mobile integration
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import time


class VARIEngine:
    """
    Radiometric VARI Processing Engine
    """
    
    def __init__(self, 
                 blur_kernel: int = 5,
                 colormap: int = cv2.COLORMAP_JET,
                 enable_timing: bool = False):
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.colormap = colormap
        self.enable_timing = enable_timing
        
        self.wb_r = 1.0
        self.wb_g = 1.0
        self.wb_b = 1.0
        
        self.last_process_time = 0.0
        
    def calibrate_white_balance(self, 
                                image: np.ndarray,
                                roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, float]:
        if roi is None:
            h, w = image.shape[:2]
            cx, cy = w // 2, h // 2
            roi_w, roi_h = w // 10, h // 10
            roi = (cx - roi_w // 2, cy - roi_h // 2, roi_w, roi_h)
        
        x, y, rw, rh = roi
        reference_patch = image[y:y+rh, x:x+rw]
        
        # Calculate mean values in BGR order
        mean_bgr = cv2.mean(reference_patch)[:3]
        mean_b, mean_g, mean_r = mean_bgr
        
        self.wb_r = mean_g / (mean_r + 1e-6)
        self.wb_g = 1.0
        self.wb_b = mean_g / (mean_b + 1e-6)
        
        return {
            'wb_r': self.wb_r,
            'wb_g': self.wb_g,
            'wb_b': self.wb_b,
            'reference_rgb': (mean_r, mean_g, mean_b),
            'roi': roi
        }
    
    def _apply_white_balance(self, image_float: np.ndarray) -> np.ndarray:
        balanced = image_float.copy()
        balanced[:, :, 0] *= self.wb_b
        balanced[:, :, 1] *= self.wb_g
        balanced[:, :, 2] *= self.wb_r
        
        balanced = np.clip(balanced, 0.0, 1.0)
        
        return balanced
    
    def calculate_vari(self, 
                      image: np.ndarray,
                      apply_white_balance: bool = True,
                      return_raw: bool = False) -> np.ndarray:
        t_start = time.perf_counter()
        
        image_float = image.astype(np.float32) / 255.0
        
        if apply_white_balance and (self.wb_r != 1.0 or self.wb_b != 1.0):
            image_float = self._apply_white_balance(image_float)
        
        if self.blur_kernel > 1:
            image_float = cv2.GaussianBlur(image_float, 
                                          (self.blur_kernel, self.blur_kernel), 
                                          0)
        
        B = image_float[:, :, 0]
        G = image_float[:, :, 1]
        R = image_float[:, :, 2]
        
        # Add epsilon to prevent division by zero
        epsilon = 1e-6
        
        numerator = G - R
        denominator = G + R - B + epsilon
        
        vari = numerator / denominator
        
        # Handle edge cases and invalid numbers
        vari = np.clip(vari, -1.0, 1.0)
        vari = np.nan_to_num(vari, nan=-1.0, posinf=-1.0, neginf=-1.0)
        
        if self.enable_timing:
            self.last_process_time = (time.perf_counter() - t_start) * 1000
            print(f"VARI calculation: {self.last_process_time:.2f} ms")
        
        if return_raw:
            return vari
        else:
            vari_norm = ((vari + 1.0) / 2.0 * 255).astype(np.uint8)
            return vari_norm
    
    def generate_stress_map(self, 
                           image: np.ndarray,
                           apply_white_balance: bool = True,
                           mask_nonveg: bool = True,
                           nonveg_threshold: float = 0.0) -> np.ndarray:
        vari_raw = self.calculate_vari(image, 
                                       apply_white_balance=apply_white_balance,
                                       return_raw=True)
        
        if mask_nonveg:
            veg_mask = vari_raw >= nonveg_threshold
        else:
            veg_mask = np.ones_like(vari_raw, dtype=bool)
        
        vari_vis = ((vari_raw + 1.0) / 2.0 * 255).astype(np.uint8)
        
        heatmap = cv2.applyColorMap(vari_vis, self.colormap)
        
        if mask_nonveg:
            heatmap[~veg_mask] = [0, 0, 0]
        
        return heatmap
    
    def create_split_view(self,
                         image: np.ndarray,
                         apply_white_balance: bool = True) -> np.ndarray:
        stress_map = self.generate_stress_map(image, apply_white_balance)
        
        h, w = image.shape[:2]
        stress_map_resized = cv2.resize(stress_map, (w, h))
        
        combined = np.hstack([image, stress_map_resized])
        
        return combined
    
    def analyze_stress_statistics(self, 
                                  image: np.ndarray,
                                  apply_white_balance: bool = True) -> Dict:
        vari_raw = self.calculate_vari(image, apply_white_balance, return_raw=True)
        
        healthy_mask = vari_raw > 0.2
        stressed_mask = (vari_raw >= 0.0) & (vari_raw <= 0.2)
        nonveg_mask = vari_raw < 0.0
        
        total_pixels = vari_raw.size
        
        veg_vari = vari_raw[vari_raw >= 0.0]
        
        return {
            'mean_vari': float(np.mean(veg_vari)) if veg_vari.size > 0 else 0.0,
            'std_vari': float(np.std(veg_vari)) if veg_vari.size > 0 else 0.0,
            'median_vari': float(np.median(veg_vari)) if veg_vari.size > 0 else 0.0,
            'healthy_pct': float(np.sum(healthy_mask) / total_pixels * 100),
            'stressed_pct': float(np.sum(stressed_mask) / total_pixels * 100),
            'nonveg_pct': float(np.sum(nonveg_mask) / total_pixels * 100),
            'processing_time_ms': self.last_process_time
        }


def demo_real_time_processing():
    print("=" * 70)
    print("VARI Engine - Real-Time Processing Demonstration")
    print("=" * 70)
    
    height, width = 1080, 1920
    
    test_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for x in range(width):
        r_val = int(50 + (x / width) * 150)
        g_val = 120
        b_val = 40
        
        test_image[:, x] = [b_val, g_val, r_val]
    
    noise = np.random.normal(0, 5, test_image.shape).astype(np.int16)
    test_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    engine = VARIEngine(blur_kernel=5, 
                       colormap=cv2.COLORMAP_VIRIDIS,
                       enable_timing=True)
    
    print("\nStep 1: White Balance Calibration")
    print("-" * 70)
    test_image[0:100, 0:100] = [200, 200, 200]
    
    calibration = engine.calibrate_white_balance(test_image, roi=(10, 10, 80, 80))
    print(f"  White Balance Factors:")
    print(f"    Red:   {calibration['wb_r']:.4f}")
    print(f"    Green: {calibration['wb_g']:.4f}")
    print(f"    Blue:  {calibration['wb_b']:.4f}")
    print(f"  Reference RGB: {calibration['reference_rgb']}")
    
    print("\nStep 2: VARI Calculation (Uncalibrated vs Calibrated)")
    print("-" * 70)
    
    vari_uncal = engine.calculate_vari(test_image, apply_white_balance=False)
    time_uncal = engine.last_process_time
    
    vari_cal = engine.calculate_vari(test_image, apply_white_balance=True)
    time_cal = engine.last_process_time
    
    print(f"  Uncalibrated processing: {time_uncal:.2f} ms")
    print(f"  Calibrated processing:   {time_cal:.2f} ms")
    print(f"  Real-time capable (1080p @ 60fps requires <16.67ms): " +
          ("✓ YES" if time_cal < 16.67 else "✗ NO (but <50ms target: " + 
           ("✓ YES" if time_cal < 50 else "✗ NO") + ")"))
    
    print("\nStep 3: Vegetation Stress Analysis")
    print("-" * 70)
    
    stats = engine.analyze_stress_statistics(test_image, apply_white_balance=True)
    print(f"  Mean VARI:       {stats['mean_vari']:.4f}")
    print(f"  Std Dev:         {stats['std_vari']:.4f}")
    print(f"  Median VARI:     {stats['median_vari']:.4f}")
    print(f"  ")
    print(f"  Healthy (>0.2):  {stats['healthy_pct']:.1f}%")
    print(f"  Stressed (0-0.2): {stats['stressed_pct']:.1f}%")
    print(f"  Non-veg (<0):    {stats['nonveg_pct']:.1f}%")
    
    print("\n" + "=" * 70)
    print("Demonstration Complete")
    print("=" * 70)
    
    return engine, test_image


if __name__ == "__main__":
    engine, test_image = demo_real_time_processing()