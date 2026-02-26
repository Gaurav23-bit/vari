import cv2
import numpy as np
import time
from typing import List, Tuple
from scipy.ndimage import gaussian_filter

# Assuming VARIEngine is imported or in the same file
from vari_engine import VARIEngine 

def create_realistic_test_image(scenario: str = "healthy", resolution: Tuple[int, int] = (1080, 1920)) -> np.ndarray:
    """Generates synthetic crop images for different health scenarios."""
    h, w = resolution
    image = np.zeros((h, w, 3), dtype=np.uint8)
    y, x = np.ogrid[0:h, 0:w]
    
    # Define base BGR colors for vegetation and stress
    if scenario == "healthy":
        R, G, B = 50, 120, 40
    elif scenario == "stressed":
        center_x, center_y = w // 2, h // 2
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        factor = np.clip(1 - dist / (min(w, h) // 3), 0, 1)
        R, G, B = (50 + factor * 80), (120 - factor * 40), 40
    elif scenario == "patchy":
        noise = cv2.resize(np.random.rand(h // 10, w // 10), (w, h))
        smooth = (noise - noise.min()) / (noise.max() - noise.min())
        R, G, B = (50 + smooth * 80), (120 - smooth * 40), 40
    elif scenario == "edge_stress":
        edge_dist = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
        factor = np.clip(edge_dist / (min(w, h) * 0.15), 0, 1)
        R, G, B = (50 + (1 - factor) * 70), (120 - (1 - factor) * 35), 40

    image[:, :, 0], image[:, :, 1], image[:, :, 2] = B, G, R
    
    # Add camera noise and a gray calibration card in the corner
    noise = np.random.normal(0, 3, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    image[10:110, 10:110] = [200, 200, 200] 
    return image

def run_field_workflow():
    """Demonstrates a standard field analysis cycle."""
    engine = VARIEngine(blur_kernel=5, colormap=cv2.COLORMAP_VIRIDIS)
    image = create_realistic_test_image("stressed")
    
    # 1. Calibrate using the gray card
    engine.calibrate_white_balance(image, roi=(10, 10, 100, 100))
    
    # 2. Analyze health metrics
    stats = engine.analyze_stress_statistics(image)
    print(f"Health Check: {stats['healthy_pct']:.1f}% Healthy, {stats['stressed_pct']:.1f}% Stressed")
    
    # 3. Visualization
    view = engine.create_split_view(image)
    return view

if __name__ == "__main__":
    run_field_workflow()