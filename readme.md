
# VARI Spectroscopic Engine (Edge CLI)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

A scientifically rigorous, command-line optical processing engine for pre-visual plant stress detection. This tool computes the radiometrically-calibrated Visible Atmospherically Resistant Index (VARI) from standard RGB telemetry. 

Engineered specifically for edge-compute deployment on autonomous platforms (UAVs, agricultural rovers), this architecture discards fragile notebook-style scripting in favor of fault-tolerant execution, strict spatial logging, and hardware-defensive exception handling.

##  Scientific Foundation

VARI is engineered to be minimally sensitive to atmospheric scattering (particularly in the blue spectrum) while maximizing sensitivity to chlorophyll concentrations, making it vastly superior to naive RGB ratios for consumer-grade sensors.

**Mathematical Formulation:**
$$VARI = \frac{Green - Red}{Green + Red - Blue}$$

* **VARI > 0.2**: High chlorophyll (Healthy vegetation)
* **VARI 0.0 - 0.2**: Chlorophyll degradation (Stressed vegetation)
* **VARI < 0.0**: Soil, shadows, or non-vegetation

###  The Radiometric Calibration Imperative
Unlike standard implementations that erroneously calculate indices directly on gamma-compressed sRGB pixels, this engine performs **linearization** of the sensor data prior to index calculation. Furthermore, it implements dynamic white-balance correction in linear space using a neutral gray reference to account for varying solar illuminants (e.g., D65 daylight).

##  Key Engineering Features

* **Radiometric Accuracy**: sRGB to Linear RGB transformation for mathematically valid reflectance ratios.
* **Numerical Stability**: 32-bit float precision pipelines to eliminate integer overflow/underflow artifacts common in 8-bit processing.
* **Sensor Calibration**: Built-in neutral-reference white balancing to neutralize ISP (Image Signal Processor) color shifting.
* **Edge-Optimized**: Highly vectorized NumPy operations achieving sub-50ms processing times at 1080p, making it viable for real-time video telemetry on constrained hardware (e.g., Raspberry Pi, Jetson).
* **Perceptually Uniform Mapping**: Utilizes modern colormaps (`cv2.COLORMAP_VIRIDIS`) to prevent artificial data boundary artifacts.


##  Architectural Superiority

* **Linearized Radiometry**: Automatically decodes sRGB gamma compression prior to index calculation, ensuring mathematically valid reflectance ratios.
* **Fault-Tolerant Ingestion**: Custom exception classes (`RadiometricCalibrationError`, `VARIPipelineError`) prevent corrupted sensor frames from crashing the host system.
* **Telemetry Logging**: Abandons standard print outputs for standard Python `logging`, allowing seamless integration with broader system diagnostic streams.
* **CLI Driven**: Fully controllable via command-line arguments for immediate integration into bash scripts, cron jobs, or containerized environments.

##  Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/Gaurav23-bits/vari.git](https://github.com/Gaurav23-bits/vari.git)
cd vari
pip install -r requirements.txt
