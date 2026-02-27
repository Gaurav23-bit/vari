import streamlit as st
import cv2
import numpy as np
from vari_engine import VARIEngine  # assume your engine file is vari_engine.py
from PIL import Image
import tempfile

st.set_page_config(layout="wide", page_title="VARI Precision Agriculture")

st.title("🌱 VARI Spectroscopic Engine")
st.markdown("Visible Atmospherically Resistant Index (Plant Stress Detection)")

# Sidebar controls
st.sidebar.header("Processing Settings")

blur_kernel = st.sidebar.slider("Gaussian Blur Kernel", 1, 15, 5, step=2)
apply_wb = st.sidebar.checkbox("Apply White Balance", value=True)
mask_nonveg = st.sidebar.checkbox("Mask Non-Vegetation", value=True)
nonveg_threshold = st.sidebar.slider("Non-Veg Threshold", -1.0, 1.0, 0.0, step=0.01)

engine = VARIEngine(
    blur_kernel=blur_kernel,
    colormap=cv2.COLORMAP_VIRIDIS,
    enable_timing=True
)

# Image input options
input_mode = st.radio("Select Input Source", ["Upload Image", "Use Webcam"])

image = None

if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload RGB Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = np.array(Image.open(uploaded_file))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

elif input_mode == "Use Webcam":
    camera_image = st.camera_input("Take a Picture")
    if camera_image:
        image = np.array(Image.open(camera_image))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

if image is not None:

    st.subheader("White Balance Calibration")

    if st.button("Auto Calibrate (Center ROI)"):
        calibration = engine.calibrate_white_balance(image)
        st.success("White Balance Calibrated")
        st.write(calibration)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

    # Generate stress map
    stress_map = engine.generate_stress_map(
        image,
        apply_white_balance=apply_wb,
        mask_nonveg=mask_nonveg,
        nonveg_threshold=nonveg_threshold
    )

    stats = engine.analyze_stress_statistics(
        image,
        apply_white_balance=apply_wb
    )

    with col2:
        st.subheader("VARI Stress Map")
        st.image(cv2.cvtColor(stress_map, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.subheader("Vegetation Statistics")

    stat_col1, stat_col2, stat_col3 = st.columns(3)

    stat_col1.metric("Mean VARI", f"{stats['mean_vari']:.4f}")
    stat_col2.metric("Healthy % (>0.2)", f"{stats['healthy_pct']:.1f}%")
    stat_col3.metric("Stressed % (0–0.2)", f"{stats['stressed_pct']:.1f}%")

    st.write(f"Non-Vegetation %: {stats['nonveg_pct']:.1f}%")
    st.write(f"Processing Time: {stats['processing_time_ms']:.2f} ms")

    st.subheader("Split View")
    split_view = engine.create_split_view(image, apply_white_balance=apply_wb)
    st.image(cv2.cvtColor(split_view, cv2.COLOR_BGR2RGB), use_column_width=True)
