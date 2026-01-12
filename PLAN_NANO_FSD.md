# Nano FSD (Laptop Edition) Implementation Plan

## 🎯 Objective
Run a real-time "FSD-like" visualization on a laptop using a single RGB camera. The system will visualize the AI's understanding of the scene and project a predicted path based on camera movement.

## 🧠 Core Technology
1.  **Visual Backbone**: **Meta DINOv3** (ViT-Small/Base).
    -   *Why?* The latest iteration (released Aug 2025) offers SOTA performance for "unsupervised visual features", with improved understanding of geometry and dense features compared to v2.
    -   *Usage*: We will use it to extract dense features for scene understanding.
2.  **Trajectory Estimation**: **Optical Flow (Lucas-Kanade)**.
    -   *Why?* Since we don't have a trained regression head for driving control, we will use visual odometry (via optical flow) to estimate the camera's ego-motion and project a "predicted path" curve.

## 🛠️ Components

### 1. `LaptopFSD` Class
- **Input**: Webcam feed (OpenCV).
- **Model**: DINOv3 (loaded via Torch Hub / Hugging Face).
- **Motion Algo**: Sparse Optical Flow to calculate simple trajectory (turning left/right/straight).

### 2. Visualization UI (MVP)
- **Main View**: Real-time camera feed with a **Green Projected Path** overlay (dynamic spline).
- **Status**: Simple text overlay showing estimated motion (e.g. "Turning Left").

### 3. Advanced Visualization (Phase 2)
- **Brain View**: A heatmap overlay showing DINOv3's attention (Principal Component Analysis of patch features), revealing how the AI parses the scene structure.
- **Why Phase 2?**: To prioritize the "path projection" fluidity first, as per user request.

## 📋 Requirements
- `torch` (Already installed)
- `opencv-python` (Need to install)
- `numpy`
- `scikit-learn` (Optional, for PCA visualization in Phase 2)
- `timm` or specific repo for DINOv3 loading

## 🚀 Workflow
1.  Install dependencies.
2.  Write `fsd_laptop_demo.py` with basic camera & path projection.
3.  Implement DINOv3 feature extraction (background thread or optimized).
4.  (Later) Add Attention/PCA visualization.
5.  Run and walk around with the laptop!
