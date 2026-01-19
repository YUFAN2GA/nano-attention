# Nano FSD - Iteration 1: Vison-Based Path Planning

## 🎯 Objective
Upgrade the system from a "Motion Visualization" demo (optical flow) to a **Vision-Based Navigation** system.
The system will analyze a video input of a driving vehicle, identify the drivable road surface, detect valid gaps between obstacles, and generate a safe trajectory path.

## 🛠 Features
1.  **Video Input Support**: Ability to process pre-recorded driving footage (MP4/AVI) instead of just Webcam.
2.  **Visual Backbone (DINO)**: 
    - Utilize **Meta DINOv2** (as the robust implementation of DINOv3 concepts) for semantic understanding.
    - DINO's "Attention" and "Feature Similarity" properties allow us to segment 'road' vs 'non-road' without training a specific supervised model.
3.  **Perception Layer (The "Brain")**:
    - **Road Segmentation**: Compare image features against a "Road Prototype" (features from the bottom of the screen).
    - **Obstacle Segregation**: Areas with low similarity to the road prototype are treated as obstacles.
4.  **Path Planning**:
    - Find the "Furthest Drivable Point" (The Horizon).
    - Generate a smooth curve (Bezier or Spline) from the car's current position to the target, avoiding low-confidence areas.
5.  **Visualization**:
    - **Heatmap Overlay**: Show the system's "confidence" of where the road is (Red=Obstacle, Blue=Road).
    - **Trajectory**: Draw the computed green path.

## 📅 Development Steps

### Step 1: Core Framework Update
- Create `fsd_video_agent.py`.
- Implement `VideoLoader` class for handling file streams.
- Integrate `torch.hub` loading for `dinov2_vits14` (Smallest/Fastest for laptop).

### Step 2: DINO Feature Extraction
- Implement a pipeline to:
    1. Resize video frame to a DINO-friendly resolution (e.g., multiple of 14, like 448x448).
    2. Run inference to get **Patch Tokens**.
    3. Reshape separate tokens back into a 2D spatial feature map.

### Step 3: Semantic Road Finder
- **Logic**: 
    - Assume the bottom center 10% of the image is "Road".
    - Calculate the *Cosine Similarity* between every patch in the image and this "Road Reference".
    - Result: A "Drivability Map".

### Step 4: Trajectory Generation
- Algorithm:
    - Threshold the Drivability Map to get a binary mask.
    - Find the connected component starting from the bottom.
    - Calculate the centroid of the furthest row in this component.
    - Draw a quadratic Bezier curve from (Bottom-Center) to (Target-Point).

### Step 5: Integration & Optimization
- Combine all parts.
- Optimize for FPS (maybe process DINO every 2-3 frames and interpolate).

---

## 📦 Requirements
- `torch`, `torchvision` (already installed)
- `opencv-python`
- `numpy`
- (Optional) `yt-dlp` if we need to download a test video.
