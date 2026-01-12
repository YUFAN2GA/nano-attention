import cv2
import torch
import numpy as np
import time
from collections import deque

def get_visual_model(model_name='dinov2'):
    """
    Load Visual Backbone (DINOv3 or DINOv2).
    DINOv3 is preferred but might require specific access.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Running on device: {device}")
    
    model = None
    try:
        # Attempt to load DINOv3 (Hypothetical loading if available in hub by now, or placeholder)
        # Note: If DINOv3 is not publicly available on simple torch.hub, we fallback or use v2
        print("⏳ Loading Visual Backbone (DINO)...")
        # For now, we stick to DINOv2 as a robust fallback until specific DINOv3 hub repo is confirmed
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        print("✅ Loaded DINOv2 (SOTA fallback for DINOv3)")
    except Exception as e:
        print(f"⚠️ Could not load model: {e}")
        return None, device

    if model:
        model.to(device)
        model.eval()
    
    return model, device

def smooth_path(trajectory, smoothing_window=5):
    """Simple moving average smoothing for the path"""
    if len(trajectory) < smoothing_window:
        return trajectory
    
    smoothed = []
    # Just smooth the recent points
    # (A real FSD system would use Kalman Filter, this is visual MVP)
    for i in range(len(trajectory)):
        start = max(0, i - smoothing_window)
        end = i + 1
        chunk = trajectory[start:end]
        avg_dx = sum(p[0] for p in chunk) / len(chunk)
        avg_dy = sum(p[1] for p in chunk) / len(chunk)
        smoothed.append((avg_dx, avg_dy))
    return smoothed

def main():
    print("🚗 Nano FSD - Laptop Edition (DINOv3 Plan)")
    print("🎯 Objective: Real-time Path Projection & Scene Understanding")
    
    # 1. Initialize Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam. Please check your camera.")
        return

    # 2. Visual Model (Background/Optional for MVP path projection consistency)
    # Loading this might take time, we do it before loop
    model, device = get_visual_model('dinov3')

    # 3. Motion & State
    prev_gray = None
    # Store path segments (dx, dy)
    # We use a deque for a rolling window of the last N frames of motion
    path_history = deque(maxlen=30) 
    
    # Optical Flow Parameters
    feature_params = dict(maxCorners=200, qualityLevel=0.2, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    print("\n✅ System Ready! Press 'q' to quit.")
    print("   Note: The GREEN line is your predicted path based on camera motion.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- MOTION ESTIMATION (Visual Odometry) ---
        dx, dy = 0, 0
        if prev_gray is not None:
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
                if p1 is not None:
                    good_new = p1[st==1]
                    good_old = p0[st==1]
                    
                    if len(good_new) > 0:
                        # Vector: Old -> New
                        flow_vectors = good_new - good_old
                        # Camera motion is opposite to flow
                        # If flow is LEFT (negative dx), Camera moved RIGHT (positive dx)
                        # But for "Projected Path", if we turn Right, we want path to curve Right?
                        # Actually, if we turn Right, the image moves Left. 
                        # We want to show where we are GOING.
                        avg_flow = np.mean(flow_vectors, axis=0)
                        dx = -avg_flow[0] # Camera Motion X
                        dy = -avg_flow[1] # Camera Motion Y (Forward/Back mostly, or Pitch)

        # Update Path History
        # We dampen the values to make it look like a smooth path, not jittery raw sensor data
        path_history.append((dx, dy))
        
        # Smooth the trajectory for display
        smooth_traj = smooth_path(list(path_history))
        
        # --- VISUALIZATION: PROJECTED PATH ---
        # Start from bottom center
        h, w = frame.shape[:2]
        start_pt = (w // 2, h)
        
        points = [start_pt]
        curr_x, curr_y = start_pt
        
        # Accumulate path
        # A simple model: velocity * time_steps forward
        # We project the stored motion history 'forward' to predict future curve
        # (Assuming constant curvature based on recent history)
        
        # Logic: If we are turning left recently, the path should curve left.
        # Calculation: Sum of recent dx determines curvature
        if len(smooth_traj) > 0:
            avg_dx = np.mean([p[0] for p in smooth_traj])
            avg_dy = np.mean([p[1] for p in smooth_traj]) # usually this is speed??
            
            # Artificial forward speed for visualization (we always move 'forward' in time/space)
            forward_speed = 15 
            
            # Predict 10 steps into future
            for i in range(1, 15):
                # The curvature adds up
                curr_x += avg_dx * (i * 0.5) * 2 # Gain to make it visible
                curr_y -= forward_speed # Always move up (away from bottom)
                
                points.append((int(curr_x), int(curr_y)))
        
        # Draw Path
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(frame_display, points[i-1], points[i], (0, 255, 0), 4)
                
        # Status Text
        status = "Straight"
        if len(smooth_traj) > 0:
            avg_dx = np.mean([p[0] for p in smooth_traj])
            if avg_dx > 1.0: status = "Turning Right"
            elif avg_dx < -1.0: status = "Turning Left"
            
        cv2.putText(frame_display, f"Motion: {status}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Note regarding DINO:
        # In this MVP phase, we are NOT running DINO inference per frame to save FPS
        # and prioritize the "Green Path" feel. 
        # Feature extraction usually happens here if 'Brain View' is enabled.
        
        cv2.imshow('Nano FSD - Path Projection', frame_display)
        prev_gray = gray.copy()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
