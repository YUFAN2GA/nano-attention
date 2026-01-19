import cv2
import torch
import numpy as np
import time
from collections import deque
import argparse

def get_visual_model(model_name='dinov2'):
    """
    Load Visual Backbone (DINOv3 or DINOv2).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Running on device: {device}")
    
    model = None
    try:
        print("⏳ Loading Visual Backbone (DINO)...")
        # For now, we stick to DINOv2
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        print("✅ Loaded DINOv2")
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
    for i in range(len(trajectory)):
        start = max(0, i - smoothing_window)
        end = i + 1
        chunk = trajectory[start:end]
        avg_x = sum(p[0] for p in chunk) / len(chunk)
        avg_y = sum(p[1] for p in chunk) / len(chunk)
        smoothed.append((int(avg_x), int(avg_y)))
    return smoothed

def main():
    print("🚗 Nano FSD - Laptop Edition (Road Segmentation Mode)")
    print("🎯 Objective: Find Drivable Area using DINOv2 Semantic Features")
    
    # 1. Initialize Camera
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help='Path to video file', default=None)
    args = parser.parse_args()
    
    source = args.video if args.video else 0
    if isinstance(source, str) and source.isdigit():
        source = int(source)
        
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Cannot open source: {source}")
        return

    # 2. Visual Model
    model, device = get_visual_model('dinov3')

    # Internal state for perception
    target_path = [] # List of (x,y) tuples
    frame_count = 0
    
    # DINO Preprocessing params
    # Fixed input size for DINO (must be divisible by 14)
    # Using 336x336 for speed/balance
    infer_w, infer_h = 336, 336 
    
    # Hard Semantic Priors
    roi_mask = None 

    print("\n✅ System Ready! Displaying:")
    print("   🟢 GREEN Region: DINOv2 Road Segmentation (High Confidence)")
    print("   🟢 GREEN Line  : Predicted Drive Path")

    while True:
        ret, frame = cap.read()
        if not ret:
            # Auto replay for video file
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Resize for display consistency if needed, but we used original frame for display
        frame_display = frame.copy()
        h, w = frame.shape[:2]
        
        # Resize for DINO
        img_small = cv2.resize(frame, (infer_w, infer_h))
        
        # 1. PREPROCESS
        img_norm = img_small.astype(np.float32) / 255.0
        img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        t_img = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        
        # 2. INFERENCE (DINOv2)
        # We run this every 3 frames
        if frame_count % 3 == 0 or not target_path:
            with torch.no_grad():
                features_dict = model.forward_features(t_img)
                features = features_dict["x_norm_patchtokens"].squeeze(0) # (N, Dim)
                
            # Grid dimensions
            grid_h = infer_h // 14
            grid_w = infer_w // 14
            
            # --- ROBUST SEGMENTATION LOGIC ---
            
            # A. ACQUIRE ANCHORS
            # Road Anchor (Positive): Lower Center region
            # We pick a point at 85% height, center width
            idx_road = int(grid_h * 0.85) * grid_w + (grid_w // 2)
            road_vec = features[idx_road].unsqueeze(0)
            
            # Sky/Background Anchor (Negative): Top Center region
            # We assume top 10% is definitely not road
            idx_sky = int(grid_h * 0.1) * grid_w + (grid_w // 2)
            sky_vec = features[idx_sky].unsqueeze(0)
            
            # Normalize
            features_n = torch.nn.functional.normalize(features, dim=1)
            road_vec_n = torch.nn.functional.normalize(road_vec, dim=1)
            sky_vec_n  = torch.nn.functional.normalize(sky_vec, dim=1)
            
            # B. COMPUTE DUAL SIMILARITY
            # Score = Sim(Patch, Road) - Alpha * Sim(Patch, Sky)
            sim_road = torch.mm(features_n, road_vec_n.transpose(0, 1)).squeeze()
            sim_sky  = torch.mm(features_n, sky_vec_n.transpose(0, 1)).squeeze()
            
            # Contrastive Score
            score = sim_road - (0.6 * sim_sky)
            score_map = score.reshape(grid_h, grid_w).cpu().numpy()
            
            # C. APPLY HARD SPATIAL PRIORS (Trapezoid Mask)
            if roi_mask is None:
                roi_mask = np.zeros((grid_h, grid_w), dtype=np.float32)
                # Create a polygon for Valid Road Area
                # Top: narrow, Bottom: wide
                pts = np.array([
                    [grid_w*0.35, grid_h*0.35], # Top Left
                    [grid_w*0.65, grid_h*0.35], # Top Right
                    [grid_w*0.95, grid_h],      # Bottom Right
                    [grid_w*0.05, grid_h]       # Bottom Left
                ], np.int32)
                cv2.fillPoly(roi_mask, [pts], 1.0)
            
            # Apply Mask to Score
            masked_score = np.where(roi_mask > 0, score_map, -1.0)
            
            # D. EXTRACT PATH FROM SCORES
            new_path_raw = []
            
            # Create a segmentation mask for visualization
            seg_mask_small = (masked_score > 0.15).astype(np.uint8) # Threshold
            seg_mask = cv2.resize(seg_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Path Extraction: Scan rows from bottom to top
            # We trust the bottom more.
            start_row = int(grid_h * 0.95)
            end_row   = int(grid_h * 0.35)
            
            curr_x_grid = grid_w / 2
            
            for r in range(start_row, end_row, -1):
                row_scores = masked_score[r, :]
                row_max = np.max(row_scores)
                
                # Confidence check
                if row_max < 0.1: 
                    continue 
                
                # Find valid pixels (connected component logic simplified)
                # We only care about pixels that are SOMEWHAT connected to our current x
                
                # Windowed search around current x
                window = int(grid_w * 0.25)
                c_min = max(0, int(curr_x_grid - window))
                c_max = min(grid_w, int(curr_x_grid + window))
                
                # Local crop
                local_scores = row_scores[c_min:c_max]
                local_indices = np.arange(c_min, c_max)
                
                # Threshold locally
                valid_local = local_scores > (row_max * 0.85)
                
                if np.any(valid_local):
                    # Weighted Average
                    w_vals = local_scores[valid_local]
                    idx_vals = local_indices[valid_local]
                    
                    centroid = np.average(idx_vals, weights=w_vals)
                    
                    # Update tracker
                    curr_x_grid = 0.6 * curr_x_grid + 0.4 * centroid
                    
                    # Store
                    new_path_raw.append((curr_x_grid, r))
            
            # Convert to smoothed screen coordinates
            target_path = []
            for (gx, gy) in new_path_raw:
               sx = int((gx / grid_w) * w)
               sy = int((gy / grid_h) * h)
               target_path.append((sx, sy))
               
            # Smooth the path points slightly
            target_path = smooth_path(target_path, smoothing_window=3)

        # 3. DRAWING
        frame_count += 1
        
        # A. Draw Segmentation Overlay
        # Create a green overlay
        if 'seg_mask' in locals():
            green_layer = np.zeros_like(frame_display)
            green_layer[:, :, 1] = 200 # Green channel
            
            # Blend
            mask_indices = seg_mask > 0
            # ROI for display
            frame_display[mask_indices] = cv2.addWeighted(frame_display[mask_indices], 0.6, green_layer[mask_indices], 0.4, 0)

        # B. Draw Path Line
        if len(target_path) > 1:
            for i in range(1, len(target_path)):
                cv2.line(frame_display, target_path[i-1], target_path[i], (0, 255, 0), 4)
            # End point
            cv2.circle(frame_display, target_path[-1], 8, (0, 255, 0), -1)

        cv2.putText(frame_display, "Nano FSD: Semantic Road Segmentation", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow('Nano FSD - Road Segmentation', frame_display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
