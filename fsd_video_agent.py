import cv2
import torch
import numpy as np
import argparse
import os


class FSDDriver:
    def __init__(self, source, model_type='dinov2_vits14', device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Initializing Nano FSD on {self.device}")
        
        # Load Video Source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"❌ Could not open video source: {source}")
            
        # Load Visual Backbone
        print(f"⏳ Loading Vision Backbone ({model_type})...")
        try:
            # We use DINOv2 as the robust production-ready version of the 'DINO' family
            self.backbone = torch.hub.load('facebookresearch/dinov2', model_type)
            self.backbone.to(self.device)
            self.backbone.eval()
            print("✅ Visual Backbone Loaded")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
            
        # Parameters
        self.patch_size = 14
        self.dino_input_size = (448, 448) # Fixed size for inference stability
        self.viz_alpha = 0.4 # heatmap transparency

    def preprocess(self, frame):
        """Prepares frame for DINO inference"""
        # Resize to fixed DINO dimensions (must be multiple of 14)
        img = cv2.resize(frame, self.dino_input_size)
        
        # Normalize (Standard ImageNet mean/std)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # CHW format
        img = np.transpose(img, (2, 0, 1))
        t_img = torch.from_numpy(img).float().unsqueeze(0).to(self.device)
        return t_img

    def get_features(self, frame):
        """Extract patch features from the image"""
        t_img = self.preprocess(frame)
        
        with torch.no_grad():
            # Extract features (not class token, but patch tokens)
            # DINOv2 returns a dictionary or tuple depending on call, usually forward_features is safer
            features_dict = self.backbone.forward_features(t_img)
            # generic approach: check output keys
            patch_tokens = features_dict["x_norm_patchtokens"]
            
        return patch_tokens

    def find_drivable_area(self, features, h_grid, w_grid):
        """
        Segment the 'road' by comparing all patches to the 'ego-vehicle' patch directly ahead.
        """
        # Features: (1, N_patches, Dim)
        features = features.squeeze(0) # (N, D)
        
        # Define Road Query: The patch at the bottom center of the image
        # This represents "The road right in front of the car" which we assume is safe.
        # Grid coordinates
        center_x = w_grid // 2
        bottom_y = h_grid - 2 # Slightly up from very bottom to avoid hood/dashboard
        
        # Get query vector (maybe average a small 3x3 region for stability)
        query_indices = []
        for dy in range(-1, 2):
            for dx in range(-2, 3):
                 idx = (bottom_y + dy) * w_grid + (center_x + dx)
                 if 0 <= idx < features.shape[0]:
                     query_indices.append(idx)
        
        query_vec = features[query_indices].mean(dim=0) # (D,)
        
        # Compute Cosine Similarity
        # Sim = (A . B) / (|A|*|B|)
        # DINOv2 features are often already normalized or well behaved, but let's be strict
        feats_norm = torch.nn.functional.normalize(features, dim=1)
        query_norm = torch.nn.functional.normalize(query_vec.unsqueeze(0), dim=1)
        
        similarity = torch.mm(feats_norm, query_norm.transpose(0, 1)) # (N, 1)
        
        # Reshape to grid
        sim_map = similarity.reshape(h_grid, w_grid).cpu().numpy()
        
        return sim_map

    def plan_path(self, sim_map):
        """
        Simple heuristic path planner based on the similarity (drivability) map.
        """
        h, w = sim_map.shape
        
        # Threshold to binary
        # Logic: If similarity > 0.5 (or dynamic), it's road.
        thresh_val = np.percentile(sim_map, 70) # Adaptive threshold? or fixed 0.6
        mask = sim_map > max(0.4, thresh_val)
        
        # Simple Logic: For each row going up from bottom, find the centroid of the 'road' pixels
        path_points = []
        
        start_x = w // 2
        
        # We assume the path starts at bottom center
        path_points.append((w//2, h-1))
        
        # Search horizon (how far up can we go?)
        # We scan from bottom up
        scan_width = 10 # Search window around previous x
        current_x = start_x
        
        for y in range(h-1, h//3, -1): # Stop at top third (horizon)
            # Get row
            row = mask[y, :]
            
            # Identify valid pixels in this row
            valid_indices = np.where(row)[0]
            
            if len(valid_indices) == 0:
                # Road blocked/ends
                break
                
            # Find the valid pixel closest to our current trajectory (continuity)
            # or just the mean of the largest connected component nearest current_x
            
            # Simple approach: Weighted average of High Similarity pixels near current_x
            # Weigh by similarity score squared to pull towards center of road
            row_sim = sim_map[y, :]
            
            # Restrict search to window to prevent jumping to other lanes
            search_min = max(0, int(current_x - scan_width))
            search_max = min(w, int(current_x + scan_width))
            
            local_valid = []
            local_weights = []
            
            for x in range(search_min, search_max):
                if mask[y, x]:
                    local_valid.append(x)
                    local_weights.append(row_sim[x]**2)
            
            if not local_valid:
                # Expand search slightly if track lost
                if scan_width < w//2:
                    scan_width += 2
                continue # Skip this row, maybe gaps
            
            # Calculate new center
            if sum(local_weights) > 0:
                new_x = np.average(local_valid, weights=local_weights)
                current_x = 0.7 * current_x + 0.3 * new_x # Smooth update
                path_points.append((int(current_x), y))
            
        return path_points

    def run(self):
        print("▶️ Starting Loop...")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                # Loop video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # 1. Inference
            features = self.get_features(frame) # (1, N, D)
            
            # Calculate Grid Dimensions
            B, N, D = features.shape
            grid_h = self.dino_input_size[0] // self.patch_size
            grid_w = self.dino_input_size[1] // self.patch_size
            
            # 2. Perception
            sim_map = self.find_drivable_area(features, grid_h, grid_w)
            
            # 3. Visualization
            display_h, display_w = frame.shape[:2]
            
            # Just use the original frame, no heatmap overlay
            layout = frame.copy()
            
            # 4. Path Planning & Drawing
            # It's easier to plan in the small grid then scale up
            path_points = self.plan_path(sim_map)
            
            if len(path_points) > 1:
                # Scale points to display size
                scale_x = display_w / grid_w
                scale_y = display_h / grid_h
                
                display_points = []
                for p in path_points:
                    px = int(p[0] * scale_x)
                    py = int(p[1] * scale_y)
                    display_points.append((px, py))
                
                # Draw Curve
                for i in range(1, len(display_points)):
                    cv2.line(layout, display_points[i-1], display_points[i], (0, 255, 0), 4)
                
                # Draw "Ghost" car position
                start = display_points[0]
                cv2.circle(layout, start, 10, (0, 0, 255), -1)

            # Metadata Info
            cv2.putText(layout, "Nano FSD Vision-Based Planner", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(layout, "Backbone: DINOv2 (ViT-S/14)", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow('Nano FSD - Iteration 1', layout)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Nano FSD Video Agent')
    parser.add_argument('--video', type=str, help='Path to video file (mp4/avi)', default=None)
    
    args = parser.parse_args()
    
    # If no video provided, fallback to webcam 0
    source = args.video if args.video else 0
    # Try to convert to int if it's a digit (webcam index)
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    
    fsd = FSDDriver(source)
    fsd.run()
