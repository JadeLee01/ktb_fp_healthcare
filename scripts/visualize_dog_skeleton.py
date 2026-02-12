import cv2
from ultralytics import YOLO
import argparse
import os
import numpy as np

# --- 1. Dog Skeleton Structure Definition (34 Keypoints) ---
# Each tuple represents a connection between two keypoint indices.
# Colors: Green (Center), Blue (Left), Red (Right)
SKELETON_CONNECTIONS = [
    # Center Line (Spine & Head)
    (19, 1),   # Nose -> Chin
    (19, 5),   # Nose -> Ear (Center Head)
    (1, 4),    # Chin -> Dorsal Scapular Spine (Neck approximation)
    (5, 33),   # Ear -> Withers (Neck approximation)
    (4, 33),   # Dorsal Scapular -> Withers
    (33, 29),  # Withers -> T13 Spinous
    (29, 8),   # T13 -> Iliac Crest
    (8, 28),   # Iliac Crest -> Sacrum
    (28, 31),  # Sacrum -> Tail Start
    (31, 30),  # Tail Start -> Tail End

    # Left Front Leg (Blue)
    (33, 9),   # Withers -> L_Acromion (Shoulder)
    (9, 14),   # L_Acromion -> L_Humeral (Upper Arm)
    (14, 16),  # L_Humeral -> L_Ulnar (Forearm)
    (16, 10),  # L_Ulnar -> L_Metacarpal (Wrist/Paw)

    # Right Front Leg (Red)
    (33, 20),  # Withers -> R_Acromion
    (20, 25),  # R_Acromion -> R_Humeral
    (25, 27),  # R_Humeral -> R_Ulnar
    (27, 21),  # R_Ulnar -> R_Metacarpal

    # Left Back Leg (Blue)
    (28, 12),  # Sacrum -> L_Femoral (Hip)
    (12, 13),  # L_Femoral -> L_Femorotibial (Thigh)
    (13, 15),  # L_Femorotibial -> L_Malleolus (Lower Leg)
    (15, 11),  # L_Malleolus -> L_Metatarsus (Ankle/Paw)

    # Right Back Leg (Red)
    (28, 23),  # Sacrum -> R_Femoral
    (23, 24),  # R_Femoral -> R_Femorotibial
    (24, 26),  # R_Femorotibial -> R_Malleolus
    (26, 22),  # R_Malleolus -> R_Metatarsus
]

COLORS = {
    'center': (0, 255, 0),    # Green (BGR)
    'left': (255, 0, 0),      # Blue
    'right': (0, 0, 255)      # Red
}

def get_color(idx):
    # Determine color based on index range from yaml file
    # Left: 9~16
    # Right: 20~27
    if 9 <= idx <= 16:
        return COLORS['left']
    elif 20 <= idx <= 27:
        return COLORS['right']
    else:
        return COLORS['center']

# --- 2. Main Processing Function ---
def process_video(video_path, model_path, output_path, conf_threshold=0.5):
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Video Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30 

    # Codec for Mac Compatibility (H.264)
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_path}")

    frame_count = 0
    lines_drawn_total = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        lines_drawn_frame = 0 # Count lines in THIS frame
        points_drawn_frame = 0 # Count points in THIS frame
        
        # Inference
        results = model.predict(frame, conf=conf_threshold, verbose=False) # Get raw results
        # Note: We use lower internal threshold here to get more points, filtering later manually
        
        result = results[0]
        
        # --- Draw Skeleton & Numbers ---
        if result.keypoints is not None and len(result.keypoints.data) > 0:
            # We take the first detected dog (index 0)
            kpts = result.keypoints.data[0].cpu().numpy() # Shape: (34, 3) -> [x, y, conf]
            
            # 1. Draw Lines (Skeleton)
            for idx1, idx2 in SKELETON_CONNECTIONS:
                # Safety check: indices must be within range
                if idx1 >= len(kpts) or idx2 >= len(kpts): continue
                
                x1, y1, conf1 = kpts[idx1]
                x2, y2, conf2 = kpts[idx2]
                
                # RELAXED CONDITION: Draw if AVERAGE confidence > threshold
                # (This helps to draw lines even if one point is slightly weak but the pair makes sense)
                avg_conf = (conf1 + conf2) / 2
                
                if avg_conf > conf_threshold:
                    color = get_color(idx1) # Use start point color
                    try:
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        lines_drawn_frame += 1
                    except Exception as e:
                        pass # Ignore drawing errors (e.g. coordinates out of bounds)

            # 2. Draw Points & Numbers
            for idx, (x, y, conf) in enumerate(kpts):
                if conf < conf_threshold: continue
                
                # Point color
                color = get_color(idx)
                
                try:
                    # Draw Circle (Keypoint)
                    cv2.circle(frame, (int(x), int(y)), 5, color, -1)
                    
                    # Draw Number (Index)
                    # Offset text slightly to not cover the point
                    cv2.putText(frame, str(idx), (int(x)+8, int(y)-8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    points_drawn_frame += 1
                except:
                    pass

        # Write frame to output video
        out.write(frame)

        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: Drawn {lines_drawn_frame} lines, {points_drawn_frame} points", end='\r')

    cap.release()
    out.release()
    print(f"\nDone! Saved result to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Dog Skeleton & Keypoint Numbers")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model .pt file")
    parser.add_argument("--output", type=str, default="output_skeleton.mp4", help="Output video path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default 0.5)")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
    process_video(args.video, args.model, args.output, args.conf)
