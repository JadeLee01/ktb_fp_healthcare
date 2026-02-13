import cv2
from ultralytics import YOLO
import argparse
import os
import numpy as np

# --- YOLOv8/26 Pose Keypoints (24 Keypoints) ---
# Based on 'dog-pose.yaml' in ultralytics/datasets/dog-pose
KP_NAMES = {
    0: "FL Paw", 1: "FL Knee", 2: "FL Elbow",
    3: "RL Paw", 4: "RL Knee", 5: "RL Hock",
    6: "FR Paw", 7: "FR Knee", 8: "FR Elbow",
    9: "RR Paw", 10: "RR Knee", 11: "RR Hock",
    12: "Tail Start", 13: "Tail End",
    14: "L Ear Base", 15: "R Ear Base",
    16: "Nose", 17: "Chin",
    18: "L Ear Tip", 19: "R Ear Tip",
    20: "L Eye", 21: "R Eye",
    22: "Withers", 23: "Throat"
}

SKELETON_CONNECTIONS = [
    # --- Legs (Front Left) ---
    (0, 1), (1, 2), # Paw -> Knee -> Elbow
    (2, 22),        # Elbow -> Withers (Connect to body)

    # --- Legs (Rear Left) ---
    (3, 4), (4, 5), # Paw -> Knee -> Hock
    (5, 12),        # Hock -> Tail Start (Approximation of Hip)

    # --- Legs (Front Right) ---
    (6, 7), (7, 8), # Paw -> Knee -> Elbow
    (8, 22),        # Elbow -> Withers

    # --- Legs (Rear Right) ---
    (9, 10), (10, 11), # Paw -> Knee -> Hock
    (11, 12),          # Hock -> Tail Start

    # --- Body / Spine ---
    (12, 13), # Tail Start -> Tail End
    (12, 22), # Tail Start -> Withers (Spine Line)
    (22, 23), # Withers -> Throat (Neck Line)

    # --- Head ---
    (23, 17), # Throat -> Chin
    (17, 16), # Chin -> Nose
    (16, 20), (16, 21), # Nose -> Eyes
    (20, 14), (21, 15), # Eyes -> Ear Base
    (14, 18), (15, 19), # Ear Base -> Ear Tip
    (14, 15)  # Ear Base -> Ear Base (Top of head)
]

COLORS = {
    'left': (255, 0, 0),      # Blue (BGR)
    'right': (0, 0, 255),     # Red
    'center': (0, 255, 0),    # Green
    'head': (0, 255, 255)     # Yellow
}

def get_color(idx):
    if idx in [0, 1, 2, 3, 4, 5, 14, 18, 20]:
        return COLORS['left']
    elif idx in [6, 7, 8, 9, 10, 11, 15, 19, 21]:
        return COLORS['right']
    elif idx in [16, 17, 23]:
        return COLORS['head']
    else:
        return COLORS['center']

def process_video(video_path, model_path, output_path, conf_threshold=0.3):
    print(f"Loading 24-Keypoint model: {model_path}")
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use 'avc1' for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {video_path} -> {output_path}")

    frame_count = 0
    
    # Statistics
    kpt_stats = {i: 0 for i in range(24)} # Count detections per keypoint
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        # Inference
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        
        detected_indices = []
        if len(results[0].keypoints) > 0:
            kpts = results[0].keypoints.data[0].cpu().numpy() # (24, 3)

            # 1. Draw Lines
            for idx1, idx2 in SKELETON_CONNECTIONS:
                if idx1 < len(kpts) and idx2 < len(kpts):
                    x1, y1, c1 = kpts[idx1]
                    x2, y2, c2 = kpts[idx2]
                    
                    if c1 > conf_threshold and c2 > conf_threshold:
                        color = get_color(idx1)
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # 2. Draw Points & IDs & Update Stats
            for i, (x, y, conf) in enumerate(kpts):
                if conf > conf_threshold:
                    detected_indices.append(i)
                    kpt_stats[i] += 1
                    
                    color = get_color(i)
                    cv2.circle(frame, (int(x), int(y)), 4, color, -1)
                    # Draw ID number
                    cv2.putText(frame, str(i), (int(x)+5, int(y)-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        out.write(frame)
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}/{total_frames}: Detected {len(detected_indices)} points: {detected_indices}")

    cap.release()
    out.release()
    print(f"\nDone! Saved to {output_path}")
    
    # Print Statistics Summary
    print("\n" + "="*40)
    print(f"Keypoint Detection Summary (Total Frames: {frame_count})")
    print(f"Confidence Threshold: {conf_threshold}")
    print("="*40)
    print(f"{'ID':<4} {'Name':<15} {'Count':<8} {'Ratio (%)'}")
    print("-" * 40)
    
    for i in range(24):
        count = kpt_stats[i]
        ratio = (count / frame_count) * 100 if frame_count > 0 else 0
        name = KP_NAMES.get(i, f"Kpt_{i}")
        status = "✅" if ratio > 80 else ("⚠️" if ratio > 20 else "❌")
        print(f"{i:<4} {name:<15} {count:<8} {ratio:.1f}% {status}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="models/best_26m.pt")
    parser.add_argument("--output", default="output_verified.mp4")
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()
    
    process_video(args.video, args.model, args.output, args.conf)
