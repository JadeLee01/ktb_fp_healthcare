import cv2
from ultralytics import YOLO
import argparse
import os

def process_video(video_path, model_path, output_path, conf_threshold=0.5):
    # 1. Load Model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # 2. Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30 # Default if unknown

    # 3. Setup Video Writer (MP4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_path}")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...", end='\r')

        # 4. Inference
        # verbose=False to keep terminal clean
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        
        # 5. Overlay Results
        annotated_frame = results[0].plot() # This draws the bounding boxes and keypoints!
        
        # 6. Write Frame
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"\nDone! Saved result to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay YOLO Pose Keypoints on Video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model", type=str, required=True, help="Path to trained YOLO model (.pt)")
    parser.add_argument("--output", type=str, default="output_overlay.mp4", help="Path to save output video")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (0.0~1.0)")
    
    args = parser.parse_args()
    
    # Ensure output directory exists if path contains directories
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
    process_video(args.video, args.model, args.output, args.conf)
