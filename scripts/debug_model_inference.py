import cv2
from ultralytics import YOLO
import sys

def debug_inference(video_path, model_path):
    print(f"--- Debugging Model Inference ---")
    print(f"Video: {video_path}")
    print(f"Model: {model_path}")

    # 1. Load Model
    try:
        model = YOLO(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Read First Frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read video frame.")
        return

    # Save input frame to check if colors are correct
    cv2.imwrite("debug_input_frame.jpg", frame)
    print("Saved first frame to 'debug_input_frame.jpg'. Please check if colors are correct (RGB vs BGR).")

    # Print Model Metadata
    if hasattr(model, 'names'):
        print(f"Model Class Names: {model.names}")
    else:
        print("Model has no 'names' attribute.")

    # 3. Predict with Extremely Low Confidence & Fixed Size
    print("\nRunning prediction with conf=0.001, imgsz=640...")
    results = model.predict(frame, conf=0.01, imgsz=640, verbose=True)
    
    # 4. Analyze Results
    result = results[0]
    
    print("\n--- Results Analysis ---")
    print(f"Boxes Detected: {len(result.boxes)}")
    
    if len(result.boxes) > 0:
        for i, box in enumerate(result.boxes):
            print(f"Box {i}: Conf={box.conf.item():.4f}, Class={box.cls.item()}")
            print(f"  Coords: {box.xyxy.tolist()}")
    else:
        print("WARNING: No boxes detected even at 0.01 confidence!")

    print(f"\nKeypoints Detected: {len(result.keypoints) if result.keypoints is not None else 0}")
    
    if result.keypoints is not None and len(result.keypoints.data) > 0:
        kpts = result.keypoints.data[0].cpu().numpy()
        print(f"Keypoints Shape: {kpts.shape}")
        
        # Count valid keypoints
        valid_kpts = 0
        for i, (x, y, conf) in enumerate(kpts):
            if conf > 0.01:
                valid_kpts += 1
                print(f"  Kpt {i}: ({x:.1f}, {y:.1f}), Conf={conf:.4f}")
        
        print(f"Total Valid Keypoints (>0.01): {valid_kpts}")
    else:
        print("WARNING: No keypoints detected!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python debug_model_inference.py <video_path> <model_path>")
    else:
        debug_inference(sys.argv[1], sys.argv[2])
