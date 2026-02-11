from ultralytics import YOLO
import cv2
import numpy as np
from scripts.features import analyze_gait
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
REPO_ID = os.getenv("HF_REPO_ID", "20-team-daeng-ddang-ai/dog-pose-estimation")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_FILENAME = "best.pt" # Prioritize the trained model
MODEL_DIR = "." # model directory
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Directory Setup
INPUT_DIR = "test_videos"
OUTPUT_DIR = "results"

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print(f"Model {MODEL_PATH} not found locally.")
        
        # Check if we should download
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            
        print(f"Attempting to download {MODEL_FILENAME} from Hugging Face ({REPO_ID})...")
        try:
            from huggingface_hub import hf_hub_download
            if not HF_TOKEN:
                print("Warning: HF_TOKEN not found in environment variables. Download might fail for private repos.")
                
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID, 
                filename=MODEL_FILENAME, 
                local_dir=MODEL_DIR,
                token=HF_TOKEN
            )
            print(f"Model downloaded to {downloaded_path}")
        except Exception as e:
            print(f"Failed to download model: {e}")
            print("Please ensure HF_TOKEN is set in .env or environment if the repo is private.")

def frame_conf_from_result(r):
    if r.keypoints is None:
        return None

    conf = r.keypoints.conf

    if conf is None or conf.numel() == 0:
        return None

    conf_np = conf.cpu().numpy()
    # Typically inst_scores = conf_np.mean(axis=1) for instance selection
    # If multiple dogs, we pick the one with highest mean confidence
    inst_scores = conf_np.mean(axis=1)
    idx = int(np.argmax(inst_scores))
    frame_conf = float(conf_np[idx].mean())
    
    # Return frame_conf, keypoints(xy), and keypoint_confs
    # keypoints shape: (N, 17, 2/3)
    kpts = r.keypoints.xy.cpu().numpy()[idx] # (17, 2)
    kp_confs = conf_np[idx] # (17,)
    
    return frame_conf, kpts, kp_confs

def run_inference(video_path, model_path, conf_th=0.2, amp=True, save=True, save_dir=OUTPUT_DIR, min_valid_frames=60):
    """
    Runs inference on a single video and calculates metrics.
    """
    download_model_if_needed()
    
    print(f"Loading model from {model_path}...")
    if os.path.exists(model_path):
        model = YOLO(model_path) 
        print("Model loaded.")
    else:
        print("Model not found. Using mock logic completely.")
        model = None

    print(f"Processing video {video_path}...")
    
    if model:
        # Prepare customized video writer
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Construct overlay path manualy
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        out_writer = None
        overlay_video_path = None
        if save:
            output_dir = os.path.join(save_dir, f"{video_name}_overlay")
            os.makedirs(output_dir, exist_ok=True)
            overlay_video_path = os.path.join(output_dir, f"{video_name}.mp4")
            
            # Robust Video Writer Initialization
            # Try 'avc1' (H.264) first - Best for macOS/QuickTime/Web
            fourcc = cv2.VideoWriter_fourcc(*'avc1') 
            out_writer = cv2.VideoWriter(overlay_video_path, fourcc, fps, (width, height))
            
            if not out_writer.isOpened():
                print(f"Warning: Could not enable 'avc1' codec. Falling back to 'mp4v' (Linux default)...")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(overlay_video_path, fourcc, fps, (width, height))
                
            if not out_writer.isOpened():
                print(f"Error: Failed to initialize VideoWriter with both 'avc1' and 'mp4v'. Video will not be saved.")
                out_writer = None
            else:
                print(f"Processing and saving overlay to {overlay_video_path}...")
        
        # Initialize Smoother
        from visualization import KeypointSmoother, draw_overlay
        smoother = KeypointSmoother(alpha=0.6) # Adjust alpha for more/less smoothing
        
        # Run inference stream
        # Device: 0 for GPU, 'mps' for Mac Consumer GPU, 'cpu' for CPU
        import torch
        if torch.cuda.is_available():
            device = 0
        elif torch.backends.mps.is_available():
             device = "mps" 
        else:
            device = "cpu"
            
        print(f"Running inference on device: {device}")

        results = model.predict(
            source=video_path,
            conf=0.25,
            iou=0.45,
            save=False, # Manual saving
            device=device, 
            stream=True
        )
        
        valid_keypoints_sequence = []
        valid_confs_sequence = []
        frame_confs = []
        total_frames = 0
        
        print(f"Processing and saving overlay to {overlay_video_path}...")
        
        for r in results:
            total_frames += 1
            
            # Get original frame
            frame = r.orig_img.copy() # Make sure to copy if needed
            
            # Extract data
            out = frame_conf_from_result(r)
            
            if out:
                frame_conf, kpts, kp_confs = out
                
                # Smooth keypoints
                smoothed_kpts = smoother.update(kpts)
                
                # Get BBox (pick highest conf box if multiple)
                # r.boxes.xyxy is (N, 4)
                bbox = None
                if r.boxes.conf is not None and len(r.boxes.conf) > 0:
                     best_idx = int(np.argmax(r.boxes.conf.cpu().numpy()))
                     bbox = r.boxes.xyxy.cpu().numpy()[best_idx]

                # Draw Overlay
                frame = draw_overlay(frame, smoothed_kpts, keypoint_confs=kp_confs, bbox=bbox, kpt_radius=6, line_thickness=2, conf_th=conf_th)
                
                # Store original (or smoothed?) for analysis - usually analyze original to avoid lag bias, 
                # but smoothed might be cleaner. Let's store smoothed for consistency.
                frame_confs.append(frame_conf)
                valid_keypoints_sequence.append(smoothed_kpts)
                valid_confs_sequence.append(kp_confs)
            
            else:
                # No detection, draw nothing or just frame
                pass

            # Write frame
            out_writer.write(frame)
            
        out_writer.release()
        cap.release()
        
        mean_conf = float(np.mean(frame_confs)) if frame_confs else 0.0
        valid_frames = int(np.sum(np.array(frame_confs) >= conf_th)) if frame_confs else 0
        valid_ratio = valid_frames / max(total_frames, 1)
        
        # Calculate mean keypoint confidence for debugging
        mean_kp_conf = 0.0
        if valid_confs_sequence:
            all_confs = np.concatenate(valid_confs_sequence)
            mean_kp_conf = float(np.mean(all_confs))
            
        print(f"Total Frames: {total_frames}, Valid Frames: {valid_frames} (Ratio: {valid_ratio:.2f})")
        print(f"Mean Keypoint Confidence: {mean_kp_conf:.4f} (Threshold used: {conf_th})")
        
        # --- KEYPOINT DEBUGGING ---
        # 0: front_left_paw, 1: front_left_knee, 2: front_left_elbow, 3: rear_left_paw, 4: rear_left_knee, 5: rear_left_elbow
        # 6: front_right_paw, 7: front_right_knee, 8: front_right_elbow, 9: rear_right_paw, 10: rear_right_knee, 11: rear_right_elbow
        # 12: tail_start, 13: tail_end, 14: left_ear_base, 15: right_ear_base, 16: nose, 17: chin, 18: left_ear_tip, 19: right_ear_tip
        # 20: left_eye, 21: right_eye, 22: withers, 23: throat
        
        KEYPOINT_NAMES = [
            "FL_Paw", "FL_Knee", "FL_Elbow", "RL_Paw", "RL_Knee", "RL_Elbow",
            "FR_Paw", "FR_Knee", "FR_Elbow", "RR_Paw", "RR_Knee", "RR_Elbow",
            "Tail_Start", "Tail_End", "L_Ear_Base", "R_Ear_Base", "Nose", "Chin",
            "L_Ear_Tip", "R_Ear_Tip", "L_Eye", "R_Eye", "Withers", "Throat"
        ]
        
        if valid_confs_sequence:
            confs_np = np.array(valid_confs_sequence) # (F, 24)
            # Check if we have 24 keypoints or 17 (standard COCO)
            num_kpts_model = confs_np.shape[1]
            
            print(f"\n--- Keypoint Detection Analysis (Total {num_kpts_model} Kpts) ---")
            
            for i in range(num_kpts_model):
                kpt_name = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else f"Kpt_{i}"
                kpt_confs = confs_np[:, i]
                mean_conf = np.mean(kpt_confs)
                max_conf = np.max(kpt_confs)
                detection_rate = np.mean(kpt_confs > conf_th) * 100
                
                print(f"[{i:2d}] {kpt_name:<12} | Mean Conf: {mean_conf:.2f} | Max Conf: {max_conf:.2f} | Det Rate: {detection_rate:.1f}%")
                
        # ---------------------------

        if valid_frames < min_valid_frames:
            print(f"FAILED: INSUFFICIENT_FRAMES (Valid: {valid_frames} < {min_valid_frames})")
            return {"status": "failed", "error": "INSUFFICIENT_FRAMES"}
            
        # Convert to numpy array for features.py
        # shape: (F, 24, 2)
        valid_keypoints_np = np.array(valid_keypoints_sequence)
        valid_confs_np = np.array(valid_confs_sequence)
        
        print("Calculating metrics...")
        # Note: analyze_gait in features.py might expect 17 keypoints. 
        # If model outputs 24, we might need to adjust or features.py should handle it.
        # But for now, let's pass it and see.
        results = analyze_gait(valid_keypoints_np, valid_confs_np, conf_th)
        
        # Add overlay video path to results
        results["overlay_video_path"] = overlay_video_path
        print(f"Overlay video saved at: {overlay_video_path}")
        
        print("\nAnalysis Results:")
        for metric, score in results["metrics"].items():
            desc = results["descriptions"][metric]
            print(f"- {metric.capitalize()}: {score:.1f} ({desc})")
            
        return results


    else:
        # Mock logic (fallback)
        cap = cv2.VideoCapture(video_path)
        num_frames = 100
        num_kpts = 17 
        mock_keypoints = np.random.rand(num_frames, num_kpts, 2)
        
        print("Calculating metrics (MOCK)...")
        results = analyze_gait(mock_keypoints)
        return results

if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Created {INPUT_DIR}. Please place test videos there.")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    video_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
    
    if not video_files:
        print(f"No video files found in {INPUT_DIR}. Please add some videos.")
    else:
        for video_file in video_files:
            video_path = os.path.join(INPUT_DIR, video_file)
            print(f"\n--- Analyzing {video_file} ---")
            results = run_inference(video_path, MODEL_PATH)
            
            # Save results (JSON or simple text)
            if results:
                import json
                result_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(video_file)[0]}_result.json")
                # Convert numpy types to native types for JSON serialization if needed
                # For now assuming simple types or handling it manually if complex
                try:
                    with open(result_path, "w", encoding="utf-8") as f:
                        # Helper to handle numpy types
                        def default_converter(o):
                            if isinstance(o, np.generic): return o.item()
                            return o
                        json.dump(results, f, indent=4, default=default_converter, ensure_ascii=False)
                    print(f"Results saved to {result_path}")
                except Exception as e:
                    print(f"Error saving results: {e}")
