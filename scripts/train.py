from ultralytics import YOLO

import os

# Configuration
MODEL_FILENAME = "yolo26s-pose.pt"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME) # Use local model file as base

DATA_YAML_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/dog-pose.yaml"
DATA_YAML = "dog-pose.yaml"
IMG_SIZE = 640

# --- Fix for Dataset Path Issue ---
from ultralytics import settings

# Update Ultralytics settings to use the current directory for datasets
# This prevents it from using a global path like /root/medical_AI/hkh/ellin/...
print(f"Original YOLO settings: {settings['datasets_dir']}")
settings.update({'datasets_dir': os.getcwd()})
print(f"Updated YOLO settings: {settings['datasets_dir']}")
# ----------------------------------

def download_dataset_yaml():
    import os
    if not os.path.exists(DATA_YAML):
        print(f"Downloading {DATA_YAML}...")
        import urllib.request
        urllib.request.urlretrieve(DATA_YAML_URL, DATA_YAML)

def train():
    download_dataset_yaml()
    # Load a model
    if os.path.exists(MODEL_PATH):
        print(f"Loading local model from {MODEL_PATH}...")
        model = YOLO(MODEL_PATH)
    else:
        raise FileNotFoundError(f"Error: Model not found at {MODEL_PATH}. Please ensure YOLO26s-pose.pt exists.") 

    # Train the model with H100 optimizations
    results = model.train(
        data=DATA_YAML,
        epochs=500, # Increased for better convergence
        imgsz=IMG_SIZE,
        batch=-1, # Auto batch (H100 can handle large batches)
        project="models/train", # Main logs directory
        name="logs", # Experiment name
        exist_ok=True,
        patience=50, # Increased patience
        amp=True,
        workers=16, # High worker count for H100
        cache=True, # RAM cache for speed
        cos_lr=True # Cosine learning rate scheduler
    )
    
    print("Training completed!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    
    # Post-training: Prepare for Deployment
    import shutil
    import json
    import yaml
    
    LATEST_DIR = "models/latest"
    if not os.path.exists(LATEST_DIR):
        os.makedirs(LATEST_DIR)
        
    # 1. Copy Best Model
    best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
    target_model_path = os.path.join(LATEST_DIR, "best.pt")
    if os.path.exists(best_model_path):
        shutil.copy2(best_model_path, target_model_path)
        print(f"Copied best model to {target_model_path}")
    else:
        print(f"Warning: best.pt not found at {best_model_path}")

    # 2. Generate Metadata for Hugging Face
    print("Generating metadata for Hugging Face...")
    
    # Read dataset yaml for class/keypoint names
    try:
        with open(DATA_YAML, 'r') as f:
            data_config = yaml.safe_load(f)
            
        # config.json
        config_data = {
            "names": data_config.get("names", {}),
            "kpt_names": data_config.get("kpt_names", {}),
            "kpt_shape": data_config.get("kpt_shape", [24, 3]),
            "task": "pose"
        }
        with open(os.path.join(LATEST_DIR, "config.json"), "w") as f:
            json.dump(config_data, f, indent=4)
            
        # preprocess.json (Standard YOLOv8)
        preprocess_data = {
            "input_size": [IMG_SIZE, IMG_SIZE],
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "reverse_channels": False # YOLO uses RGB, opencv reads BGR, but ultralytics handles it
        }
        with open(os.path.join(LATEST_DIR, "preprocess.json"), "w") as f:
            json.dump(preprocess_data, f, indent=4)

        # inference_config.json
        inference_config = {
            "conf_thres": 0.25,
            "iou_thres": 0.45,
            "max_det": 300
        }
        with open(os.path.join(LATEST_DIR, "inference_config.json"), "w") as f:
            json.dump(inference_config, f, indent=4)
            
        print(f"Metadata saved to {LATEST_DIR}")
        
    except Exception as e:
        print(f"Error generating metadata: {e}")

if __name__ == "__main__":
    train()
