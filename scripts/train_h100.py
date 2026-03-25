from ultralytics import YOLO
import argparse
import os

def parse_cache_mode(cache_mode: str):
    cache_mode = cache_mode.lower()
    if cache_mode == "false":
        return False
    if cache_mode == "true":
        return True
    return cache_mode


def train_yolo(model_name, data_path, epochs, batch_size, imgsz, device, project_name, workers, cache_mode):
    # 1. Load the model
    # If model_name ends with .pt, it loads a pretrained model (recommended for fine-tuning)
    # If model_name ends with .yaml, it builds a new model from scratch
    print(f"Loading model: {model_name}")
    model = YOLO(model_name) 

    # 2. Train the model
    # H100 Optimization:
    # - batch: 64~128 (Depending on VRAM, 80GB is huge)
    # - workers: 8~16 (To avoid dataloader bottleneck)
    # - cache: True (Cache images in RAM for speed, H100 node usually has >200GB RAM)
    # - device: 0 (or list of devices for multi-gpu)
    
    print(f"Starting training on {device}...")
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project_name,
        name=f"train_{os.path.splitext(os.path.basename(model_name))[0]}", # e.g. train_yolo26m-pose
        exist_ok=True, # Overwrite existing experiment with same name? Optional
        pretrained=True,
        optimizer='auto', # YOLOv8/26 specific optimization (MuSGD if available in library)
        verbose=True,
        workers=workers,
        cache=parse_cache_mode(cache_mode),
        val=True, # Validate during training
        cos_lr=True, # Cosine learning rate scheduler (better convergence)
        close_mosaic=10, # Disable mosaic aug for last 10 epochs (better precision)
        patience=50, # Early stopping (waits 50 epochs without improvement)
    )
    
    print("Training Completed.")
    print(f"Results saved to {results.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO Pose Model on H100")
    
    # Default Args for H100
    parser.add_argument("--model", type=str, default="models/yolo26s-pose.pt", help="Model file (e.g., models/yolo26s-pose.pt)")
    parser.add_argument("--data", type=str, default="dataset_yolo/integrated_dog_pose.yaml", help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=128, help="Batch size (Try 128 for 's' model on 80GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (640 is standard, 1280 for high-res analysis)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id (e.g. 0 or 0,1,2,3)")
    parser.add_argument("--project", type=str, default="models", help="Save results to this project folder")
    parser.add_argument("--workers", type=int, default=32, help="Dataloader workers")
    parser.add_argument("--cache", type=str, default="true", choices=["true", "false", "ram", "disk"], help="Ultralytics cache mode")
    
    args = parser.parse_args()
    
    train_yolo(args.model, args.data, args.epochs, args.batch, args.imgsz, args.device, args.project, args.workers, args.cache)
