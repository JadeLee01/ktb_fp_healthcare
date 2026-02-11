from ultralytics import YOLO
import argparse
import os

def train_yolo(model_name, data_path, epochs, batch_size, imgsz, device, project_name):
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
        workers=16, # High workers for fast I/O
        cache=True, # RAM caching for speed
        val=True, # Validate during training
        cos_lr=True, # Cosine learning rate scheduler (better convergence)
        close_mosaic=10, # Disable mosaic aug for last 10 epochs (better precision)
    )
    
    print("Training Completed.")
    print(f"Results saved to {results.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO Pose Model on H100")
    
    # Default Args for H100
    parser.add_argument("--model", type=str, default="yolo26m-pose.pt", help="Model file (e.g., yolo26m-pose.pt, yolov8x-pose-p6.pt)")
    parser.add_argument("--data", type=str, default="training/healthcare/datasets/integrated_dog_pose.yaml", help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size (Try 128 for 's' model on 80GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (640 is standard, 1280 for high-res analysis)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id (e.g. 0 or 0,1,2,3)")
    parser.add_argument("--project", type=str, default="training/healthcare/models", help="Save results to this project folder")
    
    args = parser.parse_args()
    
    train_yolo(args.model, args.data, args.epochs, args.batch, args.imgsz, args.device, args.project)
