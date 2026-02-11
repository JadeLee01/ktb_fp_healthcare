import json
import cv2
import argparse
import sys
import os
import glob
from tqdm import tqdm

def create_overlay_video(input_dir, output_path, fps=30):
    """
    Reads image frames and corresponding JSON files from a directory,
    overlays keypoints, and combines them into a video.
    
    Args:
        input_dir (str): Directory containing images and JSONs.
        output_path (str): Path to save the output video (can be a directory or a file path).
        fps (int): Frames per second for the output video.
    """
    
    # Check if output_path is a directory or a file
    if not output_path.endswith('.mp4') and not output_path.endswith('.avi'):
        # Treat as directory, create default filename
        os.makedirs(output_path, exist_ok=True)
        # Assuming input_dir name might be useful, but let's stick to a generic name or based on input dir
        input_name = os.path.basename(os.path.normpath(input_dir))
        output_video_path = os.path.join(output_path, f"{input_name}_overlay.mp4")
    else:
        # Treat as file path
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        output_video_path = output_path

    
    # 1. Collect all Image files
    # Support multiple extensions
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    # Sort carefully (assuming filename contains sequence number)
    # e.g., ..._00107.jpg
    image_files.sort()
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} frames.")

    # 2. Get Video Properties from first image
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        print(f"Error reading first image: {image_files[0]}")
        return
    
    height, width, layers = first_img.shape
    size = (width, height)
    
    # 3. Initialize Video Writer
    # MacOS often prefers 'mp4v' for QuickTime/Finder preview compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)
    
    # Check if writer opened successfully
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}. Trying 'avc1' codec...")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, size)
        
        if not out.isOpened():
             print(f"Error: Failed to open video writer with both 'mp4v' and 'avc1'.")
             return

    print(f"Creating video: {output_video_path} ({width}x{height} @ {fps}fps)")
    
    # 4. Process each frame
    for img_path in tqdm(image_files, desc="Processing Frames"):
        # Load Image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        # Look for corresponding JSON
        # Assumption: JSON has same basename as Image
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(input_dir, base_name + ".json")
        
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Overlay Keypoints
                if "annotation_info" in data:
                    for ann in data["annotation_info"]:
                        try:
                            x_norm = float(ann["x"])
                            y_norm = float(ann["y"])
                            label = ann.get("label", "") # Empty label for cleaner video? or keep it?
                            
                            # Skip if coordinates are 0 (sometimes used for missing)
                            if x_norm == 0 and y_norm == 0:
                                continue

                            x = int(x_norm * width)
                            y = int(y_norm * height)

                            # Draw Point (Red, filled)
                            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                            
                            # Optional: Draw Label text (Small)
                            # cv2.putText(img, label[:3], (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

                        except ValueError:
                            continue
                            
                # Overlay Metadata (Optional: Frame number)
                # cv2.putText(img, base_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error reading JSON {json_path}: {e}")
        else:
            pass # Just write the raw image if no JSON found

        # Resize if needed? No, keeping original resolution
        out.write(img)

    # 5. Release
    out.release()
    print(f"Video saved successfully to {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Overlay Video from Frame Images and JSONs")
    parser.add_argument("input_dir", help="Directory containing sequence of images and json files")
    parser.add_argument("--output", "-o", help="Output directory or video file path", default="results")
    parser.add_argument("--fps", type=int, help="Frames per second", default=30)
    
    args = parser.parse_args()
    create_overlay_video(args.input_dir, args.output, args.fps)
