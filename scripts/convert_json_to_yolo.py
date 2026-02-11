import json
import os
import argparse
import numpy as np
from pathlib import Path

def convert_to_yolo(input_dir, output_dir, limit_labels=None):
    """
    Converts custom JSON annotations to YOLOv8 Pose format.
    
    Args:
        input_dir (str): Directory containing JSON files and Images.
        output_dir (str): Directory to save YOLO labels and dataset.yaml.
        limit_labels (list): Optional list of specific labels to include (and order).
                             If None, scans dataset for all unique labels.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Collect all JSON files
    json_files = sorted(list(input_path.glob("*.json")))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    # 2. Determine Labels (Keypoints)
    if limit_labels:
        labels = limit_labels
        print(f"Using provided labels: {labels}")
    else:
        print("Scanning for unique labels...")
        unique_labels = set()
        for jf in json_files:
            try:
                with open(jf, "r", encoding="gray-8", errors="ignore") as f: # handling encoding issues
                   # Try utf-8 first
                   pass
            except:
                pass
            
            # Re-read properly
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "annotation_info" in data:
                        for ann in data["annotation_info"]:
                            if "label" in ann:
                                unique_labels.add(ann["label"])
            except Exception as e:
                print(f"Error reading {jf}: {e}")
        
        labels = sorted(list(unique_labels))
        print(f"Found {len(labels)} unique labels: {labels}")

    label_to_idx = {name: i for i, name in enumerate(labels)}

    # 3. Create dataset.yaml content
    yaml_content = f"""
path: {input_dir} # dataset root dir
train: . # train images (relative to 'path') - assuming all in one folder for now
val: . # val images (relative to 'path')

# Keypoints
kpt_shape: [{len(labels)}, 3] # number of keypoints, number of dims (3 for x,y,visible)

# Classes
names:
  0: dog

# Keypoint names
kpt_names:
"""
    for i, name in enumerate(labels):
        yaml_content += f"  {i}: {name}\n"
    
    with open(output_path / "custom_dog_pose.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"Created dataset config: {output_path / 'custom_dog_pose.yaml'}")

    # 4. Process each file
    count = 0
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if "annotation_info" not in data:
                continue
                
            # Extract keypoints
            keypoints = {} # name -> (x, y)
            for ann in data["annotation_info"]:
                lname = ann.get("label")
                if lname in label_to_idx:
                    # Parse x, y (strings in example)
                    try:
                        kx = float(ann["x"])
                        ky = float(ann["y"])
                        keypoints[lname] = (kx, ky)
                    except ValueError:
                        continue
            
            if not keypoints:
                continue

            # Compute Bounding Box from Keypoints
            # Find min/max x, y
            xs = [p[0] for p in keypoints.values()]
            ys = [p[1] for p in keypoints.values()]
            
            if not xs or not ys:
                continue
                
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # Add padding (e.g. 10% of size)
            w = max_x - min_x
            h = max_y - min_y
            pad_x = w * 0.1
            pad_y = h * 0.1
            
            min_x = max(0, min_x - pad_x)
            max_x = min(1, max_x + pad_x)
            min_y = max(0, min_y - pad_y)
            max_y = min(1, max_y + pad_y)
            
            # YOLO BBox format: center_x, center_y, width, height (normalized)
            bbox_w = max_x - min_x
            bbox_h = max_y - min_y
            cx = min_x + bbox_w / 2
            cy = min_y + bbox_h / 2
            
            # Format Keypoints sequence: x y v (v=2 for visible, 0 for missing)
            kpt_line = []
            for name in labels:
                if name in keypoints:
                    x, y = keypoints[name]
                    kpt_line.extend([f"{x:.6f}", f"{y:.6f}", "2"])
                else:
                    kpt_line.extend(["0.000000", "0.000000", "0"])
            
            # Construct YOLO line
            # Class ID (0) + BBox + Keypoints
            line_parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bbox_w:.6f}", f"{bbox_h:.6f}"] + kpt_line
            yolo_line = " ".join(line_parts)
            
            # Save .txt
            # Filename logic: matches image filename usually. 
            # The JSON has "filename": "치료멍멍..." (no extension)
            # We save as .txt
            
            base_name = data.get("image_info", {}).get("filename", jf.stem)
            # If filename in json has extension, strip it
            if  "." in base_name and len(base_name.split(".")[-1]) <= 4: # heuristics
                 base_name = ".".join(base_name.split(".")[:-1])
                 
            txt_filename = base_name + ".txt"
            with open(output_path / txt_filename, "w", encoding="utf-8") as out_f:
                out_f.write(yolo_line + "\n")
                
            count += 1
            
        except Exception as e:
            print(f"Error converting {jf}: {e}")

    print(f"Successfully converted {count} files to YOLO format in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Custom JSON to YOLOv8 Pose Format")
    parser.add_argument("input_dir", help="Directory containing JSON files")
    parser.add_argument("output_dir", help="Directory to save YOLO .txt labels")
    
    args = parser.parse_args()
    convert_to_yolo(args.input_dir, args.output_dir)
