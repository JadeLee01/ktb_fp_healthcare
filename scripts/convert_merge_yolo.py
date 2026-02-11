import json
import os
import argparse
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

def get_integrated_keypoints():
    """
    Defines the full list of 34 unique keypoints covering all views (Left, Right, Front, Back).
    Left/Right specific points are prefixed with L_ or R_.
    Center points (Spine, Head, Tail) are shared.
    """
    return [
        # --- Head & Spin (Center) ---
        "Nose", "Chin", "Ear", "Withers", 
        "T13 Spinous precess", "Dorsal scapular spine", 
        "Iliac crest", "Sacrum", "Tail start", "Tail end",
        # --- Front Legs (Left/Right) ---
        "L_Acromion/Greater tubercle", "R_Acromion/Greater tubercle",
        "L_Lateral humeral epicondyle", "R_Lateral humeral epicondyle",
        "L_Ulnar styloid process", "R_Ulnar styloid process",
        "L_Distal lateral aspect of fifth metacarpal bone", "R_Distal lateral aspect of fifth metacarpal bone",
        # --- Rear Legs (Left/Right) ---
        "L_Femoral greater trochanter", "R_Femoral greater trochanter",
        "L_Femorotibial joint", "R_Femorotibial joint",
        "L_Lateral malleolus of the distal tibia", "R_Lateral malleolus of the distal tibia",
        "L_Distal lateral aspect of the fifth metatarsus", "R_Distal lateral aspect of the fifth metatarsus",
        # --- Likely missing ones but added for completeness based on standard sets ---
        "L_Ear", "R_Ear", "L_Eye", "R_Eye" 
    ]

# Mapping rules: Plain name in JSON -> Integrated Name
# If the JSON is from 'Left' folder, map 'Acromion' -> 'L_Acromion...'
# If 'Right', map 'Acromion' -> 'R_Acromion...'
# Measurements that are center-aligned (Spine, Tail, Head) don't get L/R unless explicitly needed.

COMMON_POINTS = [
    "Dorsal scapular spine", "T13 Spinous precess", "Iliac crest", 
    "Sacrum", "Tail start", "Tail end", "Nose", "Chin", "Withers", "Ear"
]

def map_label(original_label, view):
    """
    Maps original label to integrated label based on view (Left/Right/Front/Back).
    """
    # 1. Normalize label
    label = original_label.strip()
    
    # 2. Check clear L/R indicators in label itself
    if "Left" in label or label.startswith("L_"):
        return label # Already specific?
        
    # 3. Check for center points
    if label in COMMON_POINTS:
        return label
        
    # 4. View-based mapping for limbs
    # If generic name like "Acromion/Greater tubercle", map based on folder view
    if view == "Left":
        return f"L_{label}"
    elif view == "Right":
        return f"R_{label}"
    
    # Front/Back might contain both, but usually labeled explicitly in those datasets?
    # If Front/Back JSONs use generic names, we might have ambiguity without more logic.
    # For now, assume Front/Back datasets label L/R explicitly or allow generic mapping.
    return label

def convert_and_merge_yolo(image_root, label_root, output_dir, split="train"):
    """
    Scans for JSONs in label_root (handling Front/Back/Left/Right subfolders).
    Locates corresponding images in image_root.
    Converts to YOLO format with UNIFIED keypoint indices and L/R correction.
    
    Args:
        split (str): 'train' or 'val'. Determines output subdirectory.
    """
    
    image_path_root = Path(image_root)
    label_path_root = Path(label_root)
    output_path = Path(output_dir)
    
    # YOLO Standard Structure:
    # dataset/
    #   images/
    #     train/
    #     val/
    #   labels/
    #     train/
    #     val/
    
    images_out = output_path / "images" / split
    labels_out = output_path / "labels" / split
    
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    
    print(f"[{split.upper()}] Step 1: Scanning for JSON labels and matching images...")
    all_files = [] # (json_path, image_path, view_type)
    
    img_exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    
    # Use rglob to find all JSONs recursively
    json_files = list(label_path_root.rglob("*.json"))
    print(f"  Found {len(json_files)} JSON files in {label_root}")
    
    for json_file in tqdm(json_files, desc="Matching Images"):
        # Determine View
        parent_name = json_file.parent.name
        view = "Unknown"
        
        # Check parent folder name
        if "Left" in parent_name or "left" in parent_name: view = "Left"
        elif "Right" in parent_name or "right" in parent_name: view = "Right"
        elif "Front" in parent_name or "front" in parent_name: view = "Front"
        elif "Back" in parent_name or "back" in parent_name: view = "Back"
        else:
            fname = json_file.name.lower()
            if "left" in fname: view = "Left"
            elif "right" in fname: view = "Right"
            elif "front" in fname: view = "Front"
            elif "back" in fname: view = "Back"
            
        # Find corresponding Image
        # We need to find the image regardless of deep nesting.
        # Efficient Strategy: Match relative path from 'label_root' to 'image_root'
        # Example: label_root/1기/2024.../Back/foo.json -> image_root/1기/2024.../Back/foo.jpg
        
        try:
            rel_path = json_file.relative_to(label_path_root)
            # Construct theoretical image path
            potential_img_dir = image_path_root / rel_path.parent
            base_name = json_file.stem
            
            img_found = None
            for ext in img_exts:
                probe = potential_img_dir / (base_name + ext)
                if probe.exists():
                    img_found = probe
                    break
            
            if img_found:
                all_files.append((json_file, img_found, view))
            else:
                # If strict structure match fails, we might want to try name-only match?
                # But with 120k files, name collisions are possible. strict is safer.
                pass
        except ValueError:
            pass

    if not all_files:
        print(f"No matched JSON-Image pairs found for {split}!")
        return

    print(f"[{split.upper()}] Found {len(all_files)} matched pairs.")

    # 2. Build Unified Label Set (We use a fixed set to ensure train/val consistency)
    # If we build dynamically from train, val might miss some keys. 
    # Let's use the PRE-DEFINED Sorted Keys from our previous run/knowledge to be safe.
    
    # However, for this script to be standalone, let's scan. 
    # BUT, we must ensure consistency.
    # BEST PRACTICE: Use the fixed list we derived earlier.
    
    SORTED_LABELS = [
        'Acromion/Greater tubercle', 'Distal lateral aspect of fifth metacarpal bone', 
        'Distal lateral aspect of the fifth metatarsus', 'Dorsal scapular spine', 'Ear', 
        'Femoral greater trochanter', 'Femorotibial joint', 'Iliac crest', 
        'L_Acromion/Greater tubercle', 'L_Distal lateral aspect of fifth metacarpal bone', 
        'L_Distal lateral aspect of the fifth metatarsus', 'L_Femoral greater trochanter', 
        'L_Femorotibial joint', 'L_Lateral humeral epicondyle', 'L_Lateral malleolus of the distal tibia', 
        'L_Ulnar styloid process', 'Lateral humeral epicondyle', 'Lateral malleolus of the distal tibia', 
        'R_Acromion/Greater tubercle', 'R_Distal lateral aspect of fifth metacarpal bone', 
        'R_Distal lateral aspect of the fifth metatarsus', 'R_Femoral greater trochanter', 
        'R_Femorotibial joint', 'R_Lateral humeral epicondyle', 'R_Lateral malleolus of the distal tibia', 
        'R_Ulnar styloid process', 'T13 Spinous precess', 'Ulnar styloid process',
        # Added purely from our manual list to ensure coverage if not found in scan:
        'Nose', 'Chin', 'Withers', 'Sacrum', 'Tail start', 'Tail end' 
    ]
    # Re-sort to be sure
    SORTED_LABELS = sorted(list(set(SORTED_LABELS)))
    
    # 3. Process & Convert
    print(f"[{split.upper()}] Step 3: Converting files...")
    success_count = 0
    
    for jf, img_path, view in tqdm(all_files, desc=f"Converting {split}"):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if "annotation_info" not in data:
                continue
            
            # 3.3 Map Keypoints
            raw_keypoints = {} 
            for ann in data["annotation_info"]:
                l = ann.get("label")
                if not l: continue
                l = l.strip()
                try:
                    kx = float(ann["x"])
                    ky = float(ann["y"])
                except ValueError:
                    continue
                if l not in raw_keypoints:
                    raw_keypoints[l] = []
                raw_keypoints[l].append((kx, ky))
            
            keypoints = {}
            for l, points in raw_keypoints.items():
                if l in COMMON_POINTS or "Left" in l or "Right" in l or l.startswith("L_") or l.startswith("R_"):
                     keypoints[map_label(l, view)] = points[0]
                     continue
                
                if view == "Left":
                    keypoints[f"L_{l}"] = points[0]
                elif view == "Right":
                    keypoints[f"R_{l}"] = points[0]
                elif view == "Front" or view == "Back":
                    if len(points) == 2:
                        p1, p2 = points[0], points[1]
                        if view == "Front":
                            if p1[0] < p2[0]: # Image Left -> Dog Right
                                keypoints[f"R_{l}"] = p1
                                keypoints[f"L_{l}"] = p2
                            else:
                                keypoints[f"L_{l}"] = p1
                                keypoints[f"R_{l}"] = p2
                        else: # Back
                            if p1[0] < p2[0]: # Image Left -> Dog Left
                                keypoints[f"L_{l}"] = p1
                                keypoints[f"R_{l}"] = p2
                            else:
                                keypoints[f"R_{l}"] = p1
                                keypoints[f"L_{l}"] = p2
                    elif len(points) == 1:
                        pass 
            
            if not keypoints:
                continue

            # 3.4 Bounding Box
            xs = [p[0] for p in keypoints.values()]
            ys = [p[1] for p in keypoints.values()]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            w_span = max_x - min_x
            h_span = max_y - min_y
            pad_x = w_span * 0.15
            pad_y = h_span * 0.15
            b_min_x = max(0, min_x - pad_x)
            b_max_x = min(1, max_x + pad_x)
            b_min_y = max(0, min_y - pad_y)
            b_max_y = min(1, max_y + pad_y)
            cx = (b_min_x + b_max_x) / 2
            cy = (b_min_y + b_max_y) / 2
            bbox_w = b_max_x - b_min_x
            bbox_h = b_max_y - b_min_y
            
            # 3.5 Construct Keypoint Sequence
            kpt_line = []
            for name in SORTED_LABELS:
                if name in keypoints:
                    px, py = keypoints[name]
                    kpt_line.extend([f"{px:.6f}", f"{py:.6f}", "2"])
                else:
                    kpt_line.extend(["0.000000", "0.000000", "0"])
            
            # 3.6 Write Output
            # Use relative path + filename to ensure uniqueness and structure?
            # Or just flat filename... with 120k files, flat folder is bad for OS.
            # Let's keep a flat structure but use unique names to avoid collision.
            # unique_filename = f"{view}_{base_name}" 
            # Better: use partial path hash or just UUID if collision likely?
            # Provided structure seemed date-based, so filenames might be unique enough?
            # Let's stick to View_Filename for now.
            
            unique_filename = f"{view}_{jf.stem}"
            
            shutil.copy2(img_path, images_out / (unique_filename + img_path.suffix))
            
            yolo_str = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bbox_w:.6f}", f"{bbox_h:.6f}"] + kpt_line
            
            with open(labels_out / (unique_filename + ".txt"), "w", encoding="utf-8") as f_out:
                f_out.write(" ".join(yolo_str) + "\n")
                
            success_count += 1
            
        except Exception as e:
            pass
            
    print(f"[{split.upper()}] Done! Converted {success_count} samples.")
    
    # 4. Generate dataset.yaml (Always overwrite to match current labels list)
    # Output this only once (maybe check if split is 'train'?)
    # Or just write it every time, it's small.
    
    yaml_content = f"""
path: {output_path.absolute()} # dataset root dir
train: images/train 
val: images/val 

# Keypoints
kpt_shape: [{len(SORTED_LABELS)}, 3] 

# Classes
names:
  0: dog

# Keypoint names
kpt_names:
"""
    for i, name in enumerate(SORTED_LABELS):
        yaml_content += f"  {i}: {name}\n"
        
    with open(output_path / "integrated_dog_pose.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"Updated config: {output_path / 'integrated_dog_pose.yaml'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Separate Image/Label Folders into Unified YOLO Dataset")
    parser.add_argument("--images", required=True, help="Root folder containing images")
    parser.add_argument("--labels", required=True, help="Root folder containing JSON labels")
    parser.add_argument("--output", required=True, help="Where to create the YOLO dataset")
    parser.add_argument("--split", choices=["train", "val"], default="train", help="Which split is this? (train/val)")
    
    args = parser.parse_args()
    convert_and_merge_yolo(args.images, args.labels, args.output, args.split)
