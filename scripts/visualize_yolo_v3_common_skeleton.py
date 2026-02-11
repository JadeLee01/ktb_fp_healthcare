import cv2
import argparse
import sys
import os
import glob
from tqdm import tqdm
import imageio
import yaml

def visualize_yolo_dataset(dataset_path, output_path, fps=30):
    
    # 1. Parse dataset.yaml to get class names
    yaml_path = os.path.join(dataset_path, "integrated_dog_pose.yaml")
    if not os.path.exists(yaml_path):
        yamls = glob.glob(os.path.join(dataset_path, "*.yaml"))
        if yamls:
            yaml_path = yamls[0]
        else:
            print(f"Error: No .yaml config found in {dataset_path}")
            return

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        
    kpt_names = config.get('kpt_names', {})
    if isinstance(kpt_names, list):
        idx_to_name = {i: n for i, n in enumerate(kpt_names)}
    else:
        idx_to_name = kpt_names

    # 2. Collect Images
    img_dir = os.path.join(dataset_path, "images")
    label_dir = os.path.join(dataset_path, "labels")
    
    if not os.path.exists(img_dir):
        print(f"Error: images directory not found at {img_dir}")
        return

    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(img_dir, ext)))
    
    image_files.sort()
    
    if not image_files:
        print(f"No images found in {img_dir}")
        return

    # 3. Output Video setup
    if not output_path.endswith('.mp4'):
        os.makedirs(output_path, exist_ok=True)
        output_video_path = os.path.join(output_path, "v3_common_skeleton.mp4")
    else:
        output_video_path = output_path
        
    print(f"Creating video: {output_video_path}")
    writer = imageio.get_writer(output_video_path, fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)

    # V3: Common Skeleton (Same for all views)
    SKELETON = [
        ("Nose", "Chin"), ("Nose", "Ear"), ("Nose", "L_Ear"), ("Nose", "R_Ear"),
        ("Ear", "Withers"), ("L_Ear", "Withers"), ("R_Ear", "Withers"),
        ("Withers", "T13 Spinous precess"), ("T13 Spinous precess", "Iliac crest"),
        ("Iliac crest", "Sacrum"), ("Sacrum", "Tail start"), ("Tail start", "Tail end"),
        ("Withers", "L_Acromion/Greater tubercle"), ("Withers", "L_Dorsal scapular spine"), 
        ("L_Acromion/Greater tubercle", "L_Lateral humeral epicondyle"),
        ("L_Lateral humeral epicondyle", "L_Ulnar styloid process"),
        ("L_Ulnar styloid process", "L_Distal lateral aspect of fifth metacarpal bone"),
        ("Withers", "R_Acromion/Greater tubercle"), ("Withers", "R_Dorsal scapular spine"),
        ("R_Acromion/Greater tubercle", "R_Lateral humeral epicondyle"),
        ("R_Lateral humeral epicondyle", "R_Ulnar styloid process"),
        ("R_Ulnar styloid process", "R_Distal lateral aspect of fifth metacarpal bone"),
        ("Iliac crest", "L_Femoral greater trochanter"),
        ("L_Femoral greater trochanter", "L_Femorotibial joint"),
        ("L_Femorotibial joint", "L_Lateral malleolus of the distal tibia"),
        ("L_Lateral malleolus of the distal tibia", "L_Distal lateral aspect of the fifth metatarsus"),
        ("Iliac crest", "R_Femoral greater trochanter"),
        ("R_Femoral greater trochanter", "R_Femorotibial joint"),
        ("R_Femorotibial joint", "R_Lateral malleolus of the distal tibia"),
        ("R_Lateral malleolus of the distal tibia", "R_Distal lateral aspect of the fifth metatarsus"),
    ]
    
    skeleton_indices = []
    name_to_idx = {v: k for k, v in idx_to_name.items()}
    for p1_name, p2_name in SKELETON:
        idx1 = name_to_idx.get(p1_name)
        idx2 = name_to_idx.get(p2_name)
        if idx1 is not None and idx2 is not None:
            skeleton_indices.append((idx1, idx2))

    # 4. Process
    for img_path in tqdm(image_files, desc="Visualizing (V3: Common Skeleton)"):
        img = cv2.imread(img_path)
        if img is None: continue
        
        h, w, _ = img.shape
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(label_dir, base_name + ".txt")
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                kpts = parts[5:]
                
                # Parse Keypoints into a dict {idx: (x, y)} for easier access
                kp_dict = {}
                num_kpts = len(kpts) // 3
                for i in range(num_kpts):
                    kx = kpts[i*3]
                    ky = kpts[i*3 + 1]
                    kv = int(kpts[i*3 + 2])
                    
                    if kv > 0 and (kx != 0 or ky != 0):
                        px, py = int(kx * w), int(ky * h)
                        kp_dict[i] = (px, py)

                # Draw Skeleton Lines First
                for i1, i2 in skeleton_indices:
                    if i1 in kp_dict and i2 in kp_dict:
                        pt1 = kp_dict[i1]
                        pt2 = kp_dict[i2]
                        # Color logic
                        n1 = idx_to_name[i1]
                        n2 = idx_to_name[i2]
                        if ("L_" in n1 or "Left" in n1) and ("L_" in n2 or "Left" in n2):
                            color = (0, 0, 255) # Red
                        elif ("R_" in n1 or "Right" in n1) and ("R_" in n2 or "Right" in n2):
                            color = (255, 0, 0) # Blue
                        else:
                            color = (0, 255, 0) # Green
                            
                        cv2.line(img, pt1, pt2, color, 2)

                # Draw Points
                for i, (px, py) in kp_dict.items():
                    kpt_name = idx_to_name.get(i, "")
                    if "L_" in kpt_name or "Left" in kpt_name:
                        color = (0, 0, 255)
                    elif "R_" in kpt_name or "Right" in kpt_name:
                        color = (255, 0, 0)
                    else:
                        color = (0, 255, 0)
                    cv2.circle(img, (px, py), 4, color, -1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        writer.append_data(img_rgb)
        
    writer.close()
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize YOLO Dataset V3 (Common Skeleton)")
    parser.add_argument("dataset_path", help="Path to dataset root")
    parser.add_argument("--output", "-o", help="Output video path", default="v3_output.mp4")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    visualize_yolo_dataset(args.dataset_path, args.output, args.fps)
