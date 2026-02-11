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
        output_video_path = os.path.join(output_path, "v2_colored_points.mp4")
    else:
        output_video_path = output_path
        
    print(f"Creating video: {output_video_path}")
    writer = imageio.get_writer(output_video_path, fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)

    # 4. Process
    for img_path in tqdm(image_files, desc="Visualizing (V2: Colored Points)"):
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
                num_kpts = len(kpts) // 3
                for i in range(num_kpts):
                    kx = kpts[i*3]
                    ky = kpts[i*3 + 1]
                    kv = int(kpts[i*3 + 2])
                    
                    if kv > 0 and (kx != 0 or ky != 0):
                        px, py = int(kx * w), int(ky * h)
                        
                        # V2: Color by Side (L/R)
                        kpt_name = idx_to_name.get(i, "")
                        if "L_" in kpt_name or "Left" in kpt_name:
                            color = (0, 0, 255) # Red (BGR)
                        elif "R_" in kpt_name or "Right" in kpt_name:
                            color = (255, 0, 0) # Blue (BGR)
                        else:
                            color = (0, 255, 0) # Green (BGR) - Center
                            
                        cv2.circle(img, (px, py), 5, color, -1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        writer.append_data(img_rgb)
        
    writer.close()
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize YOLO Dataset V2 (Colored Points)")
    parser.add_argument("dataset_path", help="Path to dataset root")
    parser.add_argument("--output", "-o", help="Output video path", default="v2_output.mp4")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    visualize_yolo_dataset(args.dataset_path, args.output, args.fps)
