import cv2
import argparse
import os

def check_dataset_gt(image_path, label_path, output_path="gt_check.jpg"):
    # 1. Read Image
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image: {image_path}")
        return

    h, w, _ = img.shape
    print(f"Image Size: {w}x{h}")

    # 2. Read Label File
    if not os.path.exists(label_path):
        print(f"Error: Label file not found: {label_path}")
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()

    print(f"Found {len(lines)} objects in label file.")

    # 3. Draw GT
    for line in lines:
        parts = list(map(float, line.strip().split()))
        
        # YOLO Format: class x_center y_center width height kp1_x kp1_y kp1_vis ...
        cls = int(parts[0])
        x_c, y_c, bw, bh = parts[1:5]
        
        # Draw BBox
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Class {cls}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        print(f"Object {cls}: Box ({x1},{y1}) -> ({x2},{y2})")

        # Draw Keypoints (Remaining parts)
        keypoints = parts[5:]
        num_kpts = len(keypoints) // 3
        
        for i in range(num_kpts):
            kx = keypoints[i * 3]
            ky = keypoints[i * 3 + 1]
            vis = int(keypoints[i * 3 + 2]) # 0: invisible, 1: visible, 2: visible & labeled
            
            if vis > 0: # Only draw visible keypoints
                px = int(kx * w)
                py = int(ky * h)
                
                # Color based on visibility (Yellow=1, Red=2)
                color = (0, 255, 255) if vis == 1 else (0, 0, 255)
                
                cv2.circle(img, (px, py), 4, color, -1)
                # cv2.putText(img, str(i), (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # 4. Save Result
    cv2.imwrite(output_path, img)
    print(f"Check result saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Ground Truth Labels (YOLO Format)")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--label", type=str, required=True, help="Path to label .txt file")
    parser.add_argument("--output", type=str, default="gt_check.jpg", help="Output image path")
    
    args = parser.parse_args()
    check_dataset_gt(args.image, args.label, args.output)
