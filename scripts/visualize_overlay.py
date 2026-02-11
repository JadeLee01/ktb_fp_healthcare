import json
import cv2
import argparse
import sys
import os

def overlay_keypoints(json_path, image_path=None):
    """
    Overlays keypoints from a JSON file onto an image.
    The JSON file is expected to have 'annotation_info' with 'x', 'y' (normalized), and 'label'.
    """
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    # Determine image path
    if image_path is None:
        # Try to infer image path from JSON content
        if "image_info" in data and "filename" in data["image_info"]:
            base_filename = data["image_info"]["filename"]
            # Check for common extensions
            for ext in [".jpg", ".jpeg", ".png"]:
                potential_path = base_filename + ext
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
                # Try relative to json dir
                json_dir = os.path.dirname(json_path)
                potential_path = os.path.join(json_dir, base_filename + ext)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
        
    if image_path is None or not os.path.exists(image_path):
        print(f"Error: Image file not found. Please provide image path explicitly.")
        # If we can't find image, we can't process further
        return

    print(f"Processing: {json_path} -> {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: valid image could not be loaded from {image_path}")
        return

    height, width = img.shape[:2]

    # Overlay keypoints
    if "annotation_info" in data:
        for ann in data["annotation_info"]:
            try:
                x_norm = float(ann["x"])
                y_norm = float(ann["y"])
                label = ann.get("label", "Unknown")

                x = int(x_norm * width)
                y = int(y_norm * height)

                # Draw point
                cv2.circle(img, (x, y), 6, (0, 0, 255), -1)

                # Draw label
                cv2.putText(
                    img,
                    label,
                    (x + 10, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255), # White text
                    1,
                    cv2.LINE_AA
                )
                # Black outline for better visibility
                cv2.putText(
                    img,
                    label,
                    (x + 10, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0), # Black outline
                    2, # Thickness
                    cv2.LINE_AA
                )
                cv2.putText(
                    img,
                    label,
                    (x + 10, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255), # White fill
                    1, 
                    cv2.LINE_AA
                )

            except ValueError:
                print(f"Skipping invalid annotation: {ann}")
                continue

    # Show or save
    output_filename = "overlay_" + os.path.basename(image_path)
    cv2.imwrite(output_filename, img)
    print(f"Result saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay keypoints from JSON onto Image")
    parser.add_argument("json_file", help="Path to the JSON annotation file")
    parser.add_argument("--image", "-i", help="Path to the corresponding image file (optional if filename is in JSON)", default=None)
    
    args = parser.parse_args()
    overlay_keypoints(args.json_file, args.image)
