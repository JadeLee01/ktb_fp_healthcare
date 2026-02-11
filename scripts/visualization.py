import cv2
import numpy as np

# Keypoint Map from features.py (for reference)
# KP_MAP = {
#     "FL_PAW": 0, "FL_KNEE": 1, "FL_ELBOW": 2,
#     "RL_PAW": 3, "RL_KNEE": 4, "RL_ELBOW": 5,
#     "FR_PAW": 6, "FR_KNEE": 7, "FR_ELBOW": 8,
#     "RR_PAW": 9, "RR_KNEE": 10, "RR_ELBOW": 11,
#     "TAIL_START": 12, "TAIL_END": 13,
#     "L_EAR_BASE": 14, "R_EAR_BASE": 15, "NOSE": 16, "CHIN": 17,
#     "L_EAR_TIP": 18, "R_EAR_TIP": 19, "L_EYE": 20, "R_EYE": 21,
#     "WITHERS": 22, "THROAT": 23
# }

# Skeleton connections (Start Index, End Index)
# Improved for anatomical correctness
# Skeleton connections (Start Index, End Index)
# Anatomically simplified structure
SKELETON = [
    # 1. HEAD AREA
    (16, 17), # Nose -> Chin
    (16, 20), (20, 21), (21, 16), # Nose -> L.Eye -> R.Eye -> Nose (Triangle)
    (20, 14), (21, 15), # Eyes -> Ear Bases
    (14, 18), (15, 19), # Ear Bases -> Ear Tips
    
    # 2. MAIN BODY AXIS (Head -> Neck -> Spine -> Tail)
    (17, 23), # Chin -> Throat
    (23, 22), # Throat -> Withers (Shoulder/Spine start)
    (22, 12), # Withers -> Tail Start (Spine/Back)
    (12, 13), # Tail Start -> Tail End
    
    # 3. LEGS (Connected to Main Body Axis)
    # Front Legs -> Connect to Withers (22)
    (22, 2), (2, 1), (1, 0),   # Withers -> L.Elbow -> L.Knee -> L.Paw
    (22, 8), (8, 7), (7, 6),   # Withers -> R.Elbow -> R.Knee -> R.Paw
    
    # Rear Legs -> Connect to Tail Start (12)
    (12, 5), (5, 4), (4, 3),   # Tail Start -> L.Elbow(Hock) -> L.Knee -> L.Paw
    (12, 11), (11, 10), (10, 9) # Tail Start -> R.Elbow(Hock) -> R.Knee -> R.Paw
]

COLORS = {
    "kpt": (0, 0, 255),    # Red
    "skeleton": (0, 255, 0), # Green
    "bbox": (255, 0, 0)    # Blue
}

class KeypointSmoother:
    def __init__(self, alpha=0.5):
        """
        Exponential Moving Average (EMA) smoother.
        alpha: Smoothing factor (0 < alpha <= 1). Lower = smoother but more lag.
        """
        self.alpha = alpha
        self.prev_kpts = None

    def update(self, current_kpts):
        """
        current_kpts: (N, 2/3) numpy array
        Returns: Smoothed keypoints
        """
        if self.prev_kpts is None:
            self.prev_kpts = current_kpts
            return current_kpts
        
        # Determine valid keypoints (not 0,0) to update
        # If current is 0,0 (missing), keep prev or set to 0? YOLO returns 0,0 for missing.
        # We should probably not smooth towards 0,0 if it's missing.
        
        # Mask for valid current keypoints (assuming 0,0 is invalid)
        valid_mask = (current_kpts[:, 0] != 0) | (current_kpts[:, 1] != 0)
        
        # Only smooth valid keypoints
        # smoothed = alpha * curr + (1-alpha) * prev
        smoothed = self.prev_kpts.copy()
        smoothed[valid_mask] = self.alpha * current_kpts[valid_mask] + (1 - self.alpha) * self.prev_kpts[valid_mask]
        
        # Update prev only for valid ones, or decay invalid ones?
        # Let's keep valid logic simple: update state with result
        self.prev_kpts = smoothed
        return smoothed

def draw_overlay(frame, keypoints, keypoint_confs=None, bbox=None, kpt_radius=5, line_thickness=2, conf_th=0.5):
    """
    Draws keypoints, skeleton, and bbox on the frame.
    keypoints: (N, 2) or (N, 3)
    keypoint_confs: (N,) confidence scores for each keypoint
    bbox: [x1, y1, x2, y2]
    conf_th: Confidence threshold for drawing
    """
    overlay = frame.copy()
    
    # Draw BBox
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLORS["bbox"], 2)

    # Draw Skeleton
    # Draw Skeleton with Logic for Missing Points
    for i, j in SKELETON:
        if i < len(keypoints) and j < len(keypoints):
            # Check confidence if available
            if keypoint_confs is not None:
                if keypoint_confs[i] < conf_th or keypoint_confs[j] < conf_th:
                    continue # Skip low confidence connections

            pt1 = keypoints[i]
            pt2 = keypoints[j]
            
            # Check if valid (not 0,0 and confidence > 0.5 ideally, but raw kpts might not have conf here)
            # Assuming keypoints are (x,y) or (x,y,conf). If (x,y), 0,0 is invalid.
            if (pt1[0] > 1 and pt1[1] > 1) and (pt2[0] > 1 and pt2[1] > 1):
                x1, y1 = int(pt1[0]), int(pt1[1])
                x2, y2 = int(pt2[0]), int(pt2[1])
                cv2.line(overlay, (x1, y1), (x2, y2), COLORS["skeleton"], line_thickness)
            
            # SPECIAL CASE: Bridge gap if Throat (23) is missing logic
            # If we are trying to connect Chin(17)->Throat(23) OR Throat(23)->Withers(22) AND Throat is missing
            # Connect Chin(17) -> Withers(22) directly
            if (i == 17 and j == 23) or (i == 23 and j == 22):
                 # Check if Throat (23) is invalid
                 is_throat_missing = False
                 if keypoint_confs is not None:
                     if keypoint_confs[23] < conf_th:
                         is_throat_missing = True
                 else:
                     kp23 = keypoints[23]
                     if kp23[0] <= 1 and kp23[1] <= 1:
                         is_throat_missing = True

                 if is_throat_missing:
                     # Throat is missing. Draw logical bridge: Chin(17) -> Withers(22)
                     # Check confidence for Chin and Withers
                     if keypoint_confs is not None:
                         if keypoint_confs[17] < conf_th or keypoint_confs[22] < conf_th:
                             continue
                             
                     kp17 = keypoints[17]
                     kp22 = keypoints[22]
                     if (kp17[0] > 1 and kp17[1] > 1) and (kp22[0] > 1 and kp22[1] > 1):
                         x1, y1 = int(kp17[0]), int(kp17[1])
                         x2, y2 = int(kp22[0]), int(kp22[1])
                         # Draw a slightly thinner line to indicate inferred connection
                         cv2.line(overlay, (x1, y1), (x2, y2), COLORS["skeleton"], 1)

    # Draw Keypoints with Indices (Debugging)
    for i, kp in enumerate(keypoints):
        # Check confidence
        if keypoint_confs is not None and keypoint_confs[i] < conf_th:
            continue
            
        x, y = int(kp[0]), int(kp[1])
        if x > 1 and y > 1:
            cv2.circle(overlay, (x, y), kpt_radius, COLORS["kpt"], -1) # Filled circle
            # Draw index number
            cv2.putText(overlay, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return overlay
