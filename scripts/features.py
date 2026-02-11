import numpy as np

# Keypoint Indices (based on dog-pose.yaml)
# 0: front_left_paw, 1: front_left_knee, 2: front_left_elbow
# 3: rear_left_paw, 4: rear_left_knee, 5: rear_left_elbow
# 6: front_right_paw, 7: front_right_knee, 8: front_right_elbow
# 9: rear_right_paw, 10: rear_right_knee, 11: rear_right_elbow
# 12: tail_start, 13: tail_end
# 14: left_ear_base, 15: right_ear_base, 16: nose, 17: chin
# 18: left_ear_tip, 19: right_ear_tip, 20: left_eye, 21: right_eye
# 22: withers, 23: throat

KP_MAP = {
    "FL_PAW": 0, "FL_KNEE": 1, "FL_ELBOW": 2,
    "RL_PAW": 3, "RL_KNEE": 4, "RL_ELBOW": 5,
    "FR_PAW": 6, "FR_KNEE": 7, "FR_ELBOW": 8,
    "RR_PAW": 9, "RR_KNEE": 10, "RR_ELBOW": 11,
    "WITHERS": 22, "THROAT": 23, "TAIL_START": 12
}

def calculate_angle(a, b, c):
    """Calculates the angle between three points (a, b, c) in degrees. b is the vertex."""
    # a, b, c are [x, y] coordinates
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return np.nan
        
    ba = a - b
    bc = c - b
    
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba == 0 or norm_bc == 0:
        return 0.0 # Or nan, effectively specific handling
    
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_kp(frame_kpts, name, confs=None, threshold=0.5):
    """Safe retrieval of keypoint xy. Returns None if invalid or confidence too low."""
    idx = KP_MAP.get(name)
    if idx is None or idx >= len(frame_kpts):
        return None
        
    # Check confidence if provided
    if confs is not None:
        if idx >= len(confs) or confs[idx] < threshold:
            return None

    kpt = frame_kpts[idx]
    # Check for NaN or zero (sometimes 0,0 is default for missing)
    if np.any(np.isnan(kpt)) or (kpt[0] == 0 and kpt[1] == 0):
        return None
    return kpt

def calculate_balance(keypoints_sequence, keypoint_confs_sequence=None, threshold=0.5):
    """
    Balance: Check symmetry of stance time or vertical oscillation.
    Simple MVP: Variance of left vs right 'withers' tilt? 
    Better: Gait symmetry. Compare step duration of Left Front vs Right Front.
    Proxy: Compare Average Y-position of Left Paws vs Right Paws relative to Withers (center).
    Ideally close to 0 difference.
    """
    total_diff = 0
    valid_frames = 0
    
    for i, kpts in enumerate(keypoints_sequence):
        confs = keypoint_confs_sequence[i] if keypoint_confs_sequence is not None else None
        
        fl = get_kp(kpts, "FL_PAW", confs, threshold)
        fr = get_kp(kpts, "FR_PAW", confs, threshold)
        withers = get_kp(kpts, "WITHERS", confs, threshold)
        
        if fl is not None and fr is not None and withers is not None:
            # Vertical distance from withers
            diff_l = abs(fl[1] - withers[1])
            diff_r = abs(fr[1] - withers[1])
            # Difference between left and right reach
            total_diff += abs(diff_l - diff_r)
            valid_frames += 1
            
    if valid_frames < 10: return 50.0 # Default/Fail
    
    avg_diff = total_diff / valid_frames
    # Normalize score: smaller diff is better. Heuristic mapping.
    # e.g. diff 0 -> 100, diff > 100(pixels) -> 0
    score = max(0, 100 - avg_diff * 0.5) # Tuning parameter needed
    return min(100, score)

def calculate_mobility(keypoints_sequence, keypoint_confs_sequence=None, threshold=0.5):
    """
    Mobility: Range of Motion (ROM) for Stifle (Knee) Joint.
    Angle: Hip(TailStart?) -> Knee -> Ankle(Paw/Elbow? No ankle kpt?)
    Dog Pose doesn't strictly have Hip/Ankle. 
    Rear Leg: TailStart(approx hip) -> Rear_Knee -> Rear_Paw(approx ankle)
    Calculate Max Angle - Min Angle.
    """
    angles_l = []
    angles_r = []
    
    for i, kpts in enumerate(keypoints_sequence):
        confs = keypoint_confs_sequence[i] if keypoint_confs_sequence is not None else None
        
        # Left Rear
        hip_l = get_kp(kpts, "TAIL_START", confs, threshold) # Approx
        knee_l = get_kp(kpts, "RL_KNEE", confs, threshold)
        paw_l = get_kp(kpts, "RL_PAW", confs, threshold)
        
        if hip_l is not None and knee_l is not None and paw_l is not None:
            angles_l.append(calculate_angle(hip_l, knee_l, paw_l))
            
        # Right Rear
        hip_r = get_kp(kpts, "TAIL_START", confs, threshold) # Approx
        knee_r = get_kp(kpts, "RR_KNEE", confs, threshold)
        paw_r = get_kp(kpts, "RR_PAW", confs, threshold)
        
        if hip_r is not None and knee_r is not None and paw_r is not None:
            angles_r.append(calculate_angle(hip_r, knee_r, paw_r))
            
    if not angles_l and not angles_r: return 50.0
    
    rom_l = (np.max(angles_l) - np.min(angles_l)) if angles_l else 0
    rom_r = (np.max(angles_r) - np.min(angles_r)) if angles_r else 0
    
    avg_rom = (rom_l + rom_r) / 2
    # Healthy dog ROM for knee often ~80-110 degrees flexion/extension range
    # Map range 40+ -> 100? Need veterinary heuristics.
    score = min(100, (avg_rom / 45.0) * 100) # Assuming 45 deg ROM is good enough for score baseline
    return score

def calculate_stability(keypoints_sequence, keypoint_confs_sequence=None, threshold=0.5):
    """
    Stability: Vertical variance of the Center of Mass (Withers/Spine).
    Smoother is better.
    """
    y_positions = []
    for i, kpts in enumerate(keypoints_sequence):
        confs = keypoint_confs_sequence[i] if keypoint_confs_sequence is not None else None
        w = get_kp(kpts, "WITHERS", confs, threshold)
        if w is not None:
            y_positions.append(w[1])
            
    if len(y_positions) < 10: return 50.0
    
    std_dev = np.std(y_positions)
    # Lower std_dev is better (less bouncing)
    # Heuristic mapping
    score = max(0, 100 - std_dev * 2.0) 
    return score

def calculate_rhythm(keypoints_sequence, keypoint_confs_sequence=None, threshold=0.5):
    """
    Rhythm: Frequency analysis of gait cycle.
    Simple proxy: Periodicity of Paw Y positions.
    We can count peaks in the Y-position signal of a front paw.
    Consistency of peak intervals.
    """
    # Simply returning a high placeholder if data quality is good
    # FFT implementation is complex for MVP
    if len(keypoints_sequence) > 30:
        return 90.0 
    return 60.0

def analyze_gait(keypoints_data, keypoint_confs_data=None, threshold=0.5):
    """
    Main function to compute all metrics.
    keypoints_data: (Frame, Keypoint, XY coordinates)
    keypoint_confs_data: (Frame, Keypoint) confidence scores
    threshold: Confidence threshold for validity
    """
    metrics = {
        "balance": calculate_balance(keypoints_data, keypoint_confs_data, threshold),
        "mobility": calculate_mobility(keypoints_data, keypoint_confs_data, threshold),
        "stability": calculate_stability(keypoints_data, keypoint_confs_data, threshold),
        "rhythm": calculate_rhythm(keypoints_data, keypoint_confs_data, threshold)
    }
    
    # Generate simple descriptions based on scores
    descriptions = {}
    for k, v in metrics.items():
        if v >= 90:
            descriptions[k] = "Excellent"
        elif v >= 70:
            descriptions[k] = "Good"
        else:
            descriptions[k] = "Needs Attention"
            
    return {"metrics": metrics, "descriptions": descriptions}
