import cv2
import numpy as np
import json
from ultralytics import YOLO
import os
import argparse
from datetime import datetime
import uuid
import requests
import tempfile

class Config:
    # --- System ---
    EMA_ALPHA = 0.6          # Exponential Moving Average Factor (Lower = Smoother but legged)
    KP_CONF_TH = 0.3         # Keypoint Confidence Threshold
    MAX_MISSING_RATIO = 0.4  # Max allowed missing ratio (if > 40%, analysis fails)
    
    # --- Robust Quantiles ---
    Q_GROUND = 15            # Percentile for defining "Ground Level" of paw
    Q_SCALE  = 20            # Percentile for measuring "Leg Length" scale
    Q_VEL_TH = 35            # Percentile for "Low Velocity" threshold (Stance detection)

    # --- Stance Detection Thresholds ---
    Y_EPS_RATIO = 0.12       # Stance Height Tolerance (Ratio of Height Scale)
    V_RATIO_FALLBACK = 1.0   # Velocity threshold fallback multiplier

    # --- Post Processing ---
    GAP_FILL = 2             # Fill small gaps (frames) in stance
    MIN_RUN  = 4             # Minimum stance duration (frames)

class DogHealthAnalyzer:
    def __init__(self, model_path, output_dir="output"):
        self.model = YOLO(model_path)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Keypoint Mapping (24 Keypoints)
        self.KP_MAP = {
            "FL_PAW": 0, "FL_KNEE": 1, "FL_ELBOW": 2,
            "RL_PAW": 3, "RL_KNEE": 4, "RL_HOCK": 5, 
            "FR_PAW": 6, "FR_KNEE": 7, "FR_ELBOW": 8,
            "RR_PAW": 9, "RR_KNEE": 10, "RR_HOCK": 11,
            "TAIL_START": 12, "TAIL_END": 13,
            "NOSE": 16, "CHIN": 17 
        }
        
        # Skeleton visualization config
        self.SKELETON = [
            (9, 10, 'right'), (10, 11, 'right'), (11, 12, 'right'), # Rear Right (Red)
            (3, 4, 'left'), (4, 5, 'left'), (5, 12, 'left'),       # Rear Left (Blue)
            (12, 13, 'tail'),                                      # Tail (Green)
            (0, 1, 'left'), (1, 2, 'left'),                        # Front Left (Blue)
            (6, 7, 'right'), (7, 8, 'right'),                      # Front Right (Red)
            (16, 17, 'head')                                       # Head (Yellow)
        ]
        self.COLORS = {
            'right': (0, 0, 255), 'left': (255, 0, 0),
            'tail': (0, 255, 0), 'head': (0, 255, 255)
        }

    # --------------------------------------------------------------------------
    # 1. Main Pipeline
    # --------------------------------------------------------------------------
    def analyze_video(self, video_source, dog_id=123, analysis_id=None):
        if analysis_id is None:
            analysis_id = str(uuid.uuid4())
        
        print(f"Starting MVP 2.0 Analysis [{analysis_id}]...")

        # 1. Video Setup
        video_path, temp_file = self._resolve_video_source(video_source)
        if not video_path:
            return self._error(analysis_id, dog_id, "VIDEO_DOWNLOAD_ERROR")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._cleanup(temp_file)
            return self._error(analysis_id, dog_id, "VIDEO_OPEN_ERROR")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output Setup
        overlay_filename = f"overlay_{analysis_id}.mp4"
        overlay_path = os.path.join(self.output_dir, overlay_filename)
        out = cv2.VideoWriter(overlay_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))

        # 2. Data Collection Loop
        # We collect BOTH X and Y now for robust analysis
        raw_data = {
            "timestamps": [],
            # Paws
            "rl_paw": {"x": [], "y": []}, "rr_paw": {"x": [], "y": []},
            "fl_paw": {"x": [], "y": []}, "fr_paw": {"x": [], "y": []},
            # Knees
            "rl_knee": {"x": [], "y": []}, "rr_knee": {"x": [], "y": []},
            # Tail
            "tail": {"x": [], "y": []},
            # Angles
            "angle_l": [], "angle_r": []
        }

        frame_idx = 0
        last_valid_kpts = None # For display smoothing

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Safety Limit
            if frame_idx > Config.MAX_PROCESS_FRAMES: break

            # Inference
            results = self.model(frame, verbose=False)
            
            # --- Data Validation & Collection ---
            has_detection = False
            kpts = None

            if len(results[0].keypoints) > 0:
                kpts = results[0].keypoints.data[0].cpu().numpy() # (24, 3)
                has_detection = True
                last_valid_kpts = kpts # Store for flicker-free display

            # Draw (Visuals: use last valid if current missing to prevent flicker)
            display_kpts = kpts if has_detection else last_valid_kpts
            if display_kpts is not None:
                self._draw_skeleton(frame, display_kpts)
            out.write(frame)

            # Store Data (Analysis: Store NaN if missing to preserve time-integrity)
            raw_data["timestamps"].append(frame_idx / fps)
            
            def _store_pt(name, kidx):
                if has_detection and kpts[kidx][2] > Config.KP_CONF_TH:
                    raw_data[name]["x"].append(kpts[kidx][0])
                    raw_data[name]["y"].append(kpts[kidx][1])
                else:
                    raw_data[name]["x"].append(np.nan)
                    raw_data[name]["y"].append(np.nan)

            _store_pt("rl_paw", self.KP_MAP["RL_PAW"])
            _store_pt("rr_paw", self.KP_MAP["RR_PAW"])
            _store_pt("fl_paw", self.KP_MAP["FL_PAW"])
            _store_pt("fr_paw", self.KP_MAP["FR_PAW"])
            
            _store_pt("rl_knee", self.KP_MAP["RL_KNEE"])
            _store_pt("rr_knee", self.KP_MAP["RR_KNEE"])
            
            _store_pt("tail", self.KP_MAP["TAIL_START"])

            # Store Angles (Mobility)
            if has_detection:
                ang_l = self._calculate_virtual_angle(kpts, "left")
                ang_r = self._calculate_virtual_angle(kpts, "right")
                raw_data["angle_l"].append(ang_l)
                raw_data["angle_r"].append(ang_r)
            else:
                raw_data["angle_l"].append(np.nan)
                raw_data["angle_r"].append(np.nan)

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Processing frame {frame_idx}/{total_frames}...")

        cap.release()
        out.release()
        self._cleanup(temp_file)

        # 3. MVP 2.0 Analysis Pipeline
        try:
            report = self._analyze_metrics(raw_data, fps, analysis_id, dog_id)
            report["artifacts"]["keypoint_overlay_video_url"] = overlay_filename
            
            output_json = os.path.join(self.output_dir, f"analysis_{analysis_id}.json")
            with open(output_json, "w") as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
                
            print(f"Success! Report: {output_json}")
            return report
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._error(analysis_id, dog_id, "METRIC_CALCULATION_ERROR")

    # --------------------------------------------------------------------------
    # 2. Logic: Preprocessing & Event Detection
    # --------------------------------------------------------------------------
    def _preprocess_series(self, data_dict, tail_x, tail_y):
        """
        Input: Raw X/Y list with NaNs
        Output: Interpolated, Smoothed, Relative Coords, Velocity
        """
        # Convert to arrays
        x = np.array(data_dict["x"], dtype=float)
        y = np.array(data_dict["y"], dtype=float)

        # Interpolation & EMA Smoothing
        x_smooth = self._ema(self._naninterp(x), Config.EMA_ALPHA)
        y_smooth = self._ema(self._naninterp(y), Config.EMA_ALPHA)
        
        # Relative Coordinates (Paw - Tail) -> Cancels Camera Move
        rel_x = x_smooth - tail_x
        rel_y = y_smooth - tail_y
        
        # Velocity (Euclidean 2D Speed of Relative Coords)
        # Using relative velocity is better if camera follows the dog
        dx = np.diff(rel_x, prepend=rel_x[0])
        dy = np.diff(rel_y, prepend=rel_y[0])
        vel = np.sqrt(dx**2 + dy**2)
        
        # Ground Level Estimation (Quantile 15 of Rel Y)
        # "Lowest point" usually means Max Y in image coords, but Rel Y can correspond
        # to distance from body. We use Q15 assuming typical stance distribution.
        # Actually, "Ground" is usually where Y is MAX (bottom of image).
        # But Rel_Y = Paw - Tail. Paw is below tail => Rel_Y is Positive large.
        # Stance is when Rel_Y is MAX (Points furthest down). 
        # Wait, Image Y increases downwards. Paw Y > Tail Y. Rel_Y is Positive.
        # Max Rel_Y = Furthest from body = Touching ground.
        # So "Ground" should be near Q85 or Q90?
        # Let's rely on robust Stance detection: Ground is "Furthest from body".
        # Let's use Q85 for Image Coordinates logic.
        # NO, user spec said: "ground = q15(rel_y)". 
        # Let's check: If Y is 0 at top. Tail=100, Paw=200. Rel=100.
        # If Paw lifts, Y=150. Rel=50.
        # So Ground (Contact) is High Rel_Y. Lift is Low Rel_Y.
        # The user proposal assumed standard math coords (Y up). Image Y is down.
        # We need to Flip or Adapt. 
        # Safest: Use Abs(Rel_Y - Ground). Ground is Quantile near contact.
        # Contact = Max Y (Bottom). So Ground is Q85/Q90.
        # Let's auto-detect ground convention: The "Cluster" of points furthest away.
        # For simplicity, let's just use Q85 as ground ref for Image Coords.
        ground_lvl = np.nanpercentile(rel_y, 85) 
        
        # Height Scale (Leg Length Estimate)
        # Median of absolute displacement
        scale = np.nanmedian(np.abs(rel_y))
        
        return {
            "rel_y": rel_y,
            "vel": vel,
            "ground": ground_lvl,
            "scale": scale,
            "missing_ratio": np.isnan(y).mean()
        }

    def _naninterp(self, arr):
        mask = np.isnan(arr)
        if mask.all(): return arr
        arr = arr.copy()
        arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
        return arr

    def _ema(self, arr, alpha):
        if np.isnan(arr).all(): return arr
        out = np.empty_like(arr)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
        return out

    def _detect_stance(self, prep_data, height_scale):
        # Unpack
        rel_y = prep_data["rel_y"]
        vel = prep_data["vel"]
        ground = prep_data["ground"] # This is roughly "Max Rel Y" (Bottom)
        
        # Dynamic Thresholds
        # 1. Height Condition: |Rel_Y - Ground| < Epsilon
        # Since Ground is Q85, Stance is around there.
        # abs(Rel_Y - Ground) < 10% of Leg Length
        y_eps = Config.Y_EPS_RATIO * height_scale
        cond_h = np.abs(rel_y - ground) < y_eps
        
        # 2. Velocity Condition: Vel < Low_Threshold
        # Stance = Foot planted = Zero relative velocity
        v_th = np.nanpercentile(vel, Config.Q_VEL_TH)
        cond_v = vel < v_th
        
        # Combine
        stance = cond_h & cond_v
        
        # 3. Post-Process (Morphological close/open)
        # Fill small gaps
        stance = self._fill_gaps(stance, Config.GAP_FILL)
        # Remove short runs
        stance = self._remove_short_runs(stance, Config.MIN_RUN)
        
        return stance

    def _fill_gaps(self, mask, max_gap):
        out = mask.copy()
        for i in range(len(mask)):
            if not out[i]: # Gap start?
                # Look ahead
                j = i + 1
                while j < len(mask) and not out[j]:
                    j += 1
                if j < len(mask) and (j - i) <= max_gap:
                    out[i:j] = True # Fill
        return out

    def _remove_short_runs(self, mask, min_run):
        out = mask.copy()
        cnt = 0
        for i in range(len(mask)):
            if out[i]:
                cnt += 1
            else:
                if 0 < cnt < min_run:
                    out[i-cnt:i] = False # Kill short run
                cnt = 0
        if 0 < cnt < min_run: # Tail check
            out[len(mask)-cnt:] = False
        return out

    # --------------------------------------------------------------------------
    # 3. Metrics Calculation (New Logic)
    # --------------------------------------------------------------------------
    def _analyze_metrics(self, raw, fps, analysis_id, dog_id):
        # A. Preprocessing (Root Ref=Tail)
        tail_x = self._ema(self._naninterp(np.array(raw["tail"]["x"])), Config.EMA_ALPHA)
        tail_y = self._ema(self._naninterp(np.array(raw["tail"]["y"])), Config.EMA_ALPHA)
        
        paws = ["rl", "rr", "fl", "fr"]
        prep = {}
        for p in paws:
            prep[p] = self._preprocess_series(raw[f"{p}_paw"], tail_x, tail_y)

        # Quality Gate
        max_miss = max(p[1]["missing_ratio"] for p in prep.items())
        if max_miss > Config.MAX_MISSING_RATIO:
            return self._error(analysis_id, dog_id, "LOW_QUALITY_VIDEO_DETECTED")

        # Global Scale (Leg Length) = Median of all paws' scales
        height_scale = np.median([prep[p]["scale"] for p in paws])

        # B. Event Detection (Stance)
        stance = {}
        for p in paws:
            stance[p] = self._detect_stance(prep[p], height_scale)

        # C. Calculate Metrics
        
        # 1. RHYTHM: Rear Stance Asymmetry + Step Difference
        # Stance Ratio = Stance Frames / Total Frames
        sr_rl = stance["rl"].mean()
        sr_rr = stance["rr"].mean()
        asym_rear = abs(sr_rl - sr_rr)
        
        # Count Steps (Stance Blocks)
        def count_steps(mask):
            return np.sum((np.diff(mask.astype(int)) == 1))
        
        step_l = count_steps(stance["rl"])
        step_r = count_steps(stance["rr"])
        step_diff = abs(step_l - step_r)
        
        # Rhythm Score: 100 - (Asym*Penalty) - (StepDiff*Penalty)
        # Asym 0.1 (10%) -> -20 pts. StepDiff 1 -> -5 pts.
        rhythm_score = int(max(0, 100 - (asym_rear * 200) - (step_diff * 5)))
        rhythm_level = "consistent" if rhythm_score > 75 else "irregular"
        rhythm_desc = "발걸음 규칙성이 좋습니다." if rhythm_level == "consistent" else f"좌우 박자 불균형이 감지됩니다. (비대칭:{asym_rear:.2f})"

        # 2. BALANCE: Min/Max Ratio of Stance Ratios
        # Uses all 4 paws if front detected well, else fallback rear
        valid_ratios = [sr_rl, sr_rr]
        if prep["fl"]["missing_ratio"] < 0.2 and prep["fr"]["missing_ratio"] < 0.2:
            valid_ratios += [stance["fl"].mean(), stance["fr"].mean()]
            
        bal_min, bal_max = min(valid_ratios), max(valid_ratios)
        bal_ratio = bal_min / (bal_max + 1e-6)
        balance_score = int(bal_ratio * 100)
        
        if balance_score > 80: bal_lvl, bal_desc = "good", "체중 분산이 고릅니다."
        elif balance_score > 60: bal_lvl, bal_desc = "fair", "체중 균형이 양호합니다."
        else: bal_lvl, bal_desc = "poor", "특정 다리에 체중이 쏠려 있습니다."

        # 3. MOBILITY: ROM Difference (Left vs Right)
        # Using Virtual Angles (already collected)
        ang_l = np.array(raw["angle_l"]); ang_l = ang_l[~np.isnan(ang_l)]
        ang_r = np.array(raw["angle_r"]); ang_r = ang_r[~np.isnan(ang_r)]
        
        rom_l = (np.percentile(ang_l, 95) - np.percentile(ang_l, 5)) if len(ang_l) > 10 else 0
        rom_r = (np.percentile(ang_r, 95) - np.percentile(ang_r, 5)) if len(ang_r) > 10 else 0
        
        avg_rom = (rom_l + rom_r) / 2
        rom_diff = abs(rom_l - rom_r)
        
        # Score: Base on Avg ROM with penalty for Diff
        # 45 deg = 100 pt base. Diff > 10 deg penalizes heavily.
        base_mob = min(100, (avg_rom / 45) * 100)
        mob_penalty = (rom_diff / 15) * 40
        mobility_score = int(max(0, base_mob - mob_penalty))
        
        if mobility_score > 70: mob_lvl, mob_desc = "normal", "관절 가동 범위 정상."
        elif mobility_score > 40: mob_lvl, mob_desc = "stiff", "관절 움직임이 뻣뻣합니다."
        else: mob_lvl, mob_desc = "stiff", "관절 가동성이 매우 떨어집니다."

        # 4. STABILITY: Tail-Knee Distance Stability
        # Internal distance is invariant to camera shift
        # Needs Knee X/Y smoothed
        def get_dist_std(k_name):
            kx = self._ema(self._naninterp(np.array(raw[k_name]["x"])), Config.EMA_ALPHA)
            ky = self._ema(self._naninterp(np.array(raw[k_name]["y"])), Config.EMA_ALPHA)
            dist = np.sqrt((kx - tail_x)**2 + (ky - tail_y)**2)
            return np.std(dist)
            
        std_l = get_dist_std("rl_knee")
        std_r = get_dist_std("rr_knee")
        avg_std = (std_l + std_r) / 2
        
        # Normalize by scale (Size invariant)
        norm_stab = avg_std / (height_scale + 1e-6)
        # Usually < 0.1 is good. > 0.3 is bad.
        stability_score = int(max(0, 100 - (norm_stab * 300)))
        
        stab_lvl = "stable" if stability_score > 70 else "unstable"
        stab_desc = "보행 중심이 안정적입니다." if stab_lvl == "stable" else "보행 시 몸통 흔들림이 있습니다."

        # 5. PATELLA: Logic Trigger
        # Skipping signs: High Rhythm Asym OR Low Rhythm Score
        patella_score = 100
        patella_lvl = "low"
        patella_desc = "슬개골 탈구 의심 신호 없음."
        
        if rhythm_score < 60 or asym_rear > 0.2:
            patella_score = 50
            patella_lvl = "high"
            patella_desc = "다리 절음(Skipping) 현상이 의심됩니다."
        elif balance_score < 60:
            patella_score = 75
            patella_lvl = "medium"
            patella_desc = "보행 불균형으로 주의가 필요합니다."

        # Result Construction
        total_score = int((rhythm_score + balance_score + mobility_score + stability_score + patella_score) / 5)
        
        return {
            "analysis_id": analysis_id, "dog_id": dog_id,
            "analyze_at": datetime.now().isoformat(),
            "result": {
                "overall_score": total_score,
                "overall_risk_level": patella_lvl,
                "summary": f"종합점수 {total_score}점. {patella_desc}"
            },
            "metrics": {
                "gait_rhythm": {"score": rhythm_score, "level": rhythm_level, "description": rhythm_desc},
                "gait_balance": {"score": balance_score, "level": bal_lvl, "description": bal_desc},
                "knee_mobility": {"score": mobility_score, "level": mob_lvl, "description": mob_desc},
                "gait_stability": {"score": stability_score, "level": stab_lvl, "description": stab_desc},
                "patella_risk_signal": {"score": patella_score, "level": patella_lvl, "description": patella_desc}
            },
            "artifacts": {},
            "debug": {
                "steps": [int(step_l), int(step_r)],
                "asym": float(asym_rear),
                "scale": float(height_scale), 
                "rom_diff": float(rom_diff)
            },
            "error_code": None
        }

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------
    def _draw_skeleton(self, frame, kpts):
        visual_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,16,17] 
        for i in visual_indices:
            if i < len(kpts):
                x, y, conf = kpts[i]
                if conf > Config.KP_CONF_TH:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
        for idx1, idx2, color_key in self.SKELETON:
            if idx1 < len(kpts) and idx2 < len(kpts):
                pt1, pt2 = kpts[idx1], kpts[idx2]
                if pt1[2] > Config.KP_CONF_TH and pt2[2] > Config.KP_CONF_TH:
                    color = self.COLORS.get(color_key, (255, 255, 255))
                    cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 2)

    def _calculate_virtual_angle(self, kpts, side):
        # Tail - Knee - Paw
        idx_map = {"left": [12, 4, 3], "right": [12, 10, 9]}
        idxs = idx_map.get(side)
        if not idxs: return np.nan
        
        p1, p2, p3 = kpts[idxs[0]], kpts[idxs[1]], kpts[idxs[2]]
        if p1[2] < Config.KP_CONF_TH or p2[2] < Config.KP_CONF_TH or p3[2] < Config.KP_CONF_TH:
            return np.nan
            
        v1 = p1[:2] - p2[:2]
        v2 = p3[:2] - p2[:2]
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
        return np.degrees(np.arccos(np.clip(dot/norm, -1.0, 1.0)))

    def _resolve_video_source(self, source):
        if source.startswith("http"):
            try:
                r = requests.get(source, stream=True)
                r.raise_for_status()
                fd, path = tempfile.mkstemp(suffix=".mp4")
                with os.fdopen(fd, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                return path, path
            except:
                return None, None
        return source, None

    def _cleanup(self, temp_path):
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    def _error(self, aid, did, code):
        return {
            "analysis_id": aid, "dog_id": did, 
            "analyze_at": datetime.now().isoformat(),
            "error_code": code, "result": None
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model", type=str, default="models/best_26m.pt")
    args = parser.parse_args()
    
    analyzer = DogHealthAnalyzer(args.model)
    res = analyzer.analyze_video(args.video)
    print(json.dumps(res, indent=4, ensure_ascii=False))
