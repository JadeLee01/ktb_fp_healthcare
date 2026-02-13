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
    # -------------------------
    # Runtime / Safety
    # -------------------------
    MAX_PROCESS_FRAMES = 600
    
    # -------------------------
    # Keypoint confidence gate
    # -------------------------
    KP_CONF_TH = 0.30

    # -------------------------
    # Smoothing (EMA)
    # -------------------------
    EMA_ALPHA = 0.60

    # -------------------------
    # Robust Quantiles for Dynamic Ground Detection
    # -------------------------
    Q_LEVEL_LOW = 15     # Candidate 1 for Ground Level
    Q_LEVEL_HIGH = 85    # Candidate 2 for Ground Level
    Q_SCALE = 20         # Percentile for leg length scale (when standing)
    Q_VEL_TH = 35        # Percentile for low-velocity threshold

    # -------------------------
    # Stance constraints
    # -------------------------
    Y_EPS_RATIO = 0.10   # Stance vertical tolerance
    MIN_HEIGHT_SCALE_PX = 5.0
    STANCE_RATIO_MIN = 0.05
    STANCE_RATIO_MAX = 0.85

    # -------------------------
    # Stance post-processing
    # -------------------------
    GAP_FILL = 2
    MIN_RUN = 5

    # -------------------------
    # Quality gate
    # -------------------------
    MAX_MISSING_RATIO = 0.40

    # -------------------------
    # Scoring Weights
    # -------------------------
    RHYTHM_ASYM_WEIGHT = 180.0
    RHYTHM_STEP_WEIGHT = 10.0
    STABILITY_WEIGHT = 300.0

    # Patella Triggers
    PATELLA_ASYM_HIGH = 0.18
    PATELLA_STEP_DIFF_HIGH = 3
    PATELLA_RHYTHM_LOW = 60
    PATELLA_BALANCE_MED = 65

class DogHealthAnalyzer:
    def __init__(self, model_path, output_dir="output"):
        self.model = YOLO(model_path)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.KP_MAP = {
            "FL_PAW": 0, "FL_KNEE": 1, "FL_ELBOW": 2,
            "RL_PAW": 3, "RL_KNEE": 4, "RL_HOCK": 5, 
            "FR_PAW": 6, "FR_KNEE": 7, "FR_ELBOW": 8,
            "RR_PAW": 9, "RR_KNEE": 10, "RR_HOCK": 11,
            "TAIL_START": 12, "TAIL_END": 13,
            "NOSE": 16, "CHIN": 17 
        }
        
        self.SKELETON = [
            (9, 10, 'right'), (10, 11, 'right'), (11, 12, 'right'),
            (3, 4, 'left'), (4, 5, 'left'), (5, 12, 'left'),
            (12, 13, 'tail'),
            (0, 1, 'left'), (1, 2, 'left'),
            (6, 7, 'right'), (7, 8, 'right'),
            (16, 17, 'head')
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
        
        print(f"Starting Refined MVP 2.0 Analysis [{analysis_id}] with Detail Descriptions...")

        video_path, temp_file = self._resolve_video_source(video_source)
        if not video_path:
            return self._error(analysis_id, dog_id, "VIDEO_DOWNLOAD_ERROR")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._cleanup(temp_file)
            return self._error(analysis_id, dog_id, "VIDEO_OPEN_ERROR")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        overlay_filename = f"overlay_{analysis_id}.mp4"
        overlay_path = os.path.join(self.output_dir, overlay_filename)
        out = cv2.VideoWriter(overlay_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))

        ts = {
            "timestamps": [],
            "tail_x": [], "tail_y": [],
            "rl_paw_x": [], "rl_paw_y": [], "rr_paw_x": [], "rr_paw_y": [], 
            "fl_paw_x": [], "fl_paw_y": [], "fr_paw_x": [], "fr_paw_y": [],
            "rl_knee_x": [], "rl_knee_y": [], "rr_knee_x": [], "rr_knee_y": [],
            "angle_l": [], "angle_r": []
        }

        frame_idx = 0
        smoothed_kpts = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_idx >= Config.MAX_PROCESS_FRAMES: break

            ts["timestamps"].append(frame_idx / fps)

            results = self.model(frame, verbose=False)
            
            curr_kpts = None
            if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints) > 0:
                raw_kpts = results[0].keypoints.data[0].cpu().numpy() # (24, 3)
                
                # Robust EMA Smoothing: Only update if Conf > TH
                if smoothed_kpts is None:
                    smoothed_kpts = raw_kpts.copy()
                else:
                    mask = raw_kpts[:, 2] > Config.KP_CONF_TH
                    smoothed_kpts[mask, :2] = (Config.EMA_ALPHA * raw_kpts[mask, :2]) + ((1 - Config.EMA_ALPHA) * smoothed_kpts[mask, :2])
                    smoothed_kpts[:, 2] = raw_kpts[:, 2]
                
                curr_kpts = smoothed_kpts.copy()
                self._draw_skeleton(frame, curr_kpts)

            def _store(name, kidx):
                if curr_kpts is not None and kidx < len(curr_kpts) and curr_kpts[kidx][2] > Config.KP_CONF_TH:
                    ts[f"{name}_x"].append(float(curr_kpts[kidx][0]))
                    ts[f"{name}_y"].append(float(curr_kpts[kidx][1]))
                else:
                    ts[f"{name}_x"].append(np.nan)
                    ts[f"{name}_y"].append(np.nan)

            _store("tail", self.KP_MAP["TAIL_START"])
            _store("rl_paw", self.KP_MAP["RL_PAW"]); _store("rr_paw", self.KP_MAP["RR_PAW"])
            _store("fl_paw", self.KP_MAP["FL_PAW"]); _store("fr_paw", self.KP_MAP["FR_PAW"])
            _store("rl_knee", self.KP_MAP["RL_KNEE"]); _store("rr_knee", self.KP_MAP["RR_KNEE"])
            
            if curr_kpts is not None:
                ang_l = self._calculate_angle(curr_kpts, "left")
                ang_r = self._calculate_angle(curr_kpts, "right")
                ts["angle_l"].append(ang_l); ts["angle_r"].append(ang_r)
            else:
                ts["angle_l"].append(np.nan); ts["angle_r"].append(np.nan)

            out.write(frame)
            frame_idx += 1
            if frame_idx % 30 == 0: print(f"Processing frame {frame_idx}/{total_frames}...")

        cap.release()
        out.release()
        self._cleanup(temp_file)

        try:
            report = self._analyze_metrics(ts, fps, analysis_id, dog_id)
            report["artifacts"]["keypoint_overlay_video_url"] = overlay_filename
            output_json = os.path.join(self.output_dir, f"analysis_{analysis_id}.json")
            with open(output_json, "w") as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
            print(f"Analysis Complete! Report: {output_json}")
            return report
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._error(analysis_id, dog_id, "METRIC_CALCULATION_ERROR")

    # --------------------------------------------------------------------------
    # 2. Logic: Analysis & Processing
    # --------------------------------------------------------------------------
    def _analyze_metrics(self, ts, fps, aid, did):
        if len(ts["timestamps"]) < 10:
            return self._error(aid, did, "NO_DATA_EXTRACTED")
            
        # 1. Quality Gate
        missing = { k: float(np.isnan(ts[f"{k}_y"]).mean()) for k in ["tail", "rl_paw", "rr_paw", "fl_paw", "fr_paw"] }
        if max(missing["tail"], missing["rl_paw"], missing["rr_paw"]) > Config.MAX_MISSING_RATIO:
             return self._error(aid, did, "LOW_QUALITY_VIDEO_DETECTED")
             
        # 2. Prepare Clean Data
        def _get_clean(name):
            x = self._ema(self._naninterp(np.array(ts[f"{name}_x"])))
            y = self._ema(self._naninterp(np.array(ts[f"{name}_y"])))
            return x, y

        tail_x, tail_y = _get_clean("tail")
        paws = ["rl_paw", "rr_paw", "fl_paw", "fr_paw"]
        prep = {}
        
        scales = []
        for p in paws:
            px, py = _get_clean(p)
            rel_x, rel_y = px - tail_x, py - tail_y
            vel = np.sqrt(np.diff(rel_x, prepend=rel_x[0])**2 + np.diff(rel_y, prepend=rel_y[0])**2)
            
            # Ground Candidates
            g15 = self._robust_q(rel_y, Config.Q_LEVEL_LOW)
            g85 = self._robust_q(rel_y, Config.Q_LEVEL_HIGH)
            
            # Scale
            s90_10 = abs(self._robust_q(rel_y, 90) - self._robust_q(rel_y, 10))
            scales.append(s90_10)
            
            prep[p] = { "rel_y": rel_y, "vel": vel, "g_candidates": (g15, g85) }
        
        height_scale = np.median(scales)
        if height_scale < Config.MIN_HEIGHT_SCALE_PX:
             return self._error(aid, did, "LOW_KEYPOINT_QUALITY")

        # 3. Detect Stance
        stance = {}
        debug_info = {}
        eps = Config.Y_EPS_RATIO * height_scale
        
        for p in paws:
            data = prep[p]
            g1, g2 = data["g_candidates"]
            
            def get_ratio(g_val):
                cond_h = np.abs(data["rel_y"] - g_val) < eps
                v_th = self._robust_q(data["vel"], Config.Q_VEL_TH)
                cond_v = data["vel"] < v_th
                st = cond_h & cond_v
                st = self._fill_gaps(st, Config.GAP_FILL)
                st = self._remove_short_runs(st, Config.MIN_RUN)
                return st, st.mean()

            st1, r1 = get_ratio(g1)
            st2, r2 = get_ratio(g2)
            
            valid1 = Config.STANCE_RATIO_MIN <= r1 <= Config.STANCE_RATIO_MAX
            valid2 = Config.STANCE_RATIO_MIN <= r2 <= Config.STANCE_RATIO_MAX
            
            if valid1 and not valid2: final_st, final_ground = st1, g1
            elif valid2 and not valid1: final_st, final_ground = st2, g2
            else: final_st, final_ground = (st1, g1) if abs(r1 - 0.5) < abs(r2 - 0.5) else (st2, g2)
                
            stance[p] = final_st
            debug_info[p] = {"ratio": float(final_st.mean()), "ground": float(final_ground)}

        # 4. Calculate Scores & Descriptions
        
        # [RHYTHM]
        sr_rl, sr_rr = stance["rl_paw"].mean(), stance["rr_paw"].mean()
        asym = abs(sr_rl - sr_rr)
        step_l, step_r = self._count_runs(stance["rl_paw"]), self._count_runs(stance["rr_paw"])
        step_diff = abs(step_l - step_r)
        
        rhythm_score = int(max(0, 100 - (asym * Config.RHYTHM_ASYM_WEIGHT) - (step_diff * Config.RHYTHM_STEP_WEIGHT)))
        rhythm_lvl = "consistent" if rhythm_score > 75 else "irregular"
        
        if rhythm_score >= 90: rhythm_desc = "양쪽 뒷다리의 보행 리듬이 완벽하게 대칭입니다. 매우 건강한 상태입니다."
        elif rhythm_score >= 75: rhythm_desc = "발걸음이 대체로 규칙적이나, 미세한 박자 차이가 있습니다."
        elif rhythm_score >= 50: rhythm_desc = f"보행 리듬이 다소 불규칙합니다. (좌우 비대칭: {asym*100:.1f}%)"
        else: rhythm_desc = "리듬이 크게 무너져 있습니다. 절음(Limping) 증상이 명확히 의심됩니다."

        # [BALANCE]
        ratios = [sr_rl, sr_rr]
        if missing["fl_paw"] < 0.2: ratios.extend([stance["fl_paw"].mean(), stance["fr_paw"].mean()])
        bal_ratio = min(ratios) / (max(ratios) + 1e-6)
        balance_score = int(bal_ratio * 100)
        
        if balance_score >= 80: bal_lvl, bal_desc = "good", "네 다리에 체중을 고르게 분산하고 있습니다. 훌륭합니다."
        elif balance_score >= 60: bal_lvl, bal_desc = "fair", "균형이 양호한 편이나, 특정 다리에 힘을 조금 덜 싣는 경향이 있습니다."
        elif balance_score >= 30: bal_lvl, bal_desc = "poor", "체중 중심이 한쪽으로 쏠려 있습니다. 아픈 다리를 보호하려는 동작일 수 있습니다."
        else: bal_lvl, bal_desc = "poor", "균형이 심각하게 무너져 있습니다. 서 있거나 걷는 것이 힘들어 보입니다."

        # [MOBILITY]
        ang_l = np.array(ts["angle_l"]); ang_l = ang_l[~np.isnan(ang_l)]
        ang_r = np.array(ts["angle_r"]); ang_r = ang_r[~np.isnan(ang_r)]
        rom_l = (np.percentile(ang_l, 95)-np.percentile(ang_l, 5)) if len(ang_l)>10 else 0
        rom_r = (np.percentile(ang_r, 95)-np.percentile(ang_r, 5)) if len(ang_r)>10 else 0
        avg_rom = (rom_l + rom_r)/2
        rom_diff = abs(rom_l - rom_r)
        
        mob_score = int(max(0, min(100, (avg_rom/45.0)*100 - (rom_diff/30.0)*60)))
        mob_lvl = "normal" if mob_score > 70 else "stiff"
        
        if mob_score >= 80: mob_desc = "무릎 관절이 매우 유연하게 움직입니다. 가동 범위가 넓고 활기찹니다."
        elif mob_score >= 50: mob_desc = "관절 움직임이 정상 범위 내에 있습니다."
        elif mob_score >= 30: mob_desc = "걷는 모습이 다소 뻣뻣합니다. 관절이 굳어 있거나 통증이 있을 수 있습니다."
        else: mob_desc = "관절 가동 범위가 매우 제한적입니다. 노령견이거나 관절염 가능성이 있습니다."

        # [STABILITY]
        kx_l, ky_l = _get_clean("rl_knee"); kx_r, ky_r = _get_clean("rr_knee")
        d_l = np.sqrt((tail_x - kx_l)**2 + (tail_y - ky_l)**2)
        d_r = np.sqrt((tail_x - kx_r)**2 + (tail_y - ky_r)**2)
        stab_std = np.std((d_l + d_r)/2)
        stab_score = int(max(0, 100 - (stab_std / (height_scale+1e-6)) * Config.STABILITY_WEIGHT))
        stab_lvl = "stable" if stab_score > 70 else "unstable"
        
        if stab_score >= 80: stab_desc = "흔들림 없이 아주 안정적으로 걷고 있습니다. 코어 근육이 튼튼해 보입니다."
        elif stab_score >= 60: stab_desc = "전반적으로 안정적이나, 가끔 몸통이 흔들리는 모습이 보입니다."
        elif stab_score >= 30: stab_desc = "걸을 때 상체나 엉덩이가 눈에 띄게 흔들립니다."
        else: stab_desc = "보행이 매우 불안정합니다. 비틀거리거나 중심을 잡기 어려워 보입니다."

        # [PATELLA]
        patella_score, patella_lvl = 100, "low"
        patella_desc = "슬개골 탈구를 의심할 만한 특이 소견이 발견되지 않았습니다."
        
        if rhythm_score < Config.PATELLA_RHYTHM_LOW or asym > Config.PATELLA_ASYM_HIGH:
            patella_score, patella_lvl = 50, "high"
            patella_desc = "간헐적인 다리 절음(Skipping) 현상이 의심됩니다. 전문의 상담을 권장합니다."
        elif balance_score < Config.PATELLA_BALANCE_MED or mob_score < 40:
             patella_score, patella_lvl = 75, "medium"
             patella_desc = "보행 패턴에서 미세한 이상 신호가 감지되었습니다. 지속적인 관찰이 필요합니다."

        overall = int((rhythm_score + balance_score + mob_score + stab_score + patella_score)/5)

        return {
            "analysis_id": aid, "dog_id": did, "analyze_at": datetime.now().isoformat(),
            "result": {"overall_score": overall, "overall_risk_level": patella_lvl, "summary": f"종합점수 {overall}점. {patella_desc}"},
            "metrics": {
                "gait_rhythm": {"score": rhythm_score, "level": rhythm_lvl, "description": rhythm_desc},
                "gait_balance": {"score": balance_score, "level": bal_lvl, "description": bal_desc},
                "knee_mobility": {"score": mob_score, "level": mob_lvl, "description": mob_desc},
                "gait_stability": {"score": stab_score, "level": stab_lvl, "description": stab_desc},
                "patella_risk_signal": {"score": patella_score, "level": patella_lvl, "description": patella_desc}
            },
            "artifacts": {
                "keypoint_overlay_video_url": None,
                "debug": {
                    "missing_ratio": missing,
                    "height_scale": float(height_scale),
                    "stance_debug": debug_info,
                    "rear_asym": float(asym),
                    "step_diff": int(step_diff),
                    "rom_diff": float(rom_diff),
                    "stab_std": float(stab_std)
                }
            },
            "error_code": None
        }

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------
    def _draw_skeleton(self, frame, kpts):
        vis = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,16,17]
        for i in vis:
            if i < len(kpts) and kpts[i][2] > Config.KP_CONF_TH:
                cv2.circle(frame, (int(kpts[i][0]), int(kpts[i][1])), 4, (0,255,0), -1)
        for i,j,c in self.SKELETON:
            if i<len(kpts) and j<len(kpts) and kpts[i][2]>Config.KP_CONF_TH and kpts[j][2]>Config.KP_CONF_TH:
                cv2.line(frame, (int(kpts[i][0]), int(kpts[i][1])), (int(kpts[j][0]), int(kpts[j][1])), self.COLORS[c], 2)

    def _calculate_angle(self, kpts, side):
        idx = [12, 4, 3] if side == "left" else [12, 10, 9]
        p = [kpts[i] for i in idx]
        if any(pt[2] < Config.KP_CONF_TH for pt in p): return np.nan
        v1 = p[0][:2]-p[1][:2]; v2 = p[2][:2]-p[1][:2]
        dot = np.dot(v1, v2); norm = np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6
        return np.degrees(np.arccos(np.clip(dot/norm, -1.0, 1.0)))

    def _resolve_video_source(self, src):
        if src.startswith("http"):
            try:
                r = requests.get(src, stream=True); r.raise_for_status()
                fd, p = tempfile.mkstemp(suffix=".mp4")
                with os.fdopen(fd, 'wb') as f:
                    for c in r.iter_content(8192): f.write(c)
                return p, p
            except: return None, None
        return src, None

    def _cleanup(self, p):
        if p and os.path.exists(p): os.remove(p)
    
    def _error(self, aid, did, code):
        return {"analysis_id": aid, "dog_id": did, "analyze_at": datetime.now().isoformat(), "error_code": code}
    
    # Maths
    def _naninterp(self, x):
        mask = np.isnan(x); x=x.copy()
        if mask.all(): return x
        x[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), x[~mask])
        return x
    
    def _ema(self, x, a=Config.EMA_ALPHA):
        if len(x)==0: return x
        y = x.copy().astype(float)
        first_valid = np.where(~np.isnan(y))[0]
        if len(first_valid) == 0: return y
        start = first_valid[0]
        for i in range(start+1, len(y)):
            if np.isnan(y[i]): y[i] = y[i-1]
            elif np.isnan(y[i-1]): pass 
            else: y[i] = a*y[i] + (1-a)*y[i-1]
        return y

    def _robust_q(self, x, q):
        x = x[~np.isnan(x)]
        return float(np.percentile(x, q)) if len(x)>0 else float('nan')

    def _fill_gaps(self, m, g):
        o = m.copy(); n=len(m); i=0
        while i<n:
            if o[i]: i+=1; continue
            j=i
            while j<n and not o[j]: j+=1
            if i>0 and j<n and (j-i)<=g: o[i:j]=True
            i=j
        return o

    def _remove_short_runs(self, m, r):
        o = m.copy(); n=len(m); i=0
        while i<n:
            if not o[i]: i+=1; continue
            j=i
            while j<n and o[j]: j+=1
            if (j-i)<r: o[i:j]=False
            i=j
        return o

    def _count_runs(self, m):
        if len(m)==0: return 0
        c=0; p=False
        for v in m: 
            if v and not p: c+=1
            p=v
        return c

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model", type=str, default="models/best_26m.pt")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()
    
    res = DogHealthAnalyzer(args.model, args.output_dir).analyze_video(args.video)
    print(json.dumps(res, indent=4, ensure_ascii=False))
