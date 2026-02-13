import cv2
import numpy as np
import json
from ultralytics import YOLO
from scipy.signal import find_peaks
import os
import argparse
from datetime import datetime
import uuid
import requests
import tempfile

class DogHealthAnalyzer:
    def __init__(self, model_path, output_dir="output"):
        self.model = YOLO(model_path)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # V3: Final Tuned Logic (Full FPS, EMA Smoothing, Relaxed Metrics)
        self.KP_MAP = {
            "FL_PAW": 0, "FL_KNEE": 1, "FL_ELBOW": 2,
            "RL_PAW": 3, "RL_KNEE": 4, "RL_HOCK": 5, 
            "FR_PAW": 6, "FR_KNEE": 7, "FR_ELBOW": 8,
            "RR_PAW": 9, "RR_KNEE": 10, "RR_HOCK": 11,
            "TAIL_START": 12, "TAIL_END": 13,
            "NOSE": 16, "CHIN": 17 
        }
        self.SKELETON = [(9, 10, 'right'), (10, 11, 'right'), (11, 12, 'right'), (3, 4, 'left'), (4, 5, 'left'), (5, 12, 'left'), (12, 13, 'tail'), (0, 1, 'left'), (1, 2, 'left'), (6, 7, 'right'), (7, 8, 'right'), (16, 17, 'head')]
        self.COLORS = {'right': (0, 0, 255), 'left': (255, 0, 0), 'tail': (0, 255, 0), 'head': (0, 255, 255)}

    def analyze_video(self, video_source, dog_id=123, analysis_id=None):
        if analysis_id is None: analysis_id = str(uuid.uuid4())
        print(f"Starting V3 analysis {analysis_id}...")
        
        cap = cv2.VideoCapture(video_source); fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(f"output/overlay_{analysis_id}.mp4", cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
        
        ts_data = {"timestamps": [], "rl_knee_y": [], "rr_knee_y": [], "rl_paw_y": [], "rr_paw_y": [], "fl_paw_y": [], "fr_paw_y": [], "back_y": [], "knee_angles_l": [], "knee_angles_r": []}

        # V3 Feature: Smoothing
        alpha = 0.6; smoothed_kpts = None 
        
        while cap.isOpened():
            ret, frame = cap.read(); 
            if not ret: break
            
            results = self.model(frame, verbose=False)
            if len(results[0].keypoints) > 0:
                raw_kpts = results[0].keypoints.data[0].cpu().numpy()
                if smoothed_kpts is None: smoothed_kpts = raw_kpts
                else: smoothed_kpts[:, :2] = (alpha * raw_kpts[:, :2]) + ((1 - alpha) * smoothed_kpts[:, :2])
                
                self._draw_skeleton(frame, smoothed_kpts)
                
                # Logic V3: Using Smoothed Data
                ts_data["rl_paw_y"].append(smoothed_kpts[self.KP_MAP["RL_PAW"]][1])
                ts_data["rr_paw_y"].append(smoothed_kpts[self.KP_MAP["RR_PAW"]][1])
                ts_data["back_y"].append(smoothed_kpts[self.KP_MAP["TAIL_START"]][1])
                
                ang_l = self._calculate_angle(smoothed_kpts[12], smoothed_kpts[4], smoothed_kpts[3])
                ang_r = self._calculate_angle(smoothed_kpts[12], smoothed_kpts[10], smoothed_kpts[9])
                ts_data["knee_angles_l"].append(ang_l); ts_data["knee_angles_r"].append(ang_r)

            out.write(frame)
        cap.release(); out.release()
        
        report = self._generate_metrics(ts_data, fps, analysis_id, dog_id)
        with open(f"output/analysis_{analysis_id}.json", "w") as f: json.dump(report, f, indent=4)
        return report

    def _draw_skeleton(self, frame, kpts): pass
    def _calculate_angle(self, p1, p2, p3): return 0

    def _generate_metrics(self, data, fps, aid, did):
        # V3 Tuning: Relaxed thresholds
        peaks_l, _ = find_peaks(data["rl_paw_y"], distance=int(fps*0.2)) # Better dist
        peaks_r, _ = find_peaks(data["rr_paw_y"], distance=int(fps*0.2))
        
        diff = abs(len(peaks_l)-len(peaks_r))
        # V3: Penalty reduced (10 per step)
        rhythm_score = max(0, 100 - diff*10) 
        
        # Stability V3: Normalized by leg length estimate
        scale = np.mean(data["rl_paw_y"]) - np.mean(data["back_y"])
        stab = max(0, 100 - int(np.std(data["back_y"])/scale * 300))
        
        return {"analysis_id": aid, "result": {"overall_score": int((rhythm_score+stab)/2)}}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    args = parser.parse_args()
    print("V3 Analyzer (Full FPS + Smoothing + Tuned)")
