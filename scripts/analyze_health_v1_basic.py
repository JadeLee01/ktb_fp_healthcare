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
        # Old Keypoint Map (Withers, Hock used)
        self.KP_MAP = {
            "FL_PAW": 0, "FL_KNEE": 1, "FL_ELBOW": 2,
            "RL_PAW": 3, "RL_KNEE": 4, "RL_HOCK": 5, 
            "FR_PAW": 6, "FR_KNEE": 7, "FR_ELBOW": 8,
            "RR_PAW": 9, "RR_KNEE": 10, "RR_HOCK": 11,
            "TAIL_START": 12, "TAIL_END": 13,
            "NOSE": 16, "CHIN": 17,
            "WITHERS": 22 
        }
        self.SKELETON = [(9, 10, 'right'), (10, 11, 'right'), (11, 12, 'right'), (3, 4, 'left'), (4, 5, 'left'), (5, 12, 'left'), (12, 13, 'tail'), (0, 1, 'left'), (1, 2, 'left'), (6, 7, 'right'), (7, 8, 'right'), (16, 17, 'head')]
        self.COLORS = {'right': (0, 0, 255), 'left': (255, 0, 0), 'tail': (0, 255, 0), 'head': (0, 255, 255)}

    def analyze_video(self, video_source, dog_id=123, analysis_id=None):
        if analysis_id is None: analysis_id = str(uuid.uuid4())
        print(f"Starting V1 analysis {analysis_id}...")
        video_path = video_source # Assuming local or handled
        if video_source.startswith("http"): # Simple download
           r = requests.get(video_source); video_path = "temp_v1.mp4"; open(video_path, 'wb').write(r.content)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(f"output/overlay_{analysis_id}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        ts_data = {"timestamps": [], "rl_knee_y": [], "rr_knee_y": [], "rl_paw_y": [], "rr_paw_y": [], "fl_paw_y": [], "fr_paw_y": [], "back_y": [], "head_y": [], "knee_angles_l": [], "knee_angles_r": []}

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            results = self.model(frame, verbose=False)
            if len(results[0].keypoints) > 0:
                kpts = results[0].keypoints.data[0].cpu().numpy()
                self._draw_skeleton(frame, kpts)
                ts_data["timestamps"].append(frame_idx / fps)
                ts_data["rl_knee_y"].append(kpts[self.KP_MAP["RL_KNEE"]][1])
                ts_data["rr_knee_y"].append(kpts[self.KP_MAP["RR_KNEE"]][1])
                ts_data["rl_paw_y"].append(kpts[self.KP_MAP["RL_PAW"]][1])
                ts_data["rr_paw_y"].append(kpts[self.KP_MAP["RR_PAW"]][1])
                ts_data["fl_paw_y"].append(kpts[self.KP_MAP["FL_PAW"]][1])
                ts_data["fr_paw_y"].append(kpts[self.KP_MAP["FR_PAW"]][1])
                ts_data["back_y"].append(kpts[self.KP_MAP["WITHERS"]][1]) # Using Withers (Unstable)
                ts_data["head_y"].append(kpts[self.KP_MAP["NOSE"]][1])
                
                # Old Angle Logic (Hock used)
                angle_l = self._calculate_angle(kpts[self.KP_MAP["RL_HOCK"]], kpts[self.KP_MAP["RL_KNEE"]], kpts[self.KP_MAP["RL_PAW"]])
                angle_r = self._calculate_angle(kpts[self.KP_MAP["RR_HOCK"]], kpts[self.KP_MAP["RR_KNEE"]], kpts[self.KP_MAP["RR_PAW"]])
                ts_data["knee_angles_l"].append(angle_l)
                ts_data["knee_angles_r"].append(angle_r)
            out.write(frame)
            frame_idx += 1
        cap.release(); out.release()
        
        report = self._generate_metrics(ts_data, fps, analysis_id, dog_id)
        with open(f"output/analysis_{analysis_id}.json", "w") as f: json.dump(report, f, indent=4)
        return report

    def _draw_skeleton(self, frame, kpts):
        for i, (x, y, conf) in enumerate(kpts):
            if conf > 0.5: cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        for i, j, c in self.SKELETON:
            if i < len(kpts) and j < len(kpts) and kpts[i][2]>0.5 and kpts[j][2]>0.5:
                cv2.line(frame, (int(kpts[i][0]), int(kpts[i][1])), (int(kpts[j][0]), int(kpts[j][1])), self.COLORS[c], 2)

    def _calculate_angle(self, p1, p2, p3):
        if p1[2]<0.5 or p2[2]<0.5 or p3[2]<0.5: return 0
        v1 = p1[:2]-p2[:2]; v2 = p3[:2]-p2[:2]
        cos = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    def _generate_metrics(self, data, fps, aid, did):
        # Basic Logic (Peak finding)
        peaks_l, _ = find_peaks(data["rl_paw_y"], distance=fps//2)
        peaks_r, _ = find_peaks(data["rr_paw_y"], distance=fps//2)
        rhythm = max(0, 100 - abs(len(peaks_l)-len(peaks_r))*20)
        
        amp_l = np.std(data["rl_paw_y"]); amp_r = np.std(data["rr_paw_y"])
        balance = int((min(amp_l, amp_r)/(max(amp_l, amp_r)+1e-6))*100)
        
        rom = (np.mean(data["knee_angles_l"]) + np.mean(data["knee_angles_r"]))/2 if data["knee_angles_l"] else 0
        mobility = min(100, int(rom/90*100))
        
        stab = max(0, 100 - int(np.std(data["back_y"])*10)) # Withers variance
        
        return {"analysis_id": aid, "result": {"overall_score": int((rhythm+balance+mobility+stab)/4)}}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model", type=str, default="models/best_26m.pt")
    args = parser.parse_args()
    DogHealthAnalyzer(args.model).analyze_video(args.video)
