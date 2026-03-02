"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         ADVANCED FALL DETECTION SYSTEM - YOLOv8 Pose + Telegram Alert        ║
║         Optimized for Raspberry Pi | Child & Elderly Patient Safety          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  Camera Feed → YOLOv8n-pose (Skeleton Extraction)   │
  │       ↓                                             │
  │  State Machine (STANDING→FALLING→FALLEN)            │
  │       ↓                                             │
  │  Context Filter (Bed Zone Exclusion)                │
  │       ↓                                             │
  │  Telegram Alert + Local Alarm                       │
  └─────────────────────────────────────────────────────┘

Fall Detection Logic:
  - Torso angle deviation from vertical (>60°)
  - Sudden drop in hip/shoulder keypoint velocity
  - Aspect ratio of bounding box (horizontal > vertical)
  - Hip height relative to frame history
  - NOT triggered when person is in BED ZONE
"""

import cv2
import numpy as np
import time
import threading
import asyncio
import logging
import json
import os
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Dict
from datetime import datetime


try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Run: pip install ultralytics")

try:
    import requests
except ImportError:
    raise ImportError("Run: pip install requests")

try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("fall_detection.log"),
    ],
)
log = logging.getLogger(__name__)

CONFIG = {
    
    "camera_index": 0,
    "frame_width": 640,
    "frame_height": 480,
    "fps_target": 15,           

   
    "pose_model": "yolov8n-pose.pt",   
    "obj_model": "yolov8n.pt",         
    "conf_threshold": 0.45,

    
    "torso_angle_threshold": 55,       
    "aspect_ratio_threshold": 1.3,      
    "hip_drop_velocity_threshold": 0.08,
    "min_fall_duration_frames": 6,     
    "confirmation_frames": 8,          
    "recovery_frames": 45,             
    "keypoint_confidence_min": 0.3,

   
    "bed_zone_manual": None,            
    "bed_detection_auto": True,         
    "bed_iou_threshold": 0.45,         

    
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "telegram_alert_cooldown": 60,     
    "send_snapshot": True,            

    
    "alert_sound": True,
    "save_clips": True,
    "clip_dir": "fall_clips",
    "display_window": True,             
}

BED_CLASSES = {59: "bed", 57: "couch", 60: "dining table"}  
PERSON_CLASS = 0

KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}


class PersonState(Enum):
    UNKNOWN   = "unknown"
    STANDING  = "standing"
    CROUCHING = "crouching"
    SITTING   = "sitting"
    FALLING   = "falling"
    FALLEN    = "fallen"
    SLEEPING  = "sleeping"    

@dataclass
class PersonTrack:
    track_id: int
    state: PersonState = PersonState.UNKNOWN
    state_history: deque = field(default_factory=lambda: deque(maxlen=60))
    hip_height_history: deque = field(default_factory=lambda: deque(maxlen=30))
    torso_angle_history: deque = field(default_factory=lambda: deque(maxlen=30))
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=30))
    fall_frame_count: int = 0
    recovery_frame_count: int = 0
    alerted: bool = False
    last_alert_time: float = 0.0
    consecutive_fall_frames: int = 0


class TelegramAlerter:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self._lock = threading.Lock()
        self._queue: List[dict] = []
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        log.info("Telegram alerter initialized.")

    def send_alert(self, message: str, image: Optional[np.ndarray] = None):
        with self._lock:
            self._queue.append({"message": message, "image": image})

    def _worker(self):
        while True:
            item = None
            with self._lock:
                if self._queue:
                    item = self._queue.pop(0)
            if item:
                self._send(item["message"], item["image"])
            time.sleep(0.1)

    def _send(self, message: str, image: Optional[np.ndarray] = None):
        try:
           
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                data={"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"},
                timeout=10,
            )
            if not resp.ok:
                log.warning(f"Telegram text failed: {resp.text}")

           
            if image is not None:
                _, img_encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                resp = requests.post(
                    f"{self.base_url}/sendPhoto",
                    data={"chat_id": self.chat_id, "caption": "⚠️ Fall Snapshot"},
                    files={"photo": ("snapshot.jpg", img_encoded.tobytes(), "image/jpeg")},
                    timeout=15,
                )
                if not resp.ok:
                    log.warning(f"Telegram photo failed: {resp.text}")
        except Exception as e:
            log.error(f"Telegram send error: {e}")


class FallDetector:
    def __init__(self, config: dict):
        self.cfg = config
        self.tracks: Dict[int, PersonTrack] = {}
        self.frame_count = 0
        self.bed_zones: List[Tuple[int,int,int,int]] = []
        self.last_obj_detection_frame = -99
        self.obj_detection_interval = 30  

        log.info("Loading YOLOv8 pose model...")
        self.pose_model = YOLO(config["pose_model"])
        self.pose_model.fuse()  

        if config["bed_detection_auto"]:
            log.info("Loading YOLOv8 object model for bed detection...")
            self.obj_model = YOLO(config["obj_model"])
            self.obj_model.fuse()
        else:
            self.obj_model = None

        if config["bed_zone_manual"]:
            log.info(f"Using manual bed zone: {config['bed_zone_manual']}")

       
        token = config["telegram_bot_token"]
        chat_id = config["telegram_chat_id"]
        if "YOUR_" in token or "YOUR_" in chat_id:
            log.warning("⚠️  Telegram not configured. Edit CONFIG with your token/chat_id.")
            self.telegram = None
        else:
            self.telegram = TelegramAlerter(token, chat_id)

        
        if config["save_clips"]:
            os.makedirs(config["clip_dir"], exist_ok=True)
        self.clip_buffer: deque = deque(maxlen=90)  

   
    def _update_bed_zones(self, frame: np.ndarray) -> List[Tuple]:
        zones = []
        h, w = frame.shape[:2]

        
        if self.cfg["bed_zone_manual"]:
            x1n, y1n, x2n, y2n = self.cfg["bed_zone_manual"]
            zones.append((int(x1n*w), int(y1n*h), int(x2n*w), int(y2n*h)))
            return zones

       
        if self.obj_model is None:
            return zones

        results = self.obj_model(
            frame, conf=0.3, classes=list(BED_CLASSES.keys()),
            verbose=False, imgsz=320
        )
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in BED_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                   
                    pad_x = int((x2 - x1) * 0.05)
                    pad_y = int((y2 - y1) * 0.1)
                    zones.append((
                        max(0, x1 - pad_x), max(0, y1 - pad_y),
                        min(w, x2 + pad_x), min(h, y2 + pad_y)
                    ))
                    log.info(f"Bed zone detected: {BED_CLASSES[cls_id]}")
        return zones

    def _is_in_bed_zone(self, bbox: Tuple, frame_shape: Tuple) -> bool:
        """Check if person bbox overlaps significantly with a bed zone."""
        h, w = frame_shape[:2]
        px1, py1, px2, py2 = bbox
        person_area = max(1, (px2 - px1) * (py2 - py1))

        for bx1, by1, bx2, by2 in self.bed_zones:
            # Intersection
            ix1 = max(px1, bx1)
            iy1 = max(py1, by1)
            ix2 = min(px2, bx2)
            iy2 = min(py2, by2)
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                iou = inter_area / person_area
                if iou >= self.cfg["bed_iou_threshold"]:
                    return True
        return False

    
    def _get_keypoint(self, kps: np.ndarray, name: str, conf_min: float = 0.3):
        """Return (x, y) or None if confidence too low."""
        idx = KP[name]
        if kps.shape[0] <= idx:
            return None
        x, y, conf = kps[idx]
        return (float(x), float(y)) if conf >= conf_min else None

    def _compute_torso_angle(self, kps: np.ndarray) -> Optional[float]:
        """
        Angle of torso from vertical (0° = upright, 90° = horizontal).
        Uses shoulder midpoint → hip midpoint vector.
        """
        ls = self._get_keypoint(kps, "left_shoulder")
        rs = self._get_keypoint(kps, "right_shoulder")
        lh = self._get_keypoint(kps, "left_hip")
        rh = self._get_keypoint(kps, "right_hip")

        if None in (ls, rs, lh, rh):
            return None

        shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)

        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]

        # Angle from vertical (dy axis)
        angle = abs(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))
        return angle

    def _compute_hip_height(self, kps: np.ndarray, frame_h: int) -> Optional[float]:
        """Normalized hip height (0=top, 1=bottom of frame)."""
        lh = self._get_keypoint(kps, "left_hip")
        rh = self._get_keypoint(kps, "right_hip")
        if lh is None and rh is None:
            return None
        pts = [p for p in [lh, rh] if p is not None]
        avg_y = sum(p[1] for p in pts) / len(pts)
        return avg_y / frame_h

    def _compute_head_height(self, kps: np.ndarray, frame_h: int) -> Optional[float]:
        """Normalized head/nose height."""
        nose = self._get_keypoint(kps, "nose")
        le = self._get_keypoint(kps, "left_ear")
        re = self._get_keypoint(kps, "right_ear")
        pts = [p for p in [nose, le, re] if p is not None]
        if not pts:
            return None
        avg_y = sum(p[1] for p in pts) / len(pts)
        return avg_y / frame_h

    def _is_horizontal(self, bbox: Tuple) -> Tuple[bool, float]:
        """Check if bounding box is more wide than tall."""
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        ratio = w / h
        return ratio >= self.cfg["aspect_ratio_threshold"], ratio

   
    def _classify_posture(
        self,
        track: PersonTrack,
        torso_angle: Optional[float],
        hip_height: Optional[float],
        is_horizontal: bool,
        bbox_ratio: float,
        in_bed: bool,
        frame_h: int,
    ) -> PersonState:
        """Rule-based posture classification."""

        if in_bed:
            return PersonState.SLEEPING

        if torso_angle is None or hip_height is None:
            return track.state 
        
        hip_drop_velocity = 0.0
        if len(track.hip_height_history) >= 3:
            recent = list(track.hip_height_history)
            hip_drop_velocity = recent[-1] - recent[-3]  

        
        fall_score = 0

        
        if torso_angle > self.cfg["torso_angle_threshold"]:
            fall_score += 2
        elif torso_angle > 40:
            fall_score += 1

       
        if is_horizontal:
            fall_score += 2
        elif bbox_ratio > 1.0:
            fall_score += 1

       
        if hip_drop_velocity > self.cfg["hip_drop_velocity_threshold"]:
            fall_score += 2

       
        if hip_height > 0.7:
            fall_score += 1

       
        if fall_score >= 4:
            track.consecutive_fall_frames += 1
            if track.consecutive_fall_frames >= self.cfg["min_fall_duration_frames"]:
                return PersonState.FALLEN
            return PersonState.FALLING
        else:
            track.consecutive_fall_frames = max(0, track.consecutive_fall_frames - 1)

        if torso_angle < 20 and hip_height < 0.85:
            return PersonState.STANDING
        if 20 <= torso_angle < self.cfg["torso_angle_threshold"]:
            if hip_height > 0.6:
                return PersonState.SITTING
            return PersonState.CROUCHING
        return PersonState.STANDING

    
    def _trigger_alert(self, track_id: int, frame: np.ndarray):
        now = time.time()
        track = self.tracks[track_id]

        cooldown = self.cfg["telegram_alert_cooldown"]
        if now - track.last_alert_time < cooldown:
            return

        track.last_alert_time = now
        track.alerted = True

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = (
            f"🚨 <b>FALL DETECTED!</b>\n"
            f"──────────────────\n"
            f"🕐 Time: <b>{ts}</b>\n"
            f"🆔 Person ID: <b>{track_id}</b>\n"
            f"📍 Location: <b>Monitored Area</b>\n"
            f"⚠️ Status: <b>Person has fallen</b>\n"
            f"──────────────────\n"
            f"🏥 Please check on the patient immediately!\n"
            f"📞 Call emergency if no response within 2 minutes."
        )

        snap = frame.copy() if self.cfg["send_snapshot"] else None
        if self.telegram:
            self.telegram.send_alert(message, snap)
            log.info(f"🚨 Telegram alert sent for track {track_id}")
        else:
            log.warning("Telegram not configured — alert NOT sent!")

        
        if self.cfg["save_clips"]:
            self._save_clip(track_id, ts)

       
        if self.cfg["alert_sound"]:
            threading.Thread(target=self._play_alarm, daemon=True).start()

    def _save_clip(self, track_id: int, ts: str):
        """Save pre-event buffer as video clip."""
        safe_ts = ts.replace(":", "-").replace(" ", "_")
        path = os.path.join(self.cfg["clip_dir"], f"fall_{track_id}_{safe_ts}.avi")
        if not self.clip_buffer:
            return
        h, w = self.clip_buffer[0].shape[:2]
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"XVID"), 10, (w, h))
        for f in self.clip_buffer:
            out.write(f)
        out.release()
        log.info(f"Clip saved: {path}")

    def _play_alarm(self):
        """Cross-platform alarm (works on Pi with audio or buzzer GPIO)."""
        try:
            import subprocess
           
            subprocess.run(["aplay", "/usr/share/sounds/alsa/Front_Center.wav"],
                           capture_output=True, timeout=3)
        except Exception:
            try:
                print("\a" * 5)  
            except Exception:
                pass

   
    def _draw_overlay(
        self,
        frame: np.ndarray,
        bbox: Tuple,
        kps: np.ndarray,
        track: PersonTrack,
        torso_angle: Optional[float],
        in_bed: bool,
    ) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        state = track.state

        # Box color by state
        COLOR_MAP = {
            PersonState.STANDING:  (0, 200, 0),
            PersonState.CROUCHING: (0, 200, 200),
            PersonState.SITTING:   (200, 200, 0),
            PersonState.FALLING:   (0, 100, 255),
            PersonState.FALLEN:    (0, 0, 255),
            PersonState.SLEEPING:  (120, 120, 120),
            PersonState.UNKNOWN:   (180, 180, 180),
        }
        color = COLOR_MAP.get(state, (180, 180, 180))
        thickness = 3 if state in (PersonState.FALLEN, PersonState.FALLING) else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label
        label = f"ID:{track.track_id} | {state.value.upper()}"
        if torso_angle is not None:
            label += f" | {torso_angle:.0f}°"
        if in_bed:
            label += " | IN BED"

        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        
        frame = self._draw_skeleton(frame, kps)

        
        if state == PersonState.FALLEN:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, "⚠  FALL DETECTED — ALERTING CARETAKER",
                        (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

        return frame

    def _draw_skeleton(self, frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
        SKELETON_PAIRS = [
            ("left_shoulder","right_shoulder"),
            ("left_shoulder","left_elbow"), ("left_elbow","left_wrist"),
            ("right_shoulder","right_elbow"), ("right_elbow","right_wrist"),
            ("left_shoulder","left_hip"), ("right_shoulder","right_hip"),
            ("left_hip","right_hip"),
            ("left_hip","left_knee"), ("left_knee","left_ankle"),
            ("right_hip","right_knee"), ("right_knee","right_ankle"),
            ("nose","left_eye"), ("nose","right_eye"),
            ("left_eye","left_ear"), ("right_eye","right_ear"),
        ]
        c_min = self.cfg["keypoint_confidence_min"]
        for a, b in SKELETON_PAIRS:
            pa = self._get_keypoint(kps, a, c_min)
            pb = self._get_keypoint(kps, b, c_min)
            if pa and pb:
                cv2.line(frame, (int(pa[0]), int(pa[1])),
                         (int(pb[0]), int(pb[1])), (0, 255, 128), 2)
        for name, idx in KP.items():
            if idx < kps.shape[0]:
                x, y, conf = kps[idx]
                if conf >= c_min:
                    cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 0), -1)
        return frame

    
    def _draw_bed_zones(self, frame: np.ndarray):
        for bx1, by1, bx2, by2 in self.bed_zones:
            overlay = frame.copy()
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (100, 180, 100), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 180, 0), 2)
            cv2.putText(frame, "BED ZONE (no alert)", (bx1+4, by1+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

  
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.frame_count += 1
        h, w = frame.shape[:2]

        self.clip_buffer.append(frame.copy())

       
        if (self.frame_count - self.last_obj_detection_frame) >= self.obj_detection_interval:
            self.bed_zones = self._update_bed_zones(frame)
            self.last_obj_detection_frame = self.frame_count

       
        results = self.pose_model.track(
            frame,
            conf=self.cfg["conf_threshold"],
            persist=True,
            verbose=False,
            imgsz=self.cfg["frame_width"],
            tracker="bytetrack.yaml",
        )

        active_ids = set()

        for r in results:
            if r.boxes is None or r.keypoints is None:
                continue

            boxes = r.boxes
            kps_data = r.keypoints.data if r.keypoints.data is not None else []

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                if cls_id != PERSON_CLASS:
                    continue

                
                track_id = int(box.id[0]) if box.id is not None else i
                active_ids.add(track_id)

                
                if track_id not in self.tracks:
                    self.tracks[track_id] = PersonTrack(track_id=track_id)
                track = self.tracks[track_id]

                bbox = tuple(map(int, box.xyxy[0]))

                
                if i < len(kps_data):
                    kps = kps_data[i].cpu().numpy()  
                else:
                    continue

               
                torso_angle = self._compute_torso_angle(kps)
                hip_height = self._compute_hip_height(kps, h)
                is_horiz, bbox_ratio = self._is_horizontal(bbox)
                in_bed = self._is_in_bed_zone(bbox, frame.shape)

                
                if hip_height is not None:
                    track.hip_height_history.append(hip_height)
                if torso_angle is not None:
                    track.torso_angle_history.append(torso_angle)
                track.bbox_history.append(bbox)

                
                new_state = self._classify_posture(
                    track, torso_angle, hip_height,
                    is_horiz, bbox_ratio, in_bed, h
                )
                track.state_history.append(new_state)
                track.state = new_state

                
                if (
                    new_state == PersonState.FALLEN
                    and not in_bed
                    and track.consecutive_fall_frames >= self.cfg["confirmation_frames"]
                ):
                    self._trigger_alert(track_id, frame)

                
                frame = self._draw_overlay(
                    frame, bbox, kps, track, torso_angle, in_bed
                )

        
        stale = [tid for tid in list(self.tracks.keys()) if tid not in active_ids]
        for tid in stale:
            del self.tracks[tid]

       
        self._draw_bed_zones(frame)

        
        self._draw_hud(frame)
        return frame

    def _draw_hud(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        ts = datetime.now().strftime("%H:%M:%S")
        alive = len(self.tracks)
        fallen = sum(1 for t in self.tracks.values() if t.state == PersonState.FALLEN)
        cv2.putText(frame, f"Time: {ts} | Persons: {alive} | Fallen: {fallen}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "FALL DETECTION SYSTEM v2.0",
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)



def main():
    log.info("=" * 60)
    log.info("  Advanced Fall Detection System — Starting")
    log.info("=" * 60)

    detector = FallDetector(CONFIG)

    cap = cv2.VideoCapture(CONFIG["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])
    cap.set(cv2.CAP_PROP_FPS, CONFIG["fps_target"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 

    if not cap.isOpened():
        log.error("Cannot open camera!")
        return

    log.info(f"Camera opened: {CONFIG['frame_width']}x{CONFIG['frame_height']} @ {CONFIG['fps_target']}fps")
    log.info("Press 'q' to quit | 'b' to mark bed zone manually | 'r' to reset tracks")

    frame_times = deque(maxlen=30)
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame grab failed, retrying...")
            time.sleep(0.1)
            continue

       
        now = time.time()
        elapsed = now - prev_time
        target_interval = 1.0 / CONFIG["fps_target"]
        if elapsed < target_interval:
            time.sleep(target_interval - elapsed)
        prev_time = time.time()

        t0 = time.perf_counter()
        output = detector.process_frame(frame)
        dt = time.perf_counter() - t0
        frame_times.append(dt)
        avg_fps = 1.0 / (sum(frame_times) / len(frame_times) + 1e-6)

        cv2.putText(output, f"FPS: {avg_fps:.1f}", (output.shape[1]-100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if CONFIG["display_window"]:
            cv2.imshow("Fall Detection System", output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                detector.tracks.clear()
                log.info("Tracks reset.")
            elif key == ord("b"):
                log.info("Manual bed zone: set 'bed_zone_manual' in CONFIG.")

    cap.release()
    cv2.destroyAllWindows()
    log.info("System stopped.")


if __name__ == "__main__":
    main()