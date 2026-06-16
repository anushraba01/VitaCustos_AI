#!/usr/bin/env python3
"""
navigation.py — Child-Following Robot  (Forward-Only Follow Fix)
================================================================
Target: 10–20 FPS on Raspberry Pi 4

CORE LOGIC:
  ┌─────────────────────────────────────────────────────────────┐
  │  DISTANCE = bbox height / frame height  (h_ratio)          │
  │                                                             │
  │  h_ratio < TARGET_H - HTOL  → person is FAR  → FORWARD    │
  │  h_ratio > TARGET_H + HTOL  → person is NEAR → STOP       │
  │  within ±HTOL               → distance OK    → steer only  │
  │                                                             │
  │  TARGET_H = 0.45  (calibrate: stand at target distance,    │
  │             read H=x.xx on screen, set that value)         │
  │                                                             │
  │  ⚠ Robot NEVER goes backward — sonar is the only stop.    │
  └─────────────────────────────────────────────────────────────┘

  STEERING (X axis):
  x_dev = cx_norm - 0.5   (-0.5=far left … +0.5=far right)
  x_dev > +TOL  → turn right
  x_dev < -TOL  → turn left

GPIO Pins:
  Left  motors : IN1=17  IN2=18  ENA=22
  Right motors : IN3=23  IN4=24  ENB=25
  Ultrasonic   : TRIG=5   ECHO=6

HOW TO CALIBRATE TARGET_H (do this first!):
  1. Run the script, stand at the distance you want the robot to follow from.
  2. Look at the HUD or terminal — note the "H=x.xx" value shown.
  3. Stop the script, set TARGET_H to that value.
  4. Restart — robot will now maintain that exact distance.

Dependencies:
  pip install ultralytics opencv-python numpy RPi.GPIO
"""

import time
import threading
import logging
import math
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import RPi.GPIO as GPIO
    ON_PI = True
except ImportError:
    ON_PI = False
    print("[WARN] RPi.GPIO not found — SIMULATION mode")

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("Missing: pip install ultralytics")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ChildBot")


class CFG:
    IN1 = 17;  IN2 = 18;  ENA = 22
    IN3 = 23;  IN4 = 24;  ENB = 25
    TRIG = 5;  ECHO = 6

    CAM_INDEX = 0
    CAM_W = 640;  CAM_H = 480;  CAM_FPS = 30

    MODEL      = "yolov8n.pt"
    INFER_SIZE = 160
    CONF       = 0.45
    IOU        = 0.45
    SKIP       = 2

    TOLERANCE = 0.12

    TARGET_H = 0.45
    HTOL     = 0.06

    LOST_LIMIT = 25

    STOP_CM = 50.0
    SLOW_CM = 90.0

    PWM_HZ    = 1000
    FWD_SPD   = 65.0
    TURN_SPD  = 55.0
    SLOW_SPD  = 38.0
    MIN_SPD   = 25.0

    SEARCH_SPD = 35.0

    SHOW = True


class ThreadedCamera:
    def __init__(self):
        self._cap = cv2.VideoCapture(CFG.CAM_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG.CAM_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.CAM_H)
        self._cap.set(cv2.CAP_PROP_FPS,          CFG.CAM_FPS)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        if not self._cap.isOpened():
            raise RuntimeError("Cannot open camera!")
        self._frame:  Optional[np.ndarray] = None
        self._cam_ms: float = 0.0
        self._lock  = threading.Lock()
        self._stop  = False
        threading.Thread(target=self._loop, daemon=True).start()
        time.sleep(0.30)
        log.info("Camera ready.")

    def _loop(self):
        while not self._stop:
            t0 = time.monotonic()
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame  = frame
                    self._cam_ms = (time.monotonic() - t0) * 1000

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        with self._lock:
            if self._frame is None:
                return False, None, 0.0
            return True, self._frame.copy(), self._cam_ms

    def release(self):
        self._stop = True
        time.sleep(0.05)
        self._cap.release()


class CentroidTracker:
    def __init__(self):
        self.bbox: Optional[Tuple] = None
        self._vel  = np.zeros(4)
        self.lost  = 0

    def update(self, bbox_norm: Optional[Tuple]):
        if bbox_norm is None:
            self.lost += 1
            if self.lost > CFG.LOST_LIMIT:
                self.bbox = None
                self._vel = np.zeros(4)
            else:
                self._step()
            return
        new = np.array(bbox_norm, dtype=float)
        if self.bbox is not None:
            self._vel = new - np.array(self.bbox)
        self.bbox = tuple(new.tolist())
        self.lost = 0

    def extrapolate(self):
        self._step()

    def _step(self):
        if self.bbox is None:
            return
        b = np.clip(np.array(self.bbox) + self._vel * 0.4, 0.0, 1.0)
        if b[0] < b[2] and b[1] < b[3]:
            self.bbox = tuple(b.tolist())
        self._vel *= 0.85

    @property
    def cx_norm(self) -> Optional[float]:
        return None if self.bbox is None else (self.bbox[0] + self.bbox[2]) / 2

    @property
    def h_ratio(self) -> Optional[float]:
        return None if self.bbox is None else (self.bbox[3] - self.bbox[1])


class Motors:
    """
    Forward-only motor control.
    The robot NEVER drives backward — sonar handles all obstacle stopping.

    If the robot physically moves the WRONG direction on 'forward()':
      → Swap IN1↔IN2 for left  (or reverse left wheel wire physically)
      → Swap IN3↔IN4 for right (or reverse right wheel wire physically)
    """

    def __init__(self):
        self._lp = self._rp = None
        if ON_PI:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for pin in [CFG.IN1, CFG.IN2, CFG.ENA,
                        CFG.IN3, CFG.IN4, CFG.ENB]:
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            self._lp = GPIO.PWM(CFG.ENA, CFG.PWM_HZ)
            self._rp = GPIO.PWM(CFG.ENB, CFG.PWM_HZ)
            self._lp.start(0)
            self._rp.start(0)
            log.info("Motors GPIO ready (BCM).")

    def _drive_left(self, spd: float):
        """
        spd: 0–100 (always FORWARD for left wheel).
        Internally maps to IN1=HIGH, IN2=LOW.
        If robot goes backward → change HIGH/LOW here.
        """
        spd = float(np.clip(abs(spd), 0, 100))
        if ON_PI:
            GPIO.output(CFG.IN1, GPIO.HIGH)
            GPIO.output(CFG.IN2, GPIO.LOW)
            self._lp.ChangeDutyCycle(spd)

    def _drive_right(self, spd: float):
        """
        spd: 0–100 (always FORWARD for right wheel).
        Internally maps to IN3=HIGH, IN4=LOW.
        If robot goes backward → change HIGH/LOW here.
        """
        spd = float(np.clip(abs(spd), 0, 100))
        if ON_PI:
            GPIO.output(CFG.IN3, GPIO.HIGH)
            GPIO.output(CFG.IN4, GPIO.LOW)
            self._rp.ChangeDutyCycle(spd)

    def _spin_left_wheel(self, spd: float):
        """Spin left wheel BACKWARD (for in-place search spin)."""
        spd = float(np.clip(abs(spd), 0, 100))
        if ON_PI:
            GPIO.output(CFG.IN1, GPIO.LOW)
            GPIO.output(CFG.IN2, GPIO.HIGH)
            self._lp.ChangeDutyCycle(spd)

    def _spin_right_wheel(self, spd: float):
        """Spin right wheel BACKWARD (for in-place search spin)."""
        spd = float(np.clip(abs(spd), 0, 100))
        if ON_PI:
            GPIO.output(CFG.IN3, GPIO.LOW)
            GPIO.output(CFG.IN4, GPIO.HIGH)
            self._rp.ChangeDutyCycle(spd)

    def forward(self, spd: float = CFG.FWD_SPD):
        """Drive straight forward."""
        spd = float(np.clip(spd, CFG.MIN_SPD, 100))
        self._drive_left(spd)
        self._drive_right(spd)
        if not ON_PI:
            log.debug(f"[SIM] FORWARD  L={spd:.0f}%  R={spd:.0f}%")

    def forward_steer_right(self, spd: float):
        """
        Move forward while leaning RIGHT (person is to the right).
        Left wheel full speed, right wheel reduced → curves right.
        """
        spd = float(np.clip(spd, CFG.MIN_SPD, 100))
        inner = max(spd * 0.30, CFG.MIN_SPD)
        self._drive_left(spd)
        self._drive_right(inner)
        if not ON_PI:
            log.debug(f"[SIM] FORWARD+R  L={spd:.0f}%  R={inner:.0f}%")

    def forward_steer_left(self, spd: float):
        """
        Move forward while leaning LEFT (person is to the left).
        Right wheel full speed, left wheel reduced → curves left.
        """
        spd = float(np.clip(spd, CFG.MIN_SPD, 100))
        inner = max(spd * 0.30, CFG.MIN_SPD)
        self._drive_left(inner)
        self._drive_right(spd)
        if not ON_PI:
            log.debug(f"[SIM] FORWARD+L  L={inner:.0f}%  R={spd:.0f}%")

    def turn_right(self, spd: float = CFG.TURN_SPD):
        """Turn right in place: left forward, right slow/stop."""
        spd = float(np.clip(spd, CFG.MIN_SPD, 100))
        self._drive_left(spd)
        self._drive_right(spd * 0.10)
        if not ON_PI:
            log.debug(f"[SIM] TURN RIGHT  L={spd:.0f}%  R={spd*0.10:.0f}%")

    def turn_left(self, spd: float = CFG.TURN_SPD):
        """Turn left in place: right forward, left slow/stop."""
        spd = float(np.clip(spd, CFG.MIN_SPD, 100))
        self._drive_left(spd * 0.10)
        self._drive_right(spd)
        if not ON_PI:
            log.debug(f"[SIM] TURN LEFT  L={spd*0.10:.0f}%  R={spd:.0f}%")

    def spin_search_right(self, spd: float = CFG.SEARCH_SPD):
        """In-place spin RIGHT to search (left fwd, right back)."""
        spd = float(np.clip(spd, CFG.MIN_SPD, 100))
        self._drive_left(spd)
        self._spin_right_wheel(spd)
        if not ON_PI:
            log.debug(f"[SIM] SPIN RIGHT  L=+{spd:.0f}%  R=-{spd:.0f}%")

    def spin_search_left(self, spd: float = CFG.SEARCH_SPD):
        """In-place spin LEFT to search (right fwd, left back)."""
        spd = float(np.clip(spd, CFG.MIN_SPD, 100))
        self._spin_left_wheel(spd)
        self._drive_right(spd)
        if not ON_PI:
            log.debug(f"[SIM] SPIN LEFT  L=-{spd:.0f}%  R=+{spd:.0f}%")

    def stop(self):
        if ON_PI:
            for pin in [CFG.IN1, CFG.IN2, CFG.IN3, CFG.IN4]:
                GPIO.output(pin, GPIO.LOW)
            self._lp.ChangeDutyCycle(0)
            self._rp.ChangeDutyCycle(0)
        if not ON_PI:
            log.debug("[SIM] STOP")

    def cleanup(self):
        self.stop()
        if ON_PI:
            if self._lp: self._lp.stop()
            if self._rp: self._rp.stop()
            GPIO.cleanup()
            log.info("GPIO cleaned up.")


class Sonar:
    _WIN = 5

    def __init__(self):
        self._dist = 999.0
        self._lock = threading.Lock()
        self._buf  = deque(maxlen=self._WIN)
        self._run  = False
        if ON_PI:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(CFG.TRIG, GPIO.OUT)
            GPIO.setup(CFG.ECHO, GPIO.IN)
            GPIO.output(CFG.TRIG, GPIO.LOW)
            time.sleep(0.05)

    def _ping(self) -> float:
        if not ON_PI:
            return 999.0
        GPIO.output(CFG.TRIG, GPIO.HIGH)
        time.sleep(1e-5)
        GPIO.output(CFG.TRIG, GPIO.LOW)
        t0 = time.monotonic()
        while GPIO.input(CFG.ECHO) == 0:
            if time.monotonic() - t0 > 0.04: return 999.0
        s = time.monotonic()
        while GPIO.input(CFG.ECHO) == 1:
            if time.monotonic() - s  > 0.04: return 999.0
        return (time.monotonic() - s) * 17150

    def _loop(self):
        while self._run:
            d = self._ping()
            if 2 < d < 400:
                self._buf.append(d)
                with self._lock:
                    self._dist = float(np.median(self._buf))
            time.sleep(0.10)

    def start(self):
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._run = False

    @property
    def cm(self) -> float:
        with self._lock:
            return self._dist


class HUD:
    C_BLUE   = (200,  80,  20)
    C_GREEN  = ( 50, 210,  50)
    C_YELLOW = (  0, 210, 210)
    C_RED    = (  0,   0, 210)
    C_WHITE  = (235, 235, 235)
    C_BLACK  = (  0,   0,   0)
    C_ORANGE = ( 20, 140, 255)
    C_GREY   = (160, 160, 160)
    C_CYAN   = (210, 210,   0)

    STATUS_COLOR = {
        "NO OBJECT": ( 50,  50, 180),
        "TRACKING":  ( 20, 190, 255),
        "AT TARGET": ( 40, 200,  40),
    }
    DIR_COLOR = {
        "FORWARD":    ( 40, 200,  40),
        "FORWARD+L":  ( 20, 200, 100),
        "FORWARD+R":  ( 20, 200, 100),
        "TURN LEFT":  (  0, 190, 255),
        "TURN RIGHT": (  0, 190, 255),
        "AT TARGET":  ( 40, 200,  40),
        "SEARCHING":  (200, 200,   0),
        "SONAR STOP": (  0,   0, 255),
        "TOO CLOSE":  (  0, 100, 255),
    }

    BAR  = 42
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def render(self,
               frame:     np.ndarray,
               bbox:      Optional[Tuple],
               x_dev:     float,
               h_ratio:   float,
               direction: str,
               speed_pct: float,
               fps:       float,
               cam_ms:    float,
               inf_ms:    float,
               status:    str,
               sonar_cm:  float) -> np.ndarray:

        H, W = frame.shape[:2]
        fcx, fcy = W // 2, H // 2
        tx = int(CFG.TOLERANCE * W)
        ty = int(CFG.TOLERANCE * H)
        out = frame.copy()

        cv2.line(out, (fcx, 0), (fcx, H), self.C_BLUE, 1, cv2.LINE_AA)
        cv2.line(out, (0, fcy), (W, fcy), self.C_BLUE, 1, cv2.LINE_AA)

        cv2.rectangle(out, (fcx-tx, fcy-ty), (fcx+tx, fcy+ty),
                      self.C_GREEN, 2)

        tgt_y = int(H * (1.0 - CFG.TARGET_H) / 2)
        cv2.line(out, (0, tgt_y), (W, tgt_y), self.C_CYAN, 1, cv2.LINE_AA)
        cv2.line(out, (0, H - tgt_y), (W, H - tgt_y), self.C_CYAN, 1, cv2.LINE_AA)
        cv2.putText(out, f"TARGET H={CFG.TARGET_H:.2f}", (4, tgt_y - 4),
                    self.FONT, 0.38, self.C_CYAN, 1)

        if bbox is not None:
            x1 = int(bbox[0]*W); y1 = int(bbox[1]*H)
            x2 = int(bbox[2]*W); y2 = int(bbox[3]*H)
            cv2.rectangle(out, (x1,y1), (x2,y2), self.C_YELLOW, 2)
            ocx = (x1+x2)//2
            ocy = (y1+y2)//2
            cv2.circle(out, (ocx, ocy), 6, self.C_RED, -1)
            cv2.line(out, (ocx, ocy), (fcx, fcy), self.C_YELLOW, 1, cv2.LINE_AA)
            cv2.putText(out, f"H={h_ratio:.2f}", (x1, y1-6),
                        self.FONT, 0.50, self.C_YELLOW, 2)

        cv2.rectangle(out, (0, 0), (W, self.BAR), self.C_BLACK, -1)
        ty_t = self.BAR - 10
        cv2.putText(out, f"FPS:{fps:.1f}", (4, ty_t),
                    self.FONT, 0.65, self.C_WHITE, 2)
        cv2.putText(out, f"Cam:{cam_ms:.0f}ms  Inf:{inf_ms:.0f}ms",
                    (100, ty_t), self.FONT, 0.50, self.C_GREY, 1)
        s_col = self.STATUS_COLOR.get(status, self.C_WHITE)
        (sw, _), _ = cv2.getTextSize(status, self.FONT, 0.65, 2)
        cv2.putText(out, status, (W-sw-6, ty_t),
                    self.FONT, 0.65, s_col, 2)

        by0 = H - self.BAR
        cv2.rectangle(out, (0, by0), (W, H), self.C_BLACK, -1)
        by_t = H - 10

        xc = self.C_ORANGE if abs(x_dev) > CFG.TOLERANCE else self.C_GREEN
        cv2.putText(out, f"X:{x_dev:+.2f}", (4, by_t),
                    self.FONT, 0.60, xc, 2)

        dist_err = h_ratio - CFG.TARGET_H
        dc = self.C_ORANGE if abs(dist_err) > CFG.HTOL else self.C_GREEN
        cv2.putText(out, f"H:{h_ratio:.2f}(T:{CFG.TARGET_H:.2f})",
                    (110, by_t), self.FONT, 0.55, dc, 2)

        d_col = self.DIR_COLOR.get(direction, self.C_YELLOW)
        (dw, _), _ = cv2.getTextSize(direction, self.FONT, 0.72, 2)
        cv2.putText(out, direction, (W//2 - dw//2, by_t),
                    self.FONT, 0.72, d_col, 2)

        info = f"{speed_pct:.0f}%  Sonar:{sonar_cm:.0f}cm"
        (iw, _), _ = cv2.getTextSize(info, self.FONT, 0.55, 2)
        cv2.putText(out, info, (W-iw-4, by_t),
                    self.FONT, 0.55, self.C_WHITE, 2)

        return out


class Navigator:
   

    def __init__(self):
        log.info(f"Loading {CFG.MODEL} …")
        self.yolo = YOLO(CFG.MODEL)
        _d = np.zeros((CFG.INFER_SIZE, CFG.INFER_SIZE, 3), np.uint8)
        self.yolo(_d, verbose=False)
        log.info("YOLO ready.")

        self.motors  = Motors()
        self.sonar   = Sonar()
        self.cam     = ThreadedCamera()
        self.tracker = CentroidTracker()
        self.hud     = HUD()

        self._skip_ctr   = 0
        self._inf_ms     = 0.0
        self._fps_buf    = deque(maxlen=12)
        self._last_t     = time.monotonic()
        self._search_dir = 1
        self._search_t   = time.monotonic()

    def _detect(self, frame: np.ndarray) -> Optional[Tuple]:
        H, W = frame.shape[:2]
        t0 = time.monotonic()
        results = self.yolo(
            frame, classes=[0], conf=CFG.CONF,
            iou=CFG.IOU, imgsz=CFG.INFER_SIZE, verbose=False,
        )
        self._inf_ms = (time.monotonic() - t0) * 1000

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        fcx, fcy = W / 2.0, H / 2.0
        best, best_d = None, float('inf')
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if (y2 - y1) / H < 0.08:
                continue
            dist = math.hypot((x1+x2)/2 - fcx, (y1+y2)/2 - fcy)
            if dist < best_d:
                best_d = dist
                best   = (x1/W, y1/H, x2/W, y2/H)
        return best

    def _control(self,
                 x_dev:      float,
                 h_ratio:    float,
                 has_target: bool,
                 dist_cm:    float) -> Tuple[str, float]:
        """
        Returns (direction_label, speed_pct).
        Drives motors directly.
        Robot NEVER goes backward — sonar is the only stop.
        """

        if dist_cm <= CFG.STOP_CM:
            self.motors.stop()
            return "SONAR STOP", 0.0

        if not has_target:
            if self._search_dir > 0:
                self.motors.spin_search_right(CFG.SEARCH_SPD)
            else:
                self.motors.spin_search_left(CFG.SEARCH_SPD)
            return "SEARCHING", CFG.SEARCH_SPD

        dist_err = h_ratio - CFG.TARGET_H

        if dist_err < -CFG.HTOL:
            gap  = abs(dist_err)
            norm = min(gap / 0.30, 1.0)
            spd  = CFG.SLOW_SPD + norm * (CFG.FWD_SPD - CFG.SLOW_SPD)
            spd  = float(np.clip(spd, CFG.MIN_SPD, CFG.FWD_SPD))

            if dist_cm < CFG.SLOW_CM:
                scale = max(0.35,
                            (dist_cm - CFG.STOP_CM) /
                            (CFG.SLOW_CM - CFG.STOP_CM))
                spd *= scale
                spd  = max(spd, CFG.MIN_SPD)

            if x_dev > CFG.TOLERANCE:
                self.motors.forward_steer_right(spd)
                return "FORWARD+R", spd
            elif x_dev < -CFG.TOLERANCE:
                self.motors.forward_steer_left(spd)
                return "FORWARD+L", spd
            else:
                self.motors.forward(spd)
                return "FORWARD", spd

        elif dist_err > CFG.HTOL:
            self.motors.stop()
            return "TOO CLOSE", 0.0

        else:
            if x_dev > CFG.TOLERANCE:
                self.motors.turn_right(CFG.TURN_SPD)
                return "TURN RIGHT", CFG.TURN_SPD
            elif x_dev < -CFG.TOLERANCE:
                self.motors.turn_left(CFG.TURN_SPD)
                return "TURN LEFT", CFG.TURN_SPD
            else:
                self.motors.stop()
                return "AT TARGET", 0.0

    def run(self):
        self.sonar.start()
        log.info("Navigator running.  Press 'q' to quit.")
        log.info(
            f"CFG: TARGET_H={CFG.TARGET_H}  HTOL={CFG.HTOL}  "
            f"TOL={CFG.TOLERANCE}  FWD={CFG.FWD_SPD}%  "
            f"STOP={CFG.STOP_CM}cm"
        )
        log.info(
            "★ CALIBRATION: Stand at follow distance. "
            "Read H=x.xx on HUD. Set --target-h to that value."
        )

        cam_ms    = inf_ms = 0.0
        fps       = 0.0
        direction = "SEARCHING"
        speed_pct = 0.0
        status    = "NO OBJECT"
        x_dev     = 0.0
        h_ratio   = 0.0

        try:
            while True:
                ok, frame, cam_ms = self.cam.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue

                self._skip_ctr += 1
                if self._skip_ctr > CFG.SKIP:
                    self._skip_ctr = 0
                    bbox_norm = self._detect(frame)
                    self.tracker.update(bbox_norm)
                    inf_ms = self._inf_ms
                else:
                    self.tracker.extrapolate()
                    inf_ms = 0.0

                bbox       = self.tracker.bbox
                has_target = bbox is not None

                if has_target:
                    cx_n    = (bbox[0] + bbox[2]) / 2
                    h_ratio = bbox[3] - bbox[1]
                    x_dev   = cx_n - 0.5

                    dist_err  = h_ratio - CFG.TARGET_H
                    at_target = (abs(x_dev)   <= CFG.TOLERANCE and
                                 abs(dist_err) <= CFG.HTOL)
                    status = "AT TARGET" if at_target else "TRACKING"
                else:
                    x_dev   = 0.0
                    h_ratio = 0.0
                    status  = "NO OBJECT"
                    if time.monotonic() - self._search_t > 3.0:
                        self._search_dir *= -1
                        self._search_t    = time.monotonic()

                direction, speed_pct = self._control(
                    x_dev, h_ratio, has_target, self.sonar.cm
                )

                now = time.monotonic()
                dt  = now - self._last_t
                self._last_t = now
                if dt > 0:
                    self._fps_buf.append(1.0 / dt)
                fps = float(np.mean(self._fps_buf)) if self._fps_buf else 0.0

                if CFG.SHOW:
                    vis = self.hud.render(
                        frame, bbox, x_dev, h_ratio,
                        direction, speed_pct,
                        fps, cam_ms, inf_ms,
                        status, self.sonar.cm,
                    )
                    cv2.imshow("Child-Bot Navigator", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        log.info("Quit.")
                        break

                if self._skip_ctr == 0:
                    dist_err = h_ratio - CFG.TARGET_H
                    arrow = ("FAR→FORWARD" if dist_err < -CFG.HTOL else
                             "NEAR→STOP"  if dist_err > CFG.HTOL  else
                             "ON-TARGET")
                    log.info(
                        f"FPS={fps:.1f}  sonar={self.sonar.cm:.0f}cm  "
                        f"X={x_dev:+.3f}  H={h_ratio:.3f}  "
                        f"err={dist_err:+.3f}({arrow})  "
                        f"→ {direction}  {speed_pct:.0f}%"
                    )

        except KeyboardInterrupt:
            log.info("Interrupted.")
        finally:
            self.shutdown()

    def shutdown(self):
        self.motors.stop()
        self.sonar.stop()
        self.cam.release()
        cv2.destroyAllWindows()
        self.motors.cleanup()
        log.info("Shutdown complete.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Child-Following Robot (Forward-Only)")
    ap.add_argument("--no-display",  action="store_true")
    ap.add_argument("--stop-dist",   type=float, default=CFG.STOP_CM,
                    help=f"Sonar hard-stop cm (default {CFG.STOP_CM})")
    ap.add_argument("--tolerance",   type=float, default=CFG.TOLERANCE,
                    help=f"Left/right dead-zone (default {CFG.TOLERANCE})")
    ap.add_argument("--target-h",    type=float, default=CFG.TARGET_H,
                    help="Bbox height at ideal follow distance. "
                         "Stand at target distance, read H=x.xx on HUD, "
                         "use that value here.")
    ap.add_argument("--htol",        type=float, default=CFG.HTOL,
                    help=f"Distance dead-zone (default {CFG.HTOL})")
    ap.add_argument("--fwd-spd",     type=float, default=CFG.FWD_SPD,
                    help=f"Forward speed %% (default {CFG.FWD_SPD})")
    ap.add_argument("--skip",        type=int,   default=CFG.SKIP)
    ap.add_argument("--model",       type=str,   default=CFG.MODEL)
    args = ap.parse_args()

    CFG.SHOW      = not args.no_display
    CFG.STOP_CM   = args.stop_dist
    CFG.TOLERANCE = args.tolerance
    CFG.TARGET_H  = args.target_h
    CFG.HTOL      = args.htol
    CFG.FWD_SPD   = args.fwd_spd
    CFG.SKIP      = args.skip
    CFG.MODEL     = args.model

    Navigator().run()