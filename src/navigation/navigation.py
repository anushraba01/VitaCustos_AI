"""
navigation.py — Child-Following Robot Navigation System
========================================================
Hardware:
  • Raspberry Pi 4
  • Camera (Pi Camera v2 or USB webcam)
  • HC-SR04 Ultrasonic Distance Sensor
  • 2× L298N Motor Driver (controls 4 DC motors)
  • 4 DC Motors (differential drive: left-pair + right-pair)

Algorithm Stack:
  • YOLOv8-nano  — ultra-lightweight real-time child/person detection
  • DeepSORT      — multi-target tracking (lock onto a single child)
  • Dual PID      — angular (steering) + linear (speed) control loops
  • Safety layer  — ultrasonic hard-stop at ≤60 cm

Dependencies:
  pip install ultralytics opencv-python-headless numpy RPi.GPIO
  pip install deep-sort-realtime

Wiring Reference:
  L298N #1 (LEFT motors)          L298N #2 (RIGHT motors)
  ─────────────────────────       ──────────────────────────
  IN1  → GPIO 17                  IN3  → GPIO 22
  IN2  → GPIO 27                  IN4  → GPIO 23
  ENA  → GPIO 18 (PWM)            ENB  → GPIO 25 (PWM)

  HC-SR04
  ───────
  TRIG → GPIO 24
  ECHO → GPIO 8   (use 1kΩ + 2kΩ voltage divider on ECHO line!)
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

# ── Conditional imports (graceful fallback for dev/test on non-Pi) ──────────
try:
    import RPi.GPIO as GPIO
    ON_PI = True
except ImportError:
    ON_PI = False
    print("[WARN] RPi.GPIO not found — running in SIMULATION mode")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed — install with: pip install ultralytics")

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    print("[WARN] deep-sort-realtime not installed. Falling back to centroid tracking.")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ChildBot")


# ═══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    # ── GPIO Pins ────────────────────────────────────────────────────────
    PIN_IN1:  int = 17    # Left  forward
    PIN_IN2:  int = 27    # Left  backward
    PIN_ENA:  int = 18    # Left  PWM speed

    PIN_IN3:  int = 22    # Right forward
    PIN_IN4:  int = 23    # Right backward
    PIN_ENB:  int = 25    # Right PWM speed

    PIN_TRIG: int = 24    # Ultrasonic trigger
    PIN_ECHO: int = 8     # Ultrasonic echo

    # ── Camera ───────────────────────────────────────────────────────────
    CAMERA_INDEX:   int   = 0
    FRAME_WIDTH:    int   = 640
    FRAME_HEIGHT:   int   = 480
    TARGET_FPS:     int   = 30

    # ── Detection ────────────────────────────────────────────────────────
    YOLO_MODEL:         str   = "yolov8n.pt"   # nano = fastest on Pi
    YOLO_CONF:          float = 0.50
    YOLO_IOU:           float = 0.45
    PERSON_CLASS_ID:    int   = 0              # COCO class 0 = person
    # Prefer detections whose box height suggests a CHILD (shorter stature)
    # Ratio of box_height / frame_height: adult ~0.6+, child ~0.3-0.55
    CHILD_HEIGHT_MAX_RATIO: float = 0.70
    CHILD_HEIGHT_MIN_RATIO: float = 0.10

    # ── Tracking ─────────────────────────────────────────────────────────
    MAX_AGE:        int   = 30    # frames before track is dropped
    N_INIT:         int   = 3     # confirmations needed to establish track
    MAX_COSINE_DIST:float = 0.4

    # ── Safety ───────────────────────────────────────────────────────────
    STOP_DISTANCE_CM:   float = 60.0   # hard stop (ultrasonic)
    SLOW_DISTANCE_CM:   float = 90.0   # start slowing down

    # ── PID — Angular (steering) ─────────────────────────────────────────
    ANG_KP: float = 0.55
    ANG_KI: float = 0.003
    ANG_KD: float = 0.12
    ANG_DEADBAND: float = 0.04  # fraction of frame width (±4% = no turn)

    # ── PID — Linear (forward speed) ─────────────────────────────────────
    LIN_KP: float = 1.10
    LIN_KI: float = 0.005
    LIN_KD: float = 0.20
    # Target: keep child box at ~35% of frame height (comfortable follow distance)
    LIN_TARGET_RATIO: float = 0.35

    # ── Motor PWM ────────────────────────────────────────────────────────
    PWM_FREQ:       int   = 1000   # Hz
    BASE_SPEED:     float = 55.0   # % duty cycle at neutral
    MAX_SPEED:      float = 85.0   # % duty cycle cap
    MIN_SPEED:      float = 30.0   # % below this → stop (dead-zone)
    TURN_BOOST:     float = 25.0   # extra % added to faster side when turning

    # ── Misc ─────────────────────────────────────────────────────────────
    LOOP_DELAY_S:   float = 0.033  # ~30 Hz main loop
    ULTRASONIC_INTERVAL_S: float = 0.10  # read distance every 100 ms


CFG = Config()


# ═══════════════════════════════════════════════════════════════════════════
# 2. PID CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════
class PID:
    """
    Discrete PID with anti-windup, derivative-on-measurement,
    and output clamping.
    """
    def __init__(self, kp: float, ki: float, kd: float,
                 out_min: float = -1.0, out_max: float = 1.0,
                 name: str = "PID"):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.name = name
        self._integral   = 0.0
        self._prev_meas  = None
        self._prev_time  = None

    def reset(self):
        self._integral   = 0.0
        self._prev_meas  = None
        self._prev_time  = None

    def compute(self, setpoint: float, measurement: float) -> float:
        now = time.monotonic()
        error = setpoint - measurement

        if self._prev_time is None:
            dt = CFG.LOOP_DELAY_S
        else:
            dt = now - self._prev_time
            dt = max(dt, 1e-4)

        # Proportional
        p = self.kp * error

        # Integral with anti-windup clamp
        self._integral += error * dt
        i_raw = self.ki * self._integral
        i_clamped = np.clip(i_raw, self.out_min, self.out_max)
        # Back-calculate anti-windup
        if i_raw != 0:
            self._integral *= (i_clamped / i_raw) if abs(i_raw) > 1e-9 else 1.0

        # Derivative on measurement (avoids derivative kick on setpoint change)
        if self._prev_meas is not None:
            d = -self.kd * (measurement - self._prev_meas) / dt
        else:
            d = 0.0

        output = np.clip(p + i_clamped + d, self.out_min, self.out_max)

        self._prev_time = now
        self._prev_meas = measurement
        return float(output)


# ═══════════════════════════════════════════════════════════════════════════
# 3. MOTOR CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════
class MotorController:
    """
    Differential-drive wrapper for 2× L298N (4 motors).
    Left pair  = IN1/IN2/ENA
    Right pair = IN3/IN4/ENB
    """

    def __init__(self):
        self._left_pwm  = None
        self._right_pwm = None
        if ON_PI:
            self._setup_gpio()

    def _setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        pins = [CFG.PIN_IN1, CFG.PIN_IN2, CFG.PIN_ENA,
                CFG.PIN_IN3, CFG.PIN_IN4, CFG.PIN_ENB]
        GPIO.setup(pins, GPIO.OUT, initial=GPIO.LOW)
        self._left_pwm  = GPIO.PWM(CFG.PIN_ENA, CFG.PWM_FREQ)
        self._right_pwm = GPIO.PWM(CFG.PIN_ENB, CFG.PWM_FREQ)
        self._left_pwm.start(0)
        self._right_pwm.start(0)
        log.info("GPIO motors initialised.")

    def _set_left(self, speed: float):
        """speed: -100 … +100  (+ve = forward)"""
        if not ON_PI:
            return
        speed = float(np.clip(speed, -100, 100))
        if speed >= 0:
            GPIO.output(CFG.PIN_IN1, GPIO.HIGH)
            GPIO.output(CFG.PIN_IN2, GPIO.LOW)
        else:
            GPIO.output(CFG.PIN_IN1, GPIO.LOW)
            GPIO.output(CFG.PIN_IN2, GPIO.HIGH)
        self._left_pwm.ChangeDutyCycle(abs(speed))

    def _set_right(self, speed: float):
        """speed: -100 … +100  (+ve = forward)"""
        if not ON_PI:
            return
        speed = float(np.clip(speed, -100, 100))
        if speed >= 0:
            GPIO.output(CFG.PIN_IN3, GPIO.HIGH)
            GPIO.output(CFG.PIN_IN4, GPIO.LOW)
        else:
            GPIO.output(CFG.PIN_IN3, GPIO.LOW)
            GPIO.output(CFG.PIN_IN4, GPIO.HIGH)
        self._right_pwm.ChangeDutyCycle(abs(speed))

    def drive(self, linear: float, angular: float):
        """
        linear  : -1.0 (back) … +1.0 (forward)
        angular : -1.0 (turn left) … +1.0 (turn right)
        """
        base  = linear  * CFG.BASE_SPEED
        turn  = angular * CFG.TURN_BOOST

        left_speed  = np.clip(base - turn, -CFG.MAX_SPEED, CFG.MAX_SPEED)
        right_speed = np.clip(base + turn, -CFG.MAX_SPEED, CFG.MAX_SPEED)

        # Dead-zone: ignore very small commands
        if abs(left_speed)  < CFG.MIN_SPEED: left_speed  = 0
        if abs(right_speed) < CFG.MIN_SPEED: right_speed = 0

        self._set_left(left_speed)
        self._set_right(right_speed)

        if not ON_PI:
            log.debug(f"[SIM] L={left_speed:+.1f}  R={right_speed:+.1f}")

    def stop(self):
        self._set_left(0)
        self._set_right(0)
        if not ON_PI:
            log.debug("[SIM] STOP")

    def cleanup(self):
        self.stop()
        if ON_PI:
            if self._left_pwm:  self._left_pwm.stop()
            if self._right_pwm: self._right_pwm.stop()
            GPIO.cleanup()
            log.info("GPIO cleaned up.")


# ═══════════════════════════════════════════════════════════════════════════
# 4. ULTRASONIC SENSOR  (runs in background thread)
# ═══════════════════════════════════════════════════════════════════════════
class UltrasonicSensor:
    """
    Non-blocking HC-SR04 reader.
    Median-filtered over a rolling window to reject noise spikes.
    """
    _WINDOW = 5
    _TIMEOUT = 0.04   # 40 ms = ~6.8 m max range

    def __init__(self):
        self._distance_cm = 999.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        if ON_PI:
            GPIO.setup(CFG.PIN_TRIG, GPIO.OUT)
            GPIO.setup(CFG.PIN_ECHO, GPIO.IN)
            GPIO.output(CFG.PIN_TRIG, GPIO.LOW)
            time.sleep(0.05)

        self._history: deque = deque(maxlen=self._WINDOW)

    def _measure_once(self) -> float:
        if not ON_PI:
            return 999.0
        GPIO.output(CFG.PIN_TRIG, GPIO.HIGH)
        time.sleep(0.00001)   # 10 µs pulse
        GPIO.output(CFG.PIN_TRIG, GPIO.LOW)

        t0 = time.monotonic()
        while GPIO.input(CFG.PIN_ECHO) == 0:
            if time.monotonic() - t0 > self._TIMEOUT:
                return 999.0
        pulse_start = time.monotonic()

        while GPIO.input(CFG.PIN_ECHO) == 1:
            if time.monotonic() - pulse_start > self._TIMEOUT:
                return 999.0
        pulse_end = time.monotonic()

        distance = (pulse_end - pulse_start) * 17150  # cm (speed of sound / 2)
        return round(distance, 1)

    def _run(self):
        while self._running:
            d = self._measure_once()
            if 2 < d < 400:           # sanity check (2 cm – 4 m)
                self._history.append(d)
            if self._history:
                with self._lock:
                    self._distance_cm = float(np.median(self._history))
            time.sleep(CFG.ULTRASONIC_INTERVAL_S)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    @property
    def distance_cm(self) -> float:
        with self._lock:
            return self._distance_cm


# ═══════════════════════════════════════════════════════════════════════════
# 5. CHILD DETECTOR  (YOLOv8-nano + optional DeepSORT)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class Detection:
    x1: float; y1: float; x2: float; y2: float
    confidence: float
    track_id: int = -1

    @property
    def cx(self) -> float:   return (self.x1 + self.x2) / 2
    @property
    def cy(self) -> float:   return (self.y1 + self.y2) / 2
    @property
    def width(self)  -> float: return self.x2 - self.x1
    @property
    def height(self) -> float: return self.y2 - self.y1
    @property
    def area(self)   -> float: return self.width * self.height


class ChildDetector:
    """
    YOLOv8-nano inference + DeepSORT tracking.
    Scores detections to prefer child-sized persons.
    Locks onto one track ID and only reports that child.
    """

    def __init__(self):
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics is required. Run: pip install ultralytics")

        log.info(f"Loading YOLO model: {CFG.YOLO_MODEL} …")
        self.model = YOLO(CFG.YOLO_MODEL)
        # Warm-up: first inference is always slow
        dummy = np.zeros((CFG.FRAME_HEIGHT, CFG.FRAME_WIDTH, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        log.info("YOLO model ready.")

        self.tracker = None
        if DEEPSORT_AVAILABLE:
            self.tracker = DeepSort(
                max_age=CFG.MAX_AGE,
                n_init=CFG.N_INIT,
                max_cosine_distance=CFG.MAX_COSINE_DIST,
                nn_budget=100,
                embedder="mobilenet",        # lightweight re-ID backbone
                embedder_gpu=False,
            )
            log.info("DeepSORT tracker ready.")
        else:
            log.warning("DeepSORT unavailable — using nearest-centroid tracker.")

        self._locked_id: int   = -1    # track ID we are following
        self._lost_frames: int = 0
        self._LOST_THRESHOLD = 20

    def _score_detection(self, det: Detection, frame_h: int) -> float:
        """
        Heuristic score: prefer mid-frame, child-height-ratio detections.
        Higher = better candidate to follow.
        """
        h_ratio = det.height / frame_h
        in_range = CFG.CHILD_HEIGHT_MIN_RATIO < h_ratio < CFG.CHILD_HEIGHT_MAX_RATIO
        return det.confidence * (1.2 if in_range else 0.5)

    def detect(self, frame: np.ndarray) -> Optional[Detection]:
        fh, fw = frame.shape[:2]

        # ── YOLO inference ───────────────────────────────────────────────
        results = self.model(
            frame,
            classes=[CFG.PERSON_CLASS_ID],
            conf=CFG.YOLO_CONF,
            iou=CFG.YOLO_IOU,
            imgsz=320,        # downsample internally → faster on Pi
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            self._lost_frames += 1
            if self._lost_frames > self._LOST_THRESHOLD:
                self._locked_id = -1
            return None

        raw_dets = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            raw_dets.append(Detection(x1, y1, x2, y2, conf))

        # ── DeepSORT tracking ────────────────────────────────────────────
        if self.tracker is not None:
            ds_input = [
                ([d.x1, d.y1, d.width, d.height], d.confidence, "person")
                for d in raw_dets
            ]
            tracks = self.tracker.update_tracks(ds_input, frame=frame)
            confirmed = [t for t in tracks if t.is_confirmed()]

            if not confirmed:
                self._lost_frames += 1
                if self._lost_frames > self._LOST_THRESHOLD:
                    self._locked_id = -1
                return None

            # Build Detection objects from confirmed tracks
            tracked_dets = []
            for t in confirmed:
                ltrb = t.to_ltrb()
                d = Detection(ltrb[0], ltrb[1], ltrb[2], ltrb[3],
                              confidence=1.0, track_id=t.track_id)
                tracked_dets.append(d)

            # Lock onto a target
            if self._locked_id == -1:
                # Pick best first candidate
                best = max(tracked_dets,
                           key=lambda d: self._score_detection(d, fh))
                self._locked_id = best.track_id
                log.info(f"Locked onto track_id={self._locked_id}")

            # Try to find our locked target
            target = next((d for d in tracked_dets
                           if d.track_id == self._locked_id), None)

            if target is None:
                self._lost_frames += 1
                if self._lost_frames > self._LOST_THRESHOLD:
                    self._locked_id = -1
                    log.info("Target lost — searching for new child.")
                return None

            self._lost_frames = 0
            return target

        # ── Fallback: nearest-centroid (no DeepSORT) ─────────────────────
        best = max(raw_dets, key=lambda d: self._score_detection(d, fh))
        self._lost_frames = 0
        return best


# ═══════════════════════════════════════════════════════════════════════════
# 6. NAVIGATOR  (fuses perception + PID + motor commands)
# ═══════════════════════════════════════════════════════════════════════════
class Navigator:
    """
    Main navigation loop.

    Angular PID:
      setpoint = frame center (0.5)
      measurement = child cx / frame_width
      output → steering (angular command)

    Linear PID:
      setpoint = CFG.LIN_TARGET_RATIO   (desired box height as % of frame)
      measurement = child box_height / frame_height
      output → forward speed (linear command)

    Hard stop: if ultrasonic ≤ STOP_DISTANCE_CM → stop motors immediately.
    """

    def __init__(self):
        self.motor    = MotorController()
        self.sonar    = UltrasonicSensor()
        self.detector = ChildDetector()

        self.pid_ang = PID(CFG.ANG_KP, CFG.ANG_KI, CFG.ANG_KD,
                           out_min=-1.0, out_max=1.0, name="Angular")
        self.pid_lin = PID(CFG.LIN_KP, CFG.LIN_KI, CFG.LIN_KD,
                           out_min=-1.0, out_max=1.0, name="Linear")

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False

        # Stats
        self._frame_count = 0
        self._fps_start   = time.monotonic()

    # ── Camera setup ─────────────────────────────────────────────────────
    def _open_camera(self):
        self._cap = cv2.VideoCapture(CFG.CAMERA_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG.FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.FRAME_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS,          CFG.TARGET_FPS)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # low latency
        if not self._cap.isOpened():
            raise RuntimeError("Cannot open camera!")
        log.info(f"Camera opened (index={CFG.CAMERA_INDEX}).")

    # ── HUD overlay ──────────────────────────────────────────────────────
    def _draw_hud(self, frame: np.ndarray,
                  det: Optional[Detection],
                  linear: float, angular: float,
                  dist_cm: float, fps: float) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Frame centre crosshair
        cv2.line(overlay, (w//2, 0), (w//2, h), (0, 255, 255), 1)

        if det:
            # Bounding box
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 0), 2)
            # Centroid
            cv2.circle(overlay, (int(det.cx), int(det.cy)), 6, (0, 200, 0), -1)
            label = f"Child #{det.track_id}  {det.confidence:.0%}"
            cv2.putText(overlay, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # Stats panel
        stats = [
            f"FPS: {fps:.1f}",
            f"Dist: {dist_cm:.1f} cm",
            f"Lin: {linear:+.2f}",
            f"Ang: {angular:+.2f}",
        ]
        if dist_cm <= CFG.STOP_DISTANCE_CM:
            stats.append("!! HARD STOP !!")
        for i, s in enumerate(stats):
            color = (0, 0, 255) if "STOP" in s else (255, 255, 255)
            cv2.putText(overlay, s, (8, 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    # ── One control tick ─────────────────────────────────────────────────
    def _control_tick(self, det: Optional[Detection],
                      frame_w: int, frame_h: int) -> Tuple[float, float]:
        """
        Returns (linear, angular) both in [-1, +1].
        """
        dist_cm = self.sonar.distance_cm

        # ── Ultrasonic hard stop ─────────────────────────────────────────
        if dist_cm <= CFG.STOP_DISTANCE_CM:
            self.motor.stop()
            self.pid_ang.reset()
            self.pid_lin.reset()
            return 0.0, 0.0

        # ── No target → spin-search slowly ──────────────────────────────
        if det is None:
            self.motor.drive(linear=0.0, angular=0.3)   # gentle spin
            return 0.0, 0.3

        # ── Angular PID ──────────────────────────────────────────────────
        cx_norm = det.cx / frame_w          # 0.0 (left) … 1.0 (right)
        ang_error = cx_norm - 0.5           # <0 = child left, >0 = child right

        if abs(ang_error) < CFG.ANG_DEADBAND:
            angular = 0.0
            self.pid_ang.reset()
        else:
            angular = self.pid_ang.compute(setpoint=0.5, measurement=cx_norm)

        # ── Linear PID ───────────────────────────────────────────────────
        # Slow down when also close per ultrasonic
        speed_scale = 1.0
        if dist_cm < CFG.SLOW_DISTANCE_CM:
            t = (dist_cm - CFG.STOP_DISTANCE_CM) / (CFG.SLOW_DISTANCE_CM - CFG.STOP_DISTANCE_CM)
            speed_scale = max(0.2, float(t))

        h_ratio = det.height / frame_h
        linear_raw = self.pid_lin.compute(
            setpoint=CFG.LIN_TARGET_RATIO,
            measurement=h_ratio,
        )
        linear = float(np.clip(linear_raw * speed_scale, -1.0, 1.0))

        # If child is far off-centre, reduce forward motion to turn first
        if abs(ang_error) > 0.25:
            linear *= max(0.0, 1.0 - abs(ang_error) * 1.5)

        self.motor.drive(linear=linear, angular=angular)
        return linear, angular

    # ── Main loop ────────────────────────────────────────────────────────
    def run(self, show_preview: bool = True):
        self._open_camera()
        self.sonar.start()
        self._running = True

        log.info("Navigator started. Press 'q' to quit.")

        linear, angular = 0.0, 0.0
        fps = 0.0

        try:
            while self._running:
                t_loop = time.monotonic()

                ret, frame = self._cap.read()
                if not ret:
                    log.warning("Dropped frame.")
                    continue

                fh, fw = frame.shape[:2]

                # ── Detect & track ───────────────────────────────────────
                det = self.detector.detect(frame)

                # ── Control ──────────────────────────────────────────────
                linear, angular = self._control_tick(det, fw, fh)

                # ── FPS ──────────────────────────────────────────────────
                self._frame_count += 1
                elapsed = time.monotonic() - self._fps_start
                if elapsed >= 1.0:
                    fps = self._frame_count / elapsed
                    self._frame_count = 0
                    self._fps_start = time.monotonic()

                # ── Preview window ───────────────────────────────────────
                if show_preview:
                    hud = self._draw_hud(frame, det, linear, angular,
                                         self.sonar.distance_cm, fps)
                    cv2.imshow("Child-Bot Navigator", hud)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        log.info("User quit.")
                        break

                # ── Loop timing ──────────────────────────────────────────
                elapsed_loop = time.monotonic() - t_loop
                sleep_t = CFG.LOOP_DELAY_S - elapsed_loop
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except KeyboardInterrupt:
            log.info("KeyboardInterrupt — shutting down.")
        finally:
            self.shutdown()

    def shutdown(self):
        self._running = False
        self.motor.stop()
        self.sonar.stop()
        if self._cap:
            self._cap.release()
        cv2.destroyAllWindows()
        self.motor.cleanup()
        log.info("Navigator shut down cleanly.")


# ═══════════════════════════════════════════════════════════════════════════
# 7. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Child-Following Robot")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable OpenCV window (headless mode)")
    parser.add_argument("--stop-dist", type=float, default=CFG.STOP_DISTANCE_CM,
                        help=f"Hard-stop distance in cm (default: {CFG.STOP_DISTANCE_CM})")
    parser.add_argument("--model", type=str, default=CFG.YOLO_MODEL,
                        help="YOLOv8 model file (yolov8n.pt recommended for Pi)")
    args = parser.parse_args()

    # Apply CLI overrides
    CFG.STOP_DISTANCE_CM = args.stop_dist
    CFG.YOLO_MODEL       = args.model

    nav = Navigator()
    nav.run(show_preview=not args.no_preview)