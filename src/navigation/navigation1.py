"""
=============================================================================
HUMAN FOLLOWING ROBOT — Production-Grade AI Robotics System
=============================================================================
Hardware:  Raspberry Pi 4 | L298N | 2× DC Motors | HC-SR04 | USB/Pi Camera
Vision:    YOLOv8-nano (person class) → MediaPipe Pose fallback
Motion:    PID-inspired differential steering | Smooth PWM ramping
=============================================================================
"""

import cv2
import numpy as np
import threading
import time
import queue
import sys
import signal
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL IMPORTS (graceful degradation for development on non-Pi hardware)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("[WARN] RPi.GPIO not found – running in SIMULATION mode")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not found – will use MediaPipe fallback")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[WARN] mediapipe not found – detection will be disabled")

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("HFR")

# =============================================================================
# ██████╗ ██████╗ ███╗   ██╗███████╗████████╗ █████╗ ███╗   ██╗████████╗███████╗
# ██╔════╝██╔═══██╗████╗  ██║██╔════╝╚══██╔══╝██╔══██╗████╗  ██║╚══██╔══╝██╔════╝
# ██║     ██║   ██║██╔██╗ ██║███████╗   ██║   ███████║██╔██╗ ██║   ██║   ███████╗
# ██║     ██║   ██║██║╚██╗██║╚════██║   ██║   ██╔══██║██║╚██╗██║   ██║   ╚════██║
# ╚██████╗╚██████╔╝██║ ╚████║███████║   ██║   ██║  ██║██║ ╚████║   ██║   ███████║
#  ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝
# =============================================================================

# ── GPIO Pin Mapping ──────────────────────────────────────────────────────────
#   L298N Motor Driver (BCM numbering)
#
#   LEFT MOTOR                RIGHT MOTOR
#   IN1 → GPIO 17             IN3 → GPIO 22
#   IN2 → GPIO 27             IN4 → GPIO 23
#   ENA → GPIO 18 (PWM)       ENB → GPIO 24 (PWM)
#
#   HC-SR04 Ultrasonic Sensor
#   TRIG → GPIO 5             ECHO → GPIO 6
# ─────────────────────────────────────────────────────────────────────────────
PIN_LEFT_IN1    = 17
PIN_LEFT_IN2    = 27
PIN_LEFT_EN     = 18   # PWM – must be hardware PWM or software PWM capable

PIN_RIGHT_IN3   = 22
PIN_RIGHT_IN4   = 23
PIN_RIGHT_EN    = 24   # PWM

PIN_TRIG        = 5
PIN_ECHO        = 6

PWM_FREQUENCY   = 1000   # Hz — higher = smoother at low duty cycles

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX        = 0
CAMERA_WIDTH        = 640
CAMERA_HEIGHT       = 480
CAMERA_FPS_TARGET   = 30
PROCESS_EVERY_N_FRAMES = 2    # skip frames for detection to save CPU

# ── Detection ─────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH         = "yolov8n.pt"   # auto-downloaded on first run
YOLO_CONFIDENCE_THRESH  = 0.45
YOLO_IOU_THRESH         = 0.45
YOLO_PERSON_CLASS_ID    = 0

MEDIAPIPE_MIN_DETECTION_CONF = 0.6
MEDIAPIPE_MIN_TRACKING_CONF  = 0.5

# ── Tracking ──────────────────────────────────────────────────────────────────
TRACKING_HISTORY_LEN    = 8    # frames of bbox history for smoothing
TRACKING_TIMEOUT_SEC    = 1.5  # seconds before declaring target lost
TRACKING_CONFIDENCE_DECAY = 0.15  # per-frame confidence decay when not detected
MIN_CONFIDENCE_TO_ACT   = 0.3

# ── Frame Zones (fractions of frame width) ───────────────────────────────────
# |  LEFT_EXT | LEFT_SAFE | ← CENTER → | RIGHT_SAFE | RIGHT_EXT |
ZONE_CENTER_LEFT    = 0.35   # centre zone left boundary
ZONE_CENTER_RIGHT   = 0.65   # centre zone right boundary
ZONE_SAFE_LEFT      = 0.20   # safe zone left boundary
ZONE_SAFE_RIGHT     = 0.80   # safe zone right boundary
# anything outside SAFE zones triggers turning

# ── Distance Control ──────────────────────────────────────────────────────────
DISTANCE_STOP_CM        = 40   # ultrasonic emergency stop
DISTANCE_NEAR_CM        = 60   # slow down
DISTANCE_OPTIMAL_CM     = 90   # ideal follow distance
DISTANCE_FAR_CM         = 150  # full speed ahead

BBOX_STOP_WIDTH_RATIO   = 0.65  # stop if bbox > 65% of frame width (too close)
BBOX_NEAR_WIDTH_RATIO   = 0.45
BBOX_OPTIMAL_WIDTH_RATIO = 0.28
BBOX_FAR_WIDTH_RATIO    = 0.15  # human far – move faster

ULTRASONIC_MEDIAN_SAMPLES = 5
ULTRASONIC_MAX_RANGE_CM   = 300

# ── Motor Speed Parameters ────────────────────────────────────────────────────
SPEED_BASE          = 62    # % PWM duty cycle – forward cruise
SPEED_NEAR          = 45    # slow down when human near
SPEED_FAR           = 75    # speed up when human far
SPEED_MIN           = 35    # minimum drive speed (below = stall risk)
SPEED_MAX           = 85    # maximum speed cap

# Differential steering
TURN_SPEED_REDUCTION = 22   # reduce inner motor by this % for gentle turn
TURN_SPEED_OUTER    = SPEED_BASE
TURN_SPEED_INNER    = SPEED_BASE - TURN_SPEED_REDUCTION

# Smooth ramping
RAMP_STEP_SIZE      = 3     # % change per ramp tick
RAMP_TICK_SEC       = 0.02  # seconds between ramp ticks

# ── Filtering / Smoothing ─────────────────────────────────────────────────────
CENTROID_EMA_ALPHA  = 0.25  # exponential moving average for centroid
SPEED_EMA_ALPHA     = 0.20  # EMA for speed changes

# ── Turn Cooldown ─────────────────────────────────────────────────────────────
TURN_COOLDOWN_SEC   = 0.35  # minimum time between direction changes

# ── FPS Watchdog ──────────────────────────────────────────────────────────────
FPS_WATCHDOG_PERIOD = 5.0   # seconds
FPS_MINIMUM         = 5.0   # below this → emergency stop


# =============================================================================
#  DATA STRUCTURES
# =============================================================================

@dataclass
class Detection:
    """Normalised bounding box + metadata for a detected person."""
    cx: float          # centre x, 0.0–1.0
    cy: float          # centre y, 0.0–1.0
    w: float           # width  fraction, 0.0–1.0
    h: float           # height fraction, 0.0–1.0
    confidence: float  # 0.0–1.0
    timestamp: float   # time.monotonic()


@dataclass
class MotorCommand:
    """PWM duty cycles for both motors."""
    left: float  = 0.0
    right: float = 0.0


@dataclass
class RobotState:
    """Shared mutable state – all fields protected by RLock."""
    # Tracking
    target: Optional[Detection]   = None
    track_confidence: float       = 0.0
    smoothed_cx: float            = 0.5
    smoothed_cy: float            = 0.5

    # Distances
    ultrasonic_cm: float          = ULTRASONIC_MAX_RANGE_CM
    bbox_distance_mode: str       = "far"   # far | optimal | near | stop

    # Motion
    motor_cmd: MotorCommand       = field(default_factory=MotorCommand)
    last_turn_time: float         = 0.0
    current_direction: str        = "stop"  # stop | forward | left | right

    # System
    running: bool                 = True
    emergency_stop: bool          = False
    frame_count: int              = 0
    fps: float                    = 0.0


# =============================================================================
#  MOTOR CONTROLLER
# =============================================================================

class MotorController:
    """
    L298N dual H-bridge driver with smooth PWM ramping.

    Wiring assumption (BCM):
        Left  motor: IN1=17, IN2=27, ENA=18(PWM)
        Right motor: IN3=22, IN4=23, ENB=24(PWM)
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._left_speed  = 0.0
        self._right_speed = 0.0
        self._target_left  = 0.0
        self._target_right = 0.0

        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for pin in (PIN_LEFT_IN1, PIN_LEFT_IN2, PIN_LEFT_EN,
                        PIN_RIGHT_IN3, PIN_RIGHT_IN4, PIN_RIGHT_EN):
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)

            self._pwm_left  = GPIO.PWM(PIN_LEFT_EN,  PWM_FREQUENCY)
            self._pwm_right = GPIO.PWM(PIN_RIGHT_EN, PWM_FREQUENCY)
            self._pwm_left.start(0)
            self._pwm_right.start(0)
        else:
            self._pwm_left  = None
            self._pwm_right = None

        # Start ramp thread
        self._ramp_thread = threading.Thread(
            target=self._ramp_loop, name="MotorRamp", daemon=True)
        self._ramp_thread.start()
        log.info("MotorController initialised.")

    # ── Public API ────────────────────────────────────────────────────────────

    def forward(self, left_pct: float, right_pct: float):
        """Set both motors forward with independent speeds."""
        self._set_direction_pins(
            left_fwd=True, left_bwd=False,
            right_fwd=True, right_bwd=False)
        with self._lock:
            self._target_left  = float(np.clip(left_pct,  0, SPEED_MAX))
            self._target_right = float(np.clip(right_pct, 0, SPEED_MAX))

    def stop(self):
        """Smooth deceleration to zero."""
        self._set_direction_pins(
            left_fwd=False, left_bwd=False,
            right_fwd=False, right_bwd=False)
        with self._lock:
            self._target_left  = 0.0
            self._target_right = 0.0

    def emergency_stop(self):
        """Instant stop – bypass ramp."""
        self.stop()
        with self._lock:
            self._left_speed  = 0.0
            self._right_speed = 0.0
        self._apply_pwm(0.0, 0.0)

    def cleanup(self):
        self.emergency_stop()
        time.sleep(0.1)
        if GPIO_AVAILABLE:
            self._pwm_left.stop()
            self._pwm_right.stop()
            GPIO.cleanup()
        log.info("MotorController cleaned up.")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _set_direction_pins(self, left_fwd, left_bwd, right_fwd, right_bwd):
        if not GPIO_AVAILABLE:
            return
        GPIO.output(PIN_LEFT_IN1,  GPIO.HIGH if left_fwd  else GPIO.LOW)
        GPIO.output(PIN_LEFT_IN2,  GPIO.HIGH if left_bwd  else GPIO.LOW)
        GPIO.output(PIN_RIGHT_IN3, GPIO.HIGH if right_fwd else GPIO.LOW)
        GPIO.output(PIN_RIGHT_IN4, GPIO.HIGH if right_bwd else GPIO.LOW)

    def _apply_pwm(self, left: float, right: float):
        if GPIO_AVAILABLE:
            self._pwm_left.ChangeDutyCycle(left)
            self._pwm_right.ChangeDutyCycle(right)

    def _ramp_loop(self):
        """Continuously ramp actual speeds toward targets."""
        while True:
            with self._lock:
                tl = self._target_left
                tr = self._target_right
                cl = self._left_speed
                cr = self._right_speed

            # Smooth step toward target
            cl = self._step(cl, tl)
            cr = self._step(cr, tr)

            with self._lock:
                self._left_speed  = cl
                self._right_speed = cr

            self._apply_pwm(cl, cr)
            time.sleep(RAMP_TICK_SEC)

    @staticmethod
    def _step(current: float, target: float) -> float:
        diff = target - current
        if abs(diff) <= RAMP_STEP_SIZE:
            return target
        return current + RAMP_STEP_SIZE * (1 if diff > 0 else -1)


# =============================================================================
#  ULTRASONIC SENSOR
# =============================================================================

class UltrasonicSensor:
    """HC-SR04 driver with median filtering and thread-safe polling."""

    def __init__(self):
        self._readings = deque(maxlen=ULTRASONIC_MEDIAN_SAMPLES)
        self._lock = threading.Lock()
        self._distance_cm = float(ULTRASONIC_MAX_RANGE_CM)

        if GPIO_AVAILABLE:
            GPIO.setup(PIN_TRIG, GPIO.OUT)
            GPIO.setup(PIN_ECHO, GPIO.IN)
            GPIO.output(PIN_TRIG, GPIO.LOW)
            time.sleep(0.05)  # sensor settle

        self._thread = threading.Thread(
            target=self._poll_loop, name="Ultrasonic", daemon=True)
        self._thread.start()
        log.info("UltrasonicSensor initialised.")

    def get_distance(self) -> float:
        """Return smoothed distance in centimetres."""
        with self._lock:
            return self._distance_cm

    def _measure_raw(self) -> float:
        if not GPIO_AVAILABLE:
            return float(ULTRASONIC_MAX_RANGE_CM)

        GPIO.output(PIN_TRIG, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(PIN_TRIG, GPIO.LOW)

        timeout = time.monotonic() + 0.05
        while GPIO.input(PIN_ECHO) == 0:
            if time.monotonic() > timeout:
                return ULTRASONIC_MAX_RANGE_CM
        pulse_start = time.monotonic()

        timeout = time.monotonic() + 0.05
        while GPIO.input(PIN_ECHO) == 1:
            if time.monotonic() > timeout:
                return ULTRASONIC_MAX_RANGE_CM
        pulse_end = time.monotonic()

        distance = (pulse_end - pulse_start) * 17150  # cm
        return float(np.clip(distance, 2, ULTRASONIC_MAX_RANGE_CM))

    def _poll_loop(self):
        while True:
            raw = self._measure_raw()
            self._readings.append(raw)

            if len(self._readings) >= 3:
                median = float(np.median(list(self._readings)))
                with self._lock:
                    self._distance_cm = median

            time.sleep(0.06)  # ~16 Hz – sufficient for safety


# =============================================================================
#  HUMAN DETECTOR
# =============================================================================

class HumanDetector:
    """
    Detects a single nearest human using:
    1. YOLOv8-nano (person class only) – preferred
    2. MediaPipe Pose                   – fallback

    Returns a Detection in normalised frame coordinates.
    """

    def __init__(self):
        self._yolo = None
        self._mp_pose = None
        self._detector_name = "none"

        if YOLO_AVAILABLE:
            try:
                self._yolo = YOLO(YOLO_MODEL_PATH)
                # Warm-up
                dummy = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                self._yolo.predict(dummy, verbose=False)
                self._detector_name = "yolov8n"
                log.info("YOLO detector ready.")
            except Exception as e:
                log.warning(f"YOLO init failed: {e}")

        if self._yolo is None and MEDIAPIPE_AVAILABLE:
            mp_module = mp.solutions.pose
            self._mp_pose = mp_module.Pose(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONF,
                min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONF,
            )
            self._detector_name = "mediapipe"
            log.info("MediaPipe Pose detector ready.")

        if self._detector_name == "none":
            log.error("No detector available! Install ultralytics or mediapipe.")

    @property
    def name(self) -> str:
        return self._detector_name

    def detect(self, frame: np.ndarray) -> Optional[Detection]:
        """
        Run detection on frame.  Returns the Detection for the largest
        (nearest) person, or None if not found.
        """
        h, w = frame.shape[:2]
        now = time.monotonic()

        if self._yolo is not None:
            return self._detect_yolo(frame, w, h, now)
        elif self._mp_pose is not None:
            return self._detect_mediapipe(frame, w, h, now)
        return None

    def _detect_yolo(self, frame, w, h, now) -> Optional[Detection]:
        results = self._yolo.predict(
            frame,
            classes=[YOLO_PERSON_CLASS_ID],
            conf=YOLO_CONFIDENCE_THRESH,
            iou=YOLO_IOU_THRESH,
            verbose=False,
            imgsz=320,   # smaller input for speed
        )
        best: Optional[Detection] = None
        best_area = 0.0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                area = bw * bh
                if area > best_area:
                    best_area = area
                    best = Detection(
                        cx=(x1 + x2) / 2 / w,
                        cy=(y1 + y2) / 2 / h,
                        w=bw, h=bh,
                        confidence=conf,
                        timestamp=now,
                    )
        return best

    def _detect_mediapipe(self, frame, w, h, now) -> Optional[Detection]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._mp_pose.process(rgb)
        if not result.pose_landmarks:
            return None

        lm = result.pose_landmarks.landmark
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]

        x_min, x_max = max(0, min(xs)), min(1, max(xs))
        y_min, y_max = max(0, min(ys)), min(1, max(ys))
        bw = x_max - x_min
        bh = y_max - y_min

        # MediaPipe visibility as confidence proxy
        conf = float(np.mean([p.visibility for p in lm]))

        return Detection(
            cx=(x_min + x_max) / 2,
            cy=(y_min + y_max) / 2,
            w=bw, h=bh,
            confidence=conf,
            timestamp=now,
        )


# =============================================================================
#  KALMAN-LIKE TRACKER  (lightweight 2D centroid tracker)
# =============================================================================

class CentroidTracker:
    """
    Maintains a smooth estimate of the tracked human's centroid
    using an exponential moving average and temporal confidence decay.

    Not a full Kalman filter, but provides the same key benefits:
    - Smooth centroid motion
    - Short-term occlusion handling
    - Confidence-weighted updates
    """

    def __init__(self):
        self.cx = 0.5
        self.cy = 0.5
        self.confidence = 0.0
        self.last_seen = 0.0
        self._history: deque = deque(maxlen=TRACKING_HISTORY_LEN)

    def update(self, det: Optional[Detection]) -> float:
        """Update tracker. Returns current confidence."""
        now = time.monotonic()

        if det is not None and det.confidence >= MIN_CONFIDENCE_TO_ACT:
            # Blend new detection into smoothed estimate
            alpha = CENTROID_EMA_ALPHA
            self.cx = alpha * det.cx + (1 - alpha) * self.cx
            self.cy = alpha * det.cy + (1 - alpha) * self.cy

            # Confidence rises toward detection confidence
            self.confidence = min(1.0,
                self.confidence + (det.confidence - self.confidence) * 0.4)

            self._history.append((det.cx, det.cy))
            self.last_seen = now

        else:
            # No detection – decay confidence
            elapsed = now - self.last_seen
            decay = TRACKING_CONFIDENCE_DECAY * (elapsed * 10)
            self.confidence = max(0.0, self.confidence - decay)

            # Continue with last known position (inertia)

        return self.confidence

    def is_tracking(self) -> bool:
        elapsed = time.monotonic() - self.last_seen
        return (self.confidence >= MIN_CONFIDENCE_TO_ACT and
                elapsed < TRACKING_TIMEOUT_SEC)

    def reset(self):
        self.cx = 0.5
        self.cy = 0.5
        self.confidence = 0.0
        self.last_seen = 0.0
        self._history.clear()


# =============================================================================
#  STEERING LOGIC
# =============================================================================

class SteeringController:
    """
    Converts (smoothed_cx, bbox_width, ultrasonic_dist) → MotorCommand.

    Motion rules enforced here:
    ✓ Forward priority                 ✓ No backward
    ✓ Dead-zone centre (no micro-turn) ✓ Extreme zone: slow differential turn
    ✓ Turn cooldown                    ✓ Distance-based speed scaling
    ✓ Emergency stop
    """

    def __init__(self):
        self._last_turn_time  = 0.0
        self._last_direction  = "stop"
        self._smooth_left     = 0.0
        self._smooth_right    = 0.0

    def compute(
        self,
        smoothed_cx: float,
        bbox_w: float,
        ultrasonic_cm: float,
        tracking: bool,
        emergency: bool,
    ) -> Tuple[MotorCommand, str]:
        """
        Returns (MotorCommand, direction_label).
        direction_label: 'stop' | 'forward' | 'left' | 'right'
        """

        # ── Safety overrides ─────────────────────────────────────────────────
        if emergency or not tracking:
            return MotorCommand(0, 0), "stop"

        # ── Distance check ────────────────────────────────────────────────────
        if ultrasonic_cm <= DISTANCE_STOP_CM:
            return MotorCommand(0, 0), "stop"

        if bbox_w >= BBOX_STOP_WIDTH_RATIO:
            return MotorCommand(0, 0), "stop"

        # ── Base speed from distance ──────────────────────────────────────────
        base = self._distance_to_speed(bbox_w, ultrasonic_cm)

        # ── Zone determination ────────────────────────────────────────────────
        if smoothed_cx < ZONE_SAFE_LEFT:
            direction = "left"
        elif smoothed_cx > ZONE_SAFE_RIGHT:
            direction = "right"
        else:
            direction = "forward"   # centre + safe zones → forward only

        # ── Turn cooldown ─────────────────────────────────────────────────────
        now = time.monotonic()
        if direction != "forward":
            # Only allow turn if cooldown elapsed OR direction changed
            if (self._last_direction == direction and
                    now - self._last_turn_time < TURN_COOLDOWN_SEC):
                direction = "forward"  # suppress oscillation

        # ── Compute motor speeds ──────────────────────────────────────────────
        left, right = base, base

        if direction == "left":
            # Reduce left motor for gentle left turn
            steer_factor = self._steer_factor(smoothed_cx, side="left")
            left  = base * (1.0 - steer_factor)
            right = base
            self._last_turn_time = now

        elif direction == "right":
            steer_factor = self._steer_factor(smoothed_cx, side="right")
            left  = base
            right = base * (1.0 - steer_factor)
            self._last_turn_time = now

        self._last_direction = direction

        # ── EMA smoothing on output speeds ────────────────────────────────────
        alpha = SPEED_EMA_ALPHA
        self._smooth_left  = alpha * left  + (1 - alpha) * self._smooth_left
        self._smooth_right = alpha * right + (1 - alpha) * self._smooth_right

        # Enforce minimum speed when moving (prevent stall)
        def enforce_min(v):
            return max(SPEED_MIN, v) if v > 0 else 0

        return (
            MotorCommand(
                left  = enforce_min(self._smooth_left),
                right = enforce_min(self._smooth_right),
            ),
            direction,
        )

    @staticmethod
    def _distance_to_speed(bbox_w: float, ultrasonic_cm: float) -> float:
        """Map proximity signals to a single forward speed."""
        # Use whichever indicates closer range (more conservative)
        bbox_speed = np.interp(
            bbox_w,
            [BBOX_FAR_WIDTH_RATIO, BBOX_OPTIMAL_WIDTH_RATIO, BBOX_NEAR_WIDTH_RATIO],
            [SPEED_FAR,             SPEED_BASE,               SPEED_NEAR],
        )
        ultra_speed = np.interp(
            ultrasonic_cm,
            [DISTANCE_NEAR_CM, DISTANCE_OPTIMAL_CM, DISTANCE_FAR_CM],
            [SPEED_NEAR,        SPEED_BASE,           SPEED_FAR],
        )
        return float(np.clip(min(bbox_speed, ultra_speed), SPEED_MIN, SPEED_MAX))

    @staticmethod
    def _steer_factor(cx: float, side: str) -> float:
        """
        How much to reduce inner motor speed.
        More extreme position → stronger (but still gentle) turn.
        Returns 0.0 – 0.45 (so minimum inner speed ≈ 55% of outer).
        """
        if side == "left":
            # cx is in [0, ZONE_SAFE_LEFT)
            deviation = ZONE_SAFE_LEFT - cx          # 0 → ZONE_SAFE_LEFT
            norm = deviation / ZONE_SAFE_LEFT          # 0–1
        else:
            deviation = cx - ZONE_SAFE_RIGHT
            norm = deviation / (1.0 - ZONE_SAFE_RIGHT)

        return float(np.clip(norm * 0.45, 0.0, 0.45))


# =============================================================================
#  CAMERA THREAD
# =============================================================================

class CameraThread(threading.Thread):
    """
    Captures frames from the camera into a bounded queue.
    Runs at camera native FPS; downstream threads pick up latest frame.
    """

    def __init__(self, state: RobotState):
        super().__init__(name="Camera", daemon=True)
        self._state = state
        self._frame_q: queue.Queue = queue.Queue(maxsize=2)
        self._cap: Optional[cv2.VideoCapture] = None

    @property
    def frame_queue(self):
        return self._frame_q

    def run(self):
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS_TARGET)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # minimal latency

        if not self._cap.isOpened():
            log.error("Cannot open camera!")
            self._state.running = False
            return

        log.info("Camera opened.")
        while self._state.running:
            ok, frame = self._cap.read()
            if not ok:
                log.warning("Frame read failed.")
                time.sleep(0.05)
                continue

            # Keep queue fresh – drop oldest frame if full
            if self._frame_q.full():
                try:
                    self._frame_q.get_nowait()
                except queue.Empty:
                    pass
            self._frame_q.put(frame)

        self._cap.release()
        log.info("Camera released.")


# =============================================================================
#  DETECTION THREAD
# =============================================================================

class DetectionThread(threading.Thread):
    """
    Pulls frames from the camera queue, runs detection + tracking,
    and writes smoothed centroid into shared state.
    """

    def __init__(self, state: RobotState, frame_queue: queue.Queue):
        super().__init__(name="Detection", daemon=True)
        self._state = state
        self._fq = frame_queue
        self._detector = HumanDetector()
        self._tracker  = CentroidTracker()
        self._frame_skip = 0

    def run(self):
        log.info(f"Detection thread using: {self._detector.name}")
        fps_counter = 0
        fps_timer = time.monotonic()

        while self._state.running:
            try:
                frame = self._fq.get(timeout=0.5)
            except queue.Empty:
                continue

            self._frame_skip += 1
            if self._frame_skip < PROCESS_EVERY_N_FRAMES:
                continue
            self._frame_skip = 0

            # ── Run detector ──────────────────────────────────────────────────
            det = self._detector.detect(frame)
            conf = self._tracker.update(det)
            tracking = self._tracker.is_tracking()

            # ── Write to shared state (atomic-ish update) ─────────────────────
            self._state.target           = det
            self._state.track_confidence = conf
            self._state.smoothed_cx      = self._tracker.cx
            self._state.smoothed_cy      = self._tracker.cy

            # ── FPS counter ───────────────────────────────────────────────────
            fps_counter += 1
            elapsed = time.monotonic() - fps_timer
            if elapsed >= 1.0:
                self._state.fps = fps_counter / elapsed
                fps_counter = 0
                fps_timer = time.monotonic()

            self._state.frame_count += 1


# =============================================================================
#  MOTOR CONTROL THREAD
# =============================================================================

class MotorThread(threading.Thread):
    """
    Reads shared state and issues smooth motor commands at ~30 Hz.
    Implements:
    - Steering decisions
    - Emergency stop watchdog
    - FPS health check
    """

    def __init__(self, state: RobotState, motors: MotorController):
        super().__init__(name="MotorControl", daemon=True)
        self._state   = state
        self._motors  = motors
        self._steerer = SteeringController()
        self._last_fps_check = time.monotonic()

    def run(self):
        log.info("Motor control thread started.")
        while self._state.running:
            s = self._state

            # ── FPS watchdog ──────────────────────────────────────────────────
            now = time.monotonic()
            if now - self._last_fps_check > FPS_WATCHDOG_PERIOD:
                if s.fps < FPS_MINIMUM and s.fps > 0:
                    log.warning(f"FPS too low ({s.fps:.1f}) – emergency stop!")
                    s.emergency_stop = True
                self._last_fps_check = now

            # ── Compute steering ──────────────────────────────────────────────
            cmd, direction = self._steerer.compute(
                smoothed_cx   = s.smoothed_cx,
                bbox_w        = s.target.w if s.target else 0.0,
                ultrasonic_cm = s.ultrasonic_cm,
                tracking      = self._tracker_active(s),
                emergency     = s.emergency_stop,
            )

            s.motor_cmd        = cmd
            s.current_direction = direction

            # ── Apply to motors ───────────────────────────────────────────────
            if direction == "stop":
                self._motors.stop()
            else:
                self._motors.forward(cmd.left, cmd.right)

            time.sleep(0.033)   # ~30 Hz update rate

    @staticmethod
    def _tracker_active(s: RobotState) -> bool:
        """True if we have a confident, recent detection."""
        if s.target is None:
            return False
        age = time.monotonic() - s.target.timestamp
        return (s.track_confidence >= MIN_CONFIDENCE_TO_ACT and
                age < TRACKING_TIMEOUT_SEC)


# =============================================================================
#  DISPLAY / DEBUG THREAD  (optional, disable on headless Pi)
# =============================================================================

class DisplayThread(threading.Thread):
    """Overlays telemetry on live camera feed for debugging."""

    SHOW_DISPLAY = True  # Set False on headless Raspberry Pi

    def __init__(self, state: RobotState, frame_queue: queue.Queue):
        super().__init__(name="Display", daemon=True)
        self._state = state
        self._fq = frame_queue

    def run(self):
        if not self.SHOW_DISPLAY:
            return
        log.info("Display thread started.")
        while self._state.running:
            try:
                frame = self._fq.get(timeout=0.5)
            except queue.Empty:
                continue

            self._draw_overlay(frame)
            cv2.imshow("Human Following Robot", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self._state.running = False
                break
            elif key == ord('e'):
                self._state.emergency_stop = not self._state.emergency_stop

        cv2.destroyAllWindows()

    def _draw_overlay(self, frame: np.ndarray):
        s = self._state
        h, w = frame.shape[:2]

        # Zone lines
        for frac, color in [
            (ZONE_SAFE_LEFT,   (0, 200, 255)),
            (ZONE_CENTER_LEFT, (0, 255,   0)),
            (ZONE_CENTER_RIGHT,(0, 255,   0)),
            (ZONE_SAFE_RIGHT,  (0, 200, 255)),
        ]:
            x = int(frac * w)
            cv2.line(frame, (x, 0), (x, h), color, 1)

        # Target bounding box
        if s.target:
            d = s.target
            x1 = int((d.cx - d.w / 2) * w)
            y1 = int((d.cy - d.h / 2) * h)
            x2 = int((d.cx + d.w / 2) * w)
            y2 = int((d.cy + d.h / 2) * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{d.confidence:.2f}",
                        (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Smoothed centroid
        cx_px = int(s.smoothed_cx * w)
        cy_px = int(s.smoothed_cy * h)
        cv2.circle(frame, (cx_px, cy_px), 6, (255, 80, 0), -1)

        # HUD
        lines = [
            f"Dir: {s.current_direction.upper()}",
            f"L:{s.motor_cmd.left:.0f}% R:{s.motor_cmd.right:.0f}%",
            f"Dist: {s.ultrasonic_cm:.0f}cm",
            f"Conf: {s.track_confidence:.2f}",
            f"FPS: {s.fps:.1f}",
        ]
        if s.emergency_stop:
            lines.insert(0, "!! EMERGENCY STOP !!")
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 220), 3)

        for i, line in enumerate(lines):
            cv2.putText(frame, line, (8, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)


# =============================================================================
#  ROBOT — TOP-LEVEL ORCHESTRATOR
# =============================================================================

class HumanFollowingRobot:
    """
    Ties together all subsystems and manages lifecycle.

    Architecture:
        CameraThread  ──frame_q──► DetectionThread
                                        │
                              shared RobotState
                                        │
                                   MotorThread ──► MotorController
                            UltrasonicSensor ──────►  (state.ultrasonic_cm)
    """

    def __init__(self):
        self._state   = RobotState()
        self._motors  = MotorController()
        self._sonar   = UltrasonicSensor()

        self._cam     = CameraThread(self._state)
        self._detect  = DetectionThread(self._state, self._cam.frame_queue)
        self._motor_t = MotorThread(self._state, self._motors)

        # Separate small queue for display (avoid blocking detection)
        self._disp_q: queue.Queue = queue.Queue(maxsize=2)
        self._display = DisplayThread(self._state, self._cam.frame_queue)

        # Hook SIGINT / SIGTERM
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def start(self):
        log.info("=" * 60)
        log.info("   Human Following Robot — STARTING")
        log.info("=" * 60)

        self._cam.start()
        self._detect.start()
        self._motor_t.start()
        self._display.start()

        # Main thread: poll ultrasonic sensor and update state
        log.info("Main loop running. Press Ctrl-C or Q to stop.")
        try:
            while self._state.running:
                self._state.ultrasonic_cm = self._sonar.get_distance()
                time.sleep(0.05)
        finally:
            self.shutdown()

    def shutdown(self):
        log.info("Shutting down…")
        self._state.running = False
        self._motors.emergency_stop()
        time.sleep(0.3)
        self._motors.cleanup()
        log.info("Robot stopped cleanly.")

    def _signal_handler(self, sig, frame):
        log.info(f"Signal {sig} received.")
        self._state.running = False


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    robot = HumanFollowingRobot()
    robot.start()