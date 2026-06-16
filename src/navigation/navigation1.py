
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
    raise SystemExit("Run:  pip install ultralytics")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ChildBot")


class CFG:
    IN1 = 17;  IN2 = 18;  ENA = 12
    IN3 = 23;  IN4 = 24;  ENB = 19
    TRIG = 5;  ECHO = 6

    CAM_INDEX = 0
    CAM_W = 640;  CAM_H = 480;  CAM_FPS = 30

    MODEL      = "yolov8n.pt"
    INFER_SIZE = 160            
    CONF       = 0.45
    IOU        = 0.45
    SKIP       = 2              

   
    X_TOL = 0.12                

    
    STOP_CM   = 70.0           
    RAMP_CM   = 120.0           

   
    MIN_FWD   = 35.0            
    MAX_FWD   = 70.0            

    TURN_MAX  = 22.0

    EMA_ALPHA      = 0.18     
    HYSTERESIS_N   = 5          
    MAX_TURN_RATE  = 5.0        
    VEL_FEEDFWD    = 0.08      

   
    FWD_EMA_ALPHA  = 0.35      

    
    STABLE_FRAMES  = 6

    
    LOST_LIMIT = 25

   
    PWM_HZ  = 1000

    
    ANG_KP = 0.65;  ANG_KI = 0.003;  ANG_KD = 0.08

    
    SHOW = True

class ThreadedCamera:
   

    def __init__(self):
        self._cap = cv2.VideoCapture(CFG.CAM_INDEX)
        
       
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG.CAM_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.CAM_H)
        self._cap.set(cv2.CAP_PROP_FPS,          CFG.CAM_FPS)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        
       
        self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  
        
        if not self._cap.isOpened():
            raise RuntimeError("Cannot open camera!")

        self._frame:  Optional[np.ndarray] = None
        self._cam_ms: float = 0.0
        self._lock  = threading.Lock()
        self._stop  = False
        
       
        self._timeout_count = 0
        self._frame_count = 0
        self._last_log_t = time.monotonic()

        threading.Thread(target=self._loop, daemon=True).start()
        time.sleep(0.30)
        log.info("Camera thread started.")

    def _loop(self):
        """Robust camera read loop with error recovery."""
        consecutive_fails = 0
        max_consecutive_fails = 10
        
        while not self._stop:
            try:
                t0 = time.monotonic()
                ret, frame = self._cap.read()
                
                if ret and frame is not None:
                    ms = (time.monotonic() - t0) * 1000
                    with self._lock:
                        self._frame = frame
                        self._cam_ms = ms
                        self._frame_count += 1
                    
                    consecutive_fails = 0  
                else:
                    consecutive_fails += 1
                    self._timeout_count += 1
                    
                    
                    if self._timeout_count % 30 == 0:
                        log.warning(
                            f"Camera timeout #{self._timeout_count} "
                            f"(consecutive: {consecutive_fails}/{max_consecutive_fails})"
                        )
                    
                
                    if consecutive_fails >= max_consecutive_fails:
                        log.error("Camera read failures exceeded limit — attempting recovery...")
                        self._recover_camera()
                        consecutive_fails = 0
                    
                   
                    time.sleep(0.001)
            
            except Exception as e:
                consecutive_fails += 1
                log.error(f"Camera exception: {e}")
                if consecutive_fails >= max_consecutive_fails:
                    log.error("Attempting camera recovery...")
                    self._recover_camera()
                    consecutive_fails = 0
                time.sleep(0.005)

    def _recover_camera(self):
        """Attempt to recover from camera timeout/error."""
        try:
            self._cap.release()
            time.sleep(0.1)
            self._cap = cv2.VideoCapture(CFG.CAM_INDEX)
            
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG.CAM_W)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.CAM_H)
            self._cap.set(cv2.CAP_PROP_FPS,          CFG.CAM_FPS)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
            self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            
            if self._cap.isOpened():
                log.info("Camera recovery successful.")
            else:
                log.error("Camera recovery failed — device may be disconnected.")
        except Exception as e:
            log.error(f"Camera recovery error: {e}")

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Get latest frame (never blocks)."""
        with self._lock:
            if self._frame is None:
                return False, None, 0.0
            return True, self._frame.copy(), self._cam_ms

    def release(self):
        """Cleanly shutdown camera thread."""
        self._stop = True
        time.sleep(0.1)
        if self._cap.isOpened():
            self._cap.release()
        
        elapsed = time.monotonic() - self._last_log_t
        if self._frame_count > 0:
            fps = self._frame_count / elapsed if elapsed > 0 else 0
            log.info(f"Camera stats: {self._frame_count} frames @ {fps:.1f} fps, "
                    f"{self._timeout_count} timeouts")



class KalmanCentroidTracker:
    

    
    _DT = 1.0 / 10.0

    def __init__(self):
        
        self.kf = cv2.KalmanFilter(4, 2)

        dt = self._DT
        
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]], dtype=np.float32)

        
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]], dtype=np.float32)

       
        Q = np.eye(4, dtype=np.float32)
        Q[0, 0] = Q[1, 1] = 1e-3  
        Q[2, 2] = Q[3, 3] = 8e-3   
        self.kf.processNoiseCov = Q

       
        R = np.eye(2, dtype=np.float32) * 5e-3
        self.kf.measurementNoiseCov = R

       
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.1

        self.initialized = False
        self.lost_frames = 0
        self.last_raw_bbox: Optional[Tuple] = None

    def update(self, bbox_norm: Optional[Tuple]) -> bool:
       
        if bbox_norm is not None:
            cx = (bbox_norm[0] + bbox_norm[2]) / 2.0
            cy = (bbox_norm[1] + bbox_norm[3]) / 2.0
            self.last_raw_bbox = bbox_norm
            self.lost_frames   = 0

            if not self.initialized:
                self.kf.statePre  = np.array(
                    [cx, cy, 0.0, 0.0], dtype=np.float32).reshape(4, 1)
                self.kf.statePost = self.kf.statePre.copy()
                self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.1
                self.initialized = True
            else:
                self.kf.predict()
                meas = np.array([cx, cy], dtype=np.float32).reshape(2, 1)
                self.kf.correct(meas)

        else:
            self.lost_frames += 1
            if not self.initialized:
                return False
            if self.lost_frames > CFG.LOST_LIMIT:
                self.initialized   = False
                self.last_raw_bbox = None
                return False
            self.kf.predict()

        return self.initialized

    def reset(self):
        self.initialized   = False
        self.lost_frames   = 0
        self.last_raw_bbox = None

    @property
    def cx(self) -> Optional[float]:
        
        if not self.initialized:
            return None
        return float(self.kf.statePost[0, 0])

    @property
    def cy(self) -> Optional[float]:
        if not self.initialized:
            return None
        return float(self.kf.statePost[1, 0])

    @property
    def vx(self) -> float:
        
        if not self.initialized:
            return 0.0
        return float(self.kf.statePost[2, 0])

    @property
    def vy(self) -> float:
        if not self.initialized:
            return 0.0
        return float(self.kf.statePost[3, 0])

    @property
    def is_active(self) -> bool:
        return self.initialized



class SmoothSteering:
  

    def __init__(self, pid: 'PID'):
        self._pid         = pid
        self._ema         = 0.0
        self._hyst_count  = 0
        self._turn_pct    = 0.0

    def compute(self, x_dev_raw: float, kalman_vx: float) -> Tuple[float, str]:
       

        α = CFG.EMA_ALPHA
        self._ema = α * x_dev_raw + (1.0 - α) * self._ema

        x_effective = self._ema + CFG.VEL_FEEDFWD * kalman_vx

        if x_effective > CFG.X_TOL:
            self._hyst_count = min(self._hyst_count + 1,
                                   CFG.HYSTERESIS_N * 3)
        elif x_effective < -CFG.X_TOL:
            self._hyst_count = max(self._hyst_count - 1,
                                   -CFG.HYSTERESIS_N * 3)
        else:
            self._hyst_count = int(self._hyst_count * 0.6)

        gate_open = abs(self._hyst_count) >= CFG.HYSTERESIS_N

        if not gate_open:
            self._pid.reset()
            target_turn = 0.0
            label = ""
        else:
            pid_out    = self._pid(x_effective)
            target_turn = pid_out * CFG.TURN_MAX
            label = "TurnR" if x_effective > 0 else "TurnL"

        delta = target_turn - self._turn_pct
        delta = float(np.clip(delta, -CFG.MAX_TURN_RATE, CFG.MAX_TURN_RATE))
        self._turn_pct += delta

        return self._turn_pct, label

    def reset(self):
        
        self._ema        = 0.0
        self._hyst_count = 0
        self._turn_pct   = 0.0
        self._pid.reset()



class PID:
    def __init__(self, kp, ki, kd, lo=-1.0, hi=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.lo, self.hi = lo, hi
        self._i  = 0.0
        self._pe = 0.0
        self._pt = None

    def reset(self):
        self._i = 0.0;  self._pe = 0.0;  self._pt = None

    def __call__(self, error: float) -> float:
        now = time.monotonic()
        dt  = (now - self._pt) if self._pt else 0.033
        dt  = max(dt, 1e-4)
        self._pt = now
        self._i += error * dt
        if abs(self.ki * self._i) > 0.4:
            self._i *= 0.4 / abs(self.ki * self._i)
        d = (error - self._pe) / dt
        self._pe = error
        return float(np.clip(
            self.kp * error + self.ki * self._i + self.kd * d,
            self.lo, self.hi))


class Motors:
  

    MIN_DUTY = 3.0

    def __init__(self):
        self._lp = self._rp = None
        self._L_prev = 0.0
        self._R_prev = 0.0
        self._ramp_limit = 8.0
        
        if ON_PI:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for pin in [CFG.IN1, CFG.IN2, CFG.ENA,
                        CFG.IN3, CFG.IN4, CFG.ENB]:
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            self._lp = GPIO.PWM(CFG.ENA, CFG.PWM_HZ)
            self._rp = GPIO.PWM(CFG.ENB, CFG.PWM_HZ)
            self._lp.start(0);  self._rp.start(0)
            log.info("Motors GPIO ready (BCM 17/18/22 + 23/24/25).")

    def _L(self, spd: float):
        
        spd = float(np.clip(spd, -100, 100))
        
        delta = spd - self._L_prev
        delta = float(np.clip(delta, -self._ramp_limit, self._ramp_limit))
        spd = self._L_prev + delta
        self._L_prev = spd
        
        if ON_PI:
            GPIO.output(CFG.IN1, GPIO.HIGH if spd >= 0 else GPIO.LOW)
            GPIO.output(CFG.IN2, GPIO.LOW  if spd >= 0 else GPIO.HIGH)
            self._lp.ChangeDutyCycle(abs(spd))

    def _R(self, spd: float):
        """Set right motor speed with ramp limiting."""
        spd = float(np.clip(spd, -100, 100))
        
        delta = spd - self._R_prev
        delta = float(np.clip(delta, -self._ramp_limit, self._ramp_limit))
        spd = self._R_prev + delta
        self._R_prev = spd
        
        if ON_PI:
            GPIO.output(CFG.IN3, GPIO.HIGH if spd >= 0 else GPIO.LOW)
            GPIO.output(CFG.IN4, GPIO.LOW  if spd >= 0 else GPIO.HIGH)
            self._rp.ChangeDutyCycle(abs(spd))

    def drive_fwd(self, fwd_pct: float, turn_pct: float):
  
        fwd_pct  = float(np.clip(fwd_pct,  0,    100))
        turn_pct = float(np.clip(turn_pct, -100, 100))

        base_fwd = fwd_pct
        
        
        turn_ratio = abs(turn_pct) / 100.0
        
        if turn_pct > 0:
            L = base_fwd + (turn_pct * 0.2)
            R = base_fwd - (turn_pct * 1.2)
            
        elif turn_pct < 0:
            L = base_fwd + (turn_pct * 1.2)
            R = base_fwd - (turn_pct * 0.2)
            
        else:
            L = base_fwd
            R = base_fwd

        max_wheel = max(abs(L), abs(R))
        if max_wheel > 100.0:
            scale = 100.0 / max_wheel
            L *= scale
            R *= scale
        
        def apply_deadband(spd: float) -> float:
            if abs(spd) < self.MIN_DUTY:
                return 0.0
            elif spd > 0:
                return max(spd, self.MIN_DUTY)
            else:
                return min(spd, -self.MIN_DUTY)
        
        L = apply_deadband(L)
        R = apply_deadband(R)

        self._L(L);  self._R(R)

        if not ON_PI:
            log.debug(f"[SIM] fwd={fwd_pct:.0f}% turn={turn_pct:+.0f}% "
                      f"→ L={L:+.0f}%  R={R:+.0f}%")

    def stop(self):
        """Hard stop motors."""
        self._L(0);  self._R(0)
        self._L_prev = 0.0
        self._R_prev = 0.0

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
            time.sleep(0.08)

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
    C_PURPLE = (200,  50, 200)

    STATUS_COL = {
        "WAITING":   (160, 160, 160),
        "TRACKING":  ( 20, 190, 255),
        "ACQUIRED":  ( 40, 200,  40),
        "OBSTACLE":  (  0,   0, 220),
    }

    BAR  = 38
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def render(self,
               frame:        np.ndarray,
               raw_bbox:     Optional[Tuple],
               kalman_cx:    Optional[float],
               kalman_cy:    Optional[float],
               x_dev_raw:    float,
               x_dev_smooth: float,
               stable_pct:   float,
               dist_cm:      float,
               direction:    str,
               fwd_pct:      float,
               turn_pct:     float,
               fps:          float,
               cam_ms:       float,
               inf_ms:       float,
               oth_ms:       float,
               status:       str) -> np.ndarray:

        H, W = frame.shape[:2]
        fcx, fcy = W // 2, H // 2
        tx = int(CFG.X_TOL * W)
        ty = int(CFG.X_TOL * H)

        out = frame.copy()

        cv2.line(out, (fcx, 0),  (fcx, H), self.C_BLUE, 1, cv2.LINE_AA)
        cv2.line(out, (0,  fcy), (W,  fcy), self.C_BLUE, 1, cv2.LINE_AA)

        cv2.rectangle(out, (fcx-tx, fcy-ty), (fcx+tx, fcy+ty), self.C_GREEN, 2)

        if raw_bbox is not None:
            x1 = int(raw_bbox[0]*W);  y1 = int(raw_bbox[1]*H)
            x2 = int(raw_bbox[2]*W);  y2 = int(raw_bbox[3]*H)
            cv2.rectangle(out, (x1, y1), (x2, y2), self.C_GREY, 1)

        if kalman_cx is not None and kalman_cy is not None:
            kcx = int(kalman_cx * W)
            kcy = int(kalman_cy * H)
            cv2.circle(out, (kcx, kcy), 8, self.C_YELLOW, -1)
            cv2.circle(out, (kcx, kcy), 8, self.C_WHITE,   2)
            cv2.line(out, (kcx, kcy), (fcx, fcy), self.C_YELLOW, 1, cv2.LINE_AA)

        bar_max_h = H - 2*self.BAR
        if dist_cm < 999:
            ratio = max(0.0, min(1.0,
                (dist_cm - CFG.STOP_CM) / max(1, CFG.RAMP_CM - CFG.STOP_CM)))
            bh    = int(ratio * bar_max_h)
            bar_x = W - 18
            cv2.rectangle(out, (bar_x, self.BAR),
                          (bar_x+14, self.BAR+bar_max_h), (40,40,40), -1)
            b_col = self.C_GREEN if ratio > 0.4 else self.C_ORANGE
            cv2.rectangle(out,
                          (bar_x, self.BAR+bar_max_h-bh),
                          (bar_x+14, self.BAR+bar_max_h), b_col, -1)
            cv2.putText(out, f"{dist_cm:.0f}", (bar_x-2, self.BAR+bar_max_h+14),
                        self.FONT, 0.38, self.C_WHITE, 1)

        if status == "WAITING" and stable_pct > 0:
            sb_h   = int(stable_pct * bar_max_h)
            sb_x   = 4
            cv2.rectangle(out, (sb_x, self.BAR),
                          (sb_x+10, self.BAR+bar_max_h), (30,30,30), -1)
            cv2.rectangle(out,
                          (sb_x, self.BAR+bar_max_h-sb_h),
                          (sb_x+10, self.BAR+bar_max_h), self.C_PURPLE, -1)

        cv2.rectangle(out, (0,0), (W, self.BAR), self.C_BLACK, -1)
        ty_t = self.BAR - 10
        cv2.putText(out, f"FPS:{fps:.1f}", (6, ty_t),
                    self.FONT, 0.68, self.C_WHITE, 2)
        cv2.putText(out,
                    f"Cam:{cam_ms:.0f}ms  Inf:{inf_ms:.0f}ms  Oth:{oth_ms:.0f}ms",
                    (115, ty_t), self.FONT, 0.48, self.C_GREY, 1)
        s_col = self.STATUS_COL.get(status, self.C_WHITE)
        (sw, _), _ = cv2.getTextSize(status, self.FONT, 0.68, 2)
        cv2.putText(out, status, (W-sw-22, ty_t), self.FONT, 0.68, s_col, 2)

        cv2.rectangle(out, (0, H-self.BAR), (W, H), self.C_BLACK, -1)
        by_t = H - 10

        xc = self.C_ORANGE if abs(x_dev_smooth) > CFG.X_TOL else self.C_GREEN
        cv2.putText(out, f"X:{x_dev_smooth:+.3f}({x_dev_raw:+.2f})",
                    (6, by_t), self.FONT, 0.52, xc, 1)

        sc = self.C_RED if dist_cm <= CFG.STOP_CM else (
             self.C_ORANGE if dist_cm < CFG.RAMP_CM else self.C_GREEN)
        cv2.putText(out, f"Sonar:{dist_cm:.0f}cm", (200, by_t),
                    self.FONT, 0.60, sc, 2)

        dir_col = self.C_PURPLE if status == "WAITING" else (
                  self.C_RED    if "OBSTACLE" in direction else (
                  self.C_GREEN  if direction == "Forward" else self.C_YELLOW))
        (dw, _), _ = cv2.getTextSize(direction, self.FONT, 0.72, 2)
        cv2.putText(out, direction, (W//2-dw//2, by_t),
                    self.FONT, 0.72, dir_col, 2)

        spd_str = f"Fwd:{fwd_pct:.0f}%  T:{turn_pct:+.0f}%"
        (ssw, _), _ = cv2.getTextSize(spd_str, self.FONT, 0.55, 1)
        cv2.putText(out, spd_str, (W-ssw-22, by_t),
                    self.FONT, 0.55, self.C_WHITE, 1)

        if status == "WAITING":
            if stable_pct > 0:
                msg = f"Acquiring... {int(stable_pct*100)}%"
            else:
                msg = "Waiting for person..."
            (mw, mh), _ = cv2.getTextSize(msg, self.FONT, 0.75, 2)
            mx = (W - mw) // 2
            my = H // 2 - 20
            cv2.rectangle(out, (mx-8, my-mh-6), (mx+mw+8, my+6),
                          (30, 30, 30), -1)
            cv2.putText(out, msg, (mx, my), self.FONT, 0.75, self.C_PURPLE, 2)

        return out


class Navigator:
   

    def __init__(self):
        log.info(f"Loading {CFG.MODEL} …")
        self.yolo = YOLO(CFG.MODEL)
        self.yolo(np.zeros((CFG.INFER_SIZE, CFG.INFER_SIZE, 3), np.uint8),
                  verbose=False)
        log.info("YOLO ready.")

        self.motors  = Motors()
        self.sonar   = Sonar()
        self.cam     = ThreadedCamera()
        self.tracker = KalmanCentroidTracker()
        self.hud     = HUD()

        pid = PID(CFG.ANG_KP, CFG.ANG_KI, CFG.ANG_KD, lo=-1.0, hi=1.0)
        self.steering = SmoothSteering(pid)

        self._skip_ctr = 0
        self._inf_ms   = 0.0
        self._fps_buf: deque = deque(maxlen=15)
        self._last_t   = time.monotonic()

        self._stable_count  = 0
        self._is_stable     = False

        self._fwd_smooth = 0.0

    def _detect(self, frame: np.ndarray) -> Optional[Tuple]:
        """Return best bbox (x1,y1,x2,y2) normalised 0–1, or None."""
        H, W = frame.shape[:2]
        t0   = time.monotonic()
        res  = self.yolo(frame, classes=[0], conf=CFG.CONF, iou=CFG.IOU,
                         imgsz=CFG.INFER_SIZE, verbose=False)
        self._inf_ms = (time.monotonic() - t0) * 1000

        boxes = res[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        fcx = W / 2.0;  fcy = H / 2.0
        best = None;  best_d = float('inf')
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if (y2 - y1) / H < 0.08:
                continue
            d = math.hypot((x1+x2)/2 - fcx, (y1+y2)/2 - fcy)
            if d < best_d:
                best_d = d
                best = (x1/W, y1/H, x2/W, y2/H)
        return best

    def _update_stability(self, detected: bool):
        
        if detected:
            self._stable_count = min(self._stable_count + 1,
                                     CFG.STABLE_FRAMES)
        else:
            self._stable_count = max(self._stable_count - 1, 0)

        self._is_stable = (self._stable_count >= CFG.STABLE_FRAMES)

    def _fwd_from_sonar(self, dist_cm: float) -> float:
       
        if dist_cm <= CFG.STOP_CM:
            target = 0.0
        elif dist_cm >= CFG.RAMP_CM:
            target = CFG.MAX_FWD
        else:
            ratio  = (dist_cm - CFG.STOP_CM) / (CFG.RAMP_CM - CFG.STOP_CM)
            target = CFG.MIN_FWD + ratio * (CFG.MAX_FWD - CFG.MIN_FWD)

        α = CFG.FWD_EMA_ALPHA
        self._fwd_smooth = α * target + (1.0 - α) * self._fwd_smooth
        return self._fwd_smooth

    def _control(self,
                 kalman_cx:    Optional[float],
                 kalman_vx:    float,
                 has_target:   bool,
                 dist_cm:      float) -> Tuple[str, float, float, str]:
       
        if not self._is_stable:
            self.steering.reset()
            self._fwd_smooth = 0.0
            self.motors.stop()
            return "Waiting...", 0.0, 0.0, "WAITING"

        if not has_target:
            self.steering.reset()
            self._fwd_smooth = 0.0
            self.motors.stop()
            return "Lost target", 0.0, 0.0, "WAITING"

        if dist_cm <= CFG.STOP_CM:
            self.motors.stop()
            self.steering.reset()
            self._fwd_smooth = 0.0
            return "OBSTACLE!", 0.0, 0.0, "OBSTACLE"


        x_dev_raw = (kalman_cx - 0.5) if kalman_cx is not None else 0.0

        turn_pct, turn_lbl = self.steering.compute(x_dev_raw, kalman_vx)

        fwd_pct = self._fwd_from_sonar(dist_cm)

        turn_ratio = abs(turn_pct) / max(CFG.TURN_MAX, 1.0)
        if turn_ratio > 0.5 and fwd_pct > CFG.MIN_FWD:
            reduction = 0.20 * (turn_ratio - 0.5) / 0.5
            fwd_pct = max(CFG.MIN_FWD, fwd_pct * (1.0 - reduction))

        self.motors.drive_fwd(fwd_pct, turn_pct)

        ema_x      = self.steering._ema
        at_centre  = abs(ema_x) <= CFG.X_TOL
        at_dist    = CFG.STOP_CM < dist_cm <= CFG.RAMP_CM
        status     = "ACQUIRED" if (at_centre and at_dist) else "TRACKING"

        if turn_lbl:
            if fwd_pct > CFG.MIN_FWD:
                direction = f"Fwd+{turn_lbl}"
            else:
                direction = f"Turn {'R' if turn_pct > 0 else 'L'}"
        else:
            direction = "Forward" if fwd_pct > 0 else "HOLD"

        return direction, fwd_pct, turn_pct, status

    def run(self):
        self.sonar.start()
        log.info("Navigator v7.0 running with improved turning response.")
        log.info("  Differential drive with motor reversal activated.")
        log.info(f"  Stability gate:  {CFG.STABLE_FRAMES} consecutive detections")
        log.info(f"  Hysteresis:      {CFG.HYSTERESIS_N} frames before turning")
        log.info(f"  Turn rate limit: {CFG.MAX_TURN_RATE}% per frame")
        log.info(f"  EMA alpha:       {CFG.EMA_ALPHA} (x_dev)")
        log.info(f"  Vel feed-fwd:    {CFG.VEL_FEEDFWD}")
        log.info(f"  Hard stop:       sonar ≤ {CFG.STOP_CM:.0f} cm")
        log.info(f"  Full speed:      sonar ≥ {CFG.RAMP_CM:.0f} cm → {CFG.MAX_FWD:.0f}%")
        log.info("  Robot stands STILL until person is stably acquired.")

        cam_ms = inf_ms = oth_ms = 0.0
        fps = 0.0
        fwd_pct = turn_pct = 0.0
        direction = "WAITING"
        status    = "WAITING"
        x_dev_raw = 0.0

        try:
            while True:
                ok, frame, cam_ms = self.cam.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue

                t_oth = time.monotonic()

                self._skip_ctr += 1
                run_yolo = self._skip_ctr > CFG.SKIP
                if run_yolo:
                    self._skip_ctr = 0
                    raw_bbox = self._detect(frame)
                    inf_ms   = self._inf_ms
                else:
                    raw_bbox = None
                    inf_ms   = 0.0

                if run_yolo:
                    active = self.tracker.update(raw_bbox)
                    self._update_stability(raw_bbox is not None)
                else:
                    active = self.tracker.update(None) if self.tracker.is_active else False

                has_target = active and self.tracker.is_active
                kalman_cx  = self.tracker.cx
                kalman_cy  = self.tracker.cy
                kalman_vx  = self.tracker.vx

                x_dev_raw = (kalman_cx - 0.5) if kalman_cx is not None else 0.0

                direction, fwd_pct, turn_pct, status = self._control(
                    kalman_cx, kalman_vx, has_target, self.sonar.cm)

                now = time.monotonic()
                dt  = now - self._last_t
                self._last_t = now
                if dt > 0:
                    self._fps_buf.append(1.0 / dt)
                fps    = float(np.mean(self._fps_buf)) if self._fps_buf else 0.0
                oth_ms = (time.monotonic() - t_oth) * 1000

                if CFG.SHOW:
                    stable_pct = self._stable_count / CFG.STABLE_FRAMES
                    vis = self.hud.render(
                        frame,
                        self.tracker.last_raw_bbox,
                        kalman_cx, kalman_cy,
                        x_dev_raw,
                        self.steering._ema,
                        stable_pct,
                        self.sonar.cm,
                        direction, fwd_pct, turn_pct,
                        fps, cam_ms, inf_ms, oth_ms, status,
                    )
                    cv2.imshow("Child-Bot Navigator v7", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        log.info("Quit requested.")
                        break

                if run_yolo:
                    log.info(
                        f"FPS={fps:.1f}  sonar={self.sonar.cm:.0f}cm  "
                        f"X_raw={x_dev_raw:+.4f}  X_ema={self.steering._ema:+.4f}  "
                        f"vx={kalman_vx:+.4f}  stable={self._stable_count}/{CFG.STABLE_FRAMES}  "
                        f"[{status}]  [{direction}]  "
                        f"fwd={fwd_pct:.0f}%  turn={turn_pct:+.0f}%"
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
    ap = argparse.ArgumentParser(description="Child-Following Robot v7.0")
    ap.add_argument("--no-display",   action="store_true", help="Headless mode")
    ap.add_argument("--stop-dist",    type=float, default=CFG.STOP_CM)
    ap.add_argument("--ramp-dist",    type=float, default=CFG.RAMP_CM)
    ap.add_argument("--min-fwd",      type=float, default=CFG.MIN_FWD)
    ap.add_argument("--max-fwd",      type=float, default=CFG.MAX_FWD)
    ap.add_argument("--turn-max",     type=float, default=CFG.TURN_MAX)
    ap.add_argument("--x-tol",        type=float, default=CFG.X_TOL)
    ap.add_argument("--lost-limit",   type=int,   default=CFG.LOST_LIMIT)
    ap.add_argument("--skip",         type=int,   default=CFG.SKIP)
    ap.add_argument("--model",        type=str,   default=CFG.MODEL)
    ap.add_argument("--stable",       type=int,   default=CFG.STABLE_FRAMES,
                    help=f"Detections before moving (default {CFG.STABLE_FRAMES})")
    ap.add_argument("--hysteresis",   type=int,   default=CFG.HYSTERESIS_N,
                    help=f"Frames before turn commits (default {CFG.HYSTERESIS_N})")
    ap.add_argument("--ema-alpha",    type=float, default=CFG.EMA_ALPHA,
                    help=f"x_dev EMA weight 0-1 (default {CFG.EMA_ALPHA})")
    ap.add_argument("--turn-rate",    type=float, default=CFG.MAX_TURN_RATE,
                    help=f"Max turn change per frame (default {CFG.MAX_TURN_RATE})")
    ap.add_argument("--vel-ff",       type=float, default=CFG.VEL_FEEDFWD,
                    help=f"Kalman vx feed-forward gain (default {CFG.VEL_FEEDFWD})")
    args = ap.parse_args()

    CFG.SHOW           = not args.no_display
    CFG.STOP_CM        = args.stop_dist
    CFG.RAMP_CM        = args.ramp_dist
    CFG.MIN_FWD        = args.min_fwd
    CFG.MAX_FWD        = args.max_fwd
    CFG.TURN_MAX       = args.turn_max
    CFG.X_TOL          = args.x_tol
    CFG.LOST_LIMIT     = args.lost_limit
    CFG.SKIP           = args.skip
    CFG.MODEL          = args.model
    CFG.STABLE_FRAMES  = args.stable
    CFG.HYSTERESIS_N   = args.hysteresis
    CFG.EMA_ALPHA      = args.ema_alpha
    CFG.MAX_TURN_RATE  = args.turn_rate
    CFG.VEL_FEEDFWD    = args.vel_ff

    Navigator().run()