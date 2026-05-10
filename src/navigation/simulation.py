"""
=============================================================================
SIMULATION TEST HARNESS
=============================================================================
Run this on any PC (no Pi, no GPIO, no camera required) to visualise and
validate the tracking + steering logic with a synthetic moving target.

Usage:
    python3 simulate.py

Controls:
    Mouse click  → set human centroid position
    W / A / S / D → nudge simulated human
    +/-          → resize simulated bounding box (distance simulation)
    E            → toggle emergency stop
    Q / Esc      → quit
=============================================================================
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import time
import math

# ── Stub out GPIO before importing robot_main ─────────────────────────────────
import sys
from unittest.mock import MagicMock
sys.modules['RPi']       = MagicMock()
sys.modules['RPi.GPIO']  = MagicMock()

# Import core logic (GPIO_AVAILABLE will be False)
from robot_main import (
    RobotState, Detection, CentroidTracker,
    SteeringController, CAMERA_WIDTH, CAMERA_HEIGHT,
    ZONE_SAFE_LEFT, ZONE_SAFE_RIGHT,
    ZONE_CENTER_LEFT, ZONE_CENTER_RIGHT,
)

# ─────────────────────────────────────────────────────────────────────────────

W, H = CAMERA_WIDTH, CAMERA_HEIGHT

state    = RobotState()
tracker  = CentroidTracker()
steerer  = SteeringController()

# Simulated human
sim_cx   = 0.5
sim_cy   = 0.5
sim_bw   = 0.25   # bbox width ratio
sim_dist = 120.0  # fake ultrasonic cm
tracking_override = True

frame_index = 0

def mouse_cb(event, x, y, flags, param):
    global sim_cx, sim_cy
    if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON):
        sim_cx = x / W
        sim_cy = y / H

cv2.namedWindow("HFR Simulator")
cv2.setMouseCallback("HFR Simulator", mouse_cb)

print("Simulation running. Click to move human. Q to quit.")

while True:
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:] = (25, 25, 35)   # dark background

    # Oscillate human slightly to test stability
    t = time.monotonic()
    jitter_x = math.sin(t * 0.4) * 0.02  # very slow sway

    # Build a fake detection
    det = Detection(
        cx=sim_cx + jitter_x,
        cy=sim_cy,
        w=sim_bw,
        h=sim_bw * 2.2,
        confidence=0.88,
        timestamp=time.monotonic(),
    )

    conf = tracker.update(det)
    tracking = tracker.is_tracking()

    state.smoothed_cx      = tracker.cx
    state.smoothed_cy      = tracker.cy
    state.track_confidence = conf
    state.target           = det
    state.ultrasonic_cm    = sim_dist

    cmd, direction = steerer.compute(
        smoothed_cx   = tracker.cx,
        bbox_w        = det.w,
        ultrasonic_cm = sim_dist,
        tracking      = tracking,
        emergency     = state.emergency_stop,
    )
    state.motor_cmd        = cmd
    state.current_direction = direction

    # ── Draw ──────────────────────────────────────────────────────────────────

    # Zone bands
    for frac, col in [
        (ZONE_SAFE_LEFT,    (20, 80, 150)),
        (ZONE_CENTER_LEFT,  (20, 120, 60)),
        (ZONE_CENTER_RIGHT, (20, 120, 60)),
        (ZONE_SAFE_RIGHT,   (20, 80, 150)),
    ]:
        x = int(frac * W)
        cv2.line(frame, (x, 0), (x, H), col, 1)

    # Zone labels
    for label, xf in [
        ("EXT", 0.08), ("SAFE", 0.27), ("CENTER", 0.47),
        ("SAFE", 0.73), ("EXT", 0.9)
    ]:
        cv2.putText(frame, label, (int(xf * W) - 15, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    # Human bounding box
    bx1 = int((det.cx - det.w / 2) * W)
    bx2 = int((det.cx + det.w / 2) * W)
    by1 = int((det.cy - det.h / 2) * H)
    by2 = int((det.cy + det.h / 2) * H)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 230, 60), 2)

    # Smoothed centroid dot
    scx = int(tracker.cx * W)
    scy = int(tracker.cy * H)
    cv2.circle(frame, (scx, scy), 8, (255, 120, 0), -1)
    cv2.circle(frame, (scx, scy), 8, (255, 255, 255), 1)

    # Motor bars
    bar_y = H - 60
    bar_h = 40
    bar_w = 80
    # Left motor
    cv2.rectangle(frame, (20, bar_y), (20 + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    fill = int(cmd.left / 100 * bar_w)
    cv2.rectangle(frame, (20, bar_y), (20 + fill, bar_y + bar_h), (0, 200, 255), -1)
    cv2.putText(frame, f"L {cmd.left:.0f}%", (22, bar_y + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Right motor
    rx = W - 20 - bar_w
    cv2.rectangle(frame, (rx, bar_y), (rx + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    fill = int(cmd.right / 100 * bar_w)
    cv2.rectangle(frame, (rx, bar_y), (rx + fill, bar_y + bar_h), (0, 200, 255), -1)
    cv2.putText(frame, f"R {cmd.right:.0f}%", (rx + 2, bar_y + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Direction arrow
    arrow_map = {
        "forward": (W//2 - 5, H//2 + 20, W//2 - 5, H//2 - 40),
        "left":    (W//2 + 30, H//2, W//2 - 30, H//2),
        "right":   (W//2 - 30, H//2, W//2 + 30, H//2),
        "stop":    None,
    }
    arrow = arrow_map.get(direction)
    if arrow:
        cv2.arrowedLine(frame, (arrow[0], arrow[1]), (arrow[2], arrow[3]),
                        (255, 80, 0), 3, tipLength=0.35)
    else:
        cv2.putText(frame, "STOP", (W//2 - 25, H//2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2)

    # HUD text
    hud = [
        f"Dir: {direction.upper()}",
        f"Conf: {conf:.2f}  cx: {tracker.cx:.2f}",
        f"Dist: {sim_dist:.0f}cm  BBox: {det.w:.2f}",
        f"E-stop: {state.emergency_stop}",
    ]
    for i, line in enumerate(hud):
        cv2.putText(frame, line, (8, 32 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    if state.emergency_stop:
        cv2.rectangle(frame, (0,0), (W,H), (0, 0, 180), 4)
        cv2.putText(frame, "!! EMERGENCY STOP !!", (W//2 - 110, H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("HFR Simulator", frame)

    key = cv2.waitKey(33) & 0xFF
    if key in (ord('q'), 27):
        break
    elif key == ord('w'): sim_cy  = max(0.1, sim_cy  - 0.05)
    elif key == ord('s'): sim_cy  = min(0.9, sim_cy  + 0.05)
    elif key == ord('a'): sim_cx  = max(0.05, sim_cx - 0.05)
    elif key == ord('d'): sim_cx  = min(0.95, sim_cx + 0.05)
    elif key == ord('+'): sim_bw  = min(0.8, sim_bw + 0.03); sim_dist = max(40, sim_dist - 10)
    elif key == ord('-'): sim_bw  = max(0.1, sim_bw - 0.03); sim_dist = min(300, sim_dist + 10)
    elif key == ord('e'): state.emergency_stop = not state.emergency_stop

cv2.destroyAllWindows()
print("Simulation ended.")