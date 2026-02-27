import cv2
import numpy as np
import time
from ultralytics import YOLO

# ==============================
# CONFIGURATION
# ==============================

MODEL_PATH = r"C:\Users\anush\ANUSHRABA\FYRP\VitaCustos_AI\training\runs\pose\train6\weights\best.pt"
CAMERA_INDEX = 0
CONFIRM_FRAMES = 15        # Number of consecutive frames to confirm fall
ANGLE_THRESHOLD = 30       # Torso angle threshold (degrees)
RESIZE_WIDTH = 640

# ==============================
# LOAD MODEL
# ==============================

print("[INFO] Loading model...")
model = YOLO(MODEL_PATH)
print("[INFO] Model loaded successfully.")

# ==============================
# TORSO ANGLE CALCULATION
# ==============================

def compute_torso_angle(kpts):
    """
    Compute torso angle using shoulder and hip midpoint.
    """
    left_shoulder = kpts[5][:2]
    right_shoulder = kpts[6][:2]
    left_hip = kpts[11][:2]
    right_hip = kpts[12][:2]

    shoulder_mid = (left_shoulder + right_shoulder) / 2
    hip_mid = (left_hip + right_hip) / 2

    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]

    angle = np.degrees(np.arctan2(dy, dx))
    return abs(angle)

# ==============================
# MAIN FUNCTION
# ==============================

def main():

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Camera not detected.")
        return

    print("[INFO] Camera started.")

    fall_counter = 0
    prev_time = 0

    try:
        while True:

            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame.")
                break

            # Resize for performance
            height, width = frame.shape[:2]
            scale = RESIZE_WIDTH / width
            frame = cv2.resize(frame, (RESIZE_WIDTH, int(height * scale)))

            # Inference
            results = model(frame)

            fall_detected = False

            for r in results:
                if r.keypoints is not None:

                    keypoints = r.keypoints.xy.cpu().numpy()

                    for person in keypoints:
                        angle = compute_torso_angle(person)

                        # If torso nearly horizontal
                        if angle < ANGLE_THRESHOLD:
                            fall_counter += 1
                        else:
                            fall_counter = 0

                        if fall_counter > CONFIRM_FRAMES:
                            fall_detected = True

            # Display status
            if fall_detected:
                label = "FALL DETECTED"
                color = (0, 0, 255)
            else:
                label = "Normal"
                color = (0, 255, 0)

            # FPS calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            cv2.putText(frame, label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            cv2.putText(frame, f"FPS: {int(fps)}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            cv2.imshow("Advanced Fall Detection", frame)

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera safely closed.")

# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":
    main()