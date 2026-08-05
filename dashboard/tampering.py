import cv2
import numpy as np

DARK_THRESHOLD = 25      # Darkness threshold
BLUR_THRESHOLD = 30    # Blur threshold

def check_tampering(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Average briqqghtness
    brightness = np.mean(gray)

    # Blur detection
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    if brightness < DARK_THRESHOLD:
        return True, "Camera Covered"

    if blur < BLUR_THRESHOLD:
        return True, "Camera Blurred"

    return False, ""