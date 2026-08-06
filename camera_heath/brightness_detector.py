import cv2
import numpy as np

def detect_brightness(frame, baseline):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    current_brightness = np.mean(gray)

    baseline_brightness = baseline["brightness"]

    difference = abs(current_brightness - baseline_brightness)

    THRESHOLD =30
    brightness_change = bool(difference > THRESHOLD)

    return {

        "brightness_change": brightness_change,

        "current_brightness": current_brightness,

        "baseline_brightness": baseline_brightness,

        "difference": difference,
          }