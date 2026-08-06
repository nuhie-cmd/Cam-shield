import cv2
import numpy as np

# Store previous frame
previous_frame = None

# Count how many consecutive frames are almost identical
freeze_count = 0

def detect_freeze(frame):

    global previous_frame
    global freeze_count

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # First frame
    if previous_frame is None:
        previous_frame = gray
        return {
            "freeze": False,
            "difference": 100.0
        }

    # Compare current frame with previous frame
    difference = cv2.absdiff(previous_frame, gray)

    mean_difference = np.mean(difference)

    # If frames are almost identical
    if mean_difference < 2:
        freeze_count += 1
    else:
        freeze_count = 0

    previous_frame = gray

    # Freeze if identical for ~30 frames
    freeze = freeze_count >= 30

    return {
        "freeze": freeze,
        "difference": mean_difference
    }