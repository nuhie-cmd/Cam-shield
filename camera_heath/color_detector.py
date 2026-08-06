import cv2

def detect_color(frame, baseline):

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Calculate current histogram
    current_hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [50, 60],
        [0, 180, 0, 256]
    )

    cv2.normalize(current_hist, current_hist)

    # Compare with baseline histogram
    similarity = cv2.compareHist(
        baseline["histogram"],
        current_hist,
        cv2.HISTCMP_CORREL
    )

    # Threshold
    THRESHOLD = 0.85

    color_change = similarity < THRESHOLD

    return {
        "color_change": color_change,
        "similarity": similarity
    }