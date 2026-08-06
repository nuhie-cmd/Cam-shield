import cv2

def detect_tilt(frame, baseline):

    reference = baseline["reference_frame"]

    # Convert to grayscale
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ORB Detector
    orb = cv2.ORB_create(500)

    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_cur, None)

    # No descriptors found
    if des1 is None or des2 is None:
        return {
            "tilt": True,
            "matches": 0
        }

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    match_count = len(matches)

    # Threshold
    tilt = match_count < 80

    return {
        "tilt": tilt,
        "matches": match_count
    }