import cv2

def detect_fov(frame, baseline):

    reference = baseline["reference_frame"]

    # Convert to grayscale
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ORB Feature Detector
    orb = cv2.ORB_create(1000)

    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_cur, None)

    if des1 is None or des2 is None:
        return {
            "fov_change": True,
            "good_matches": 0
        }

    # KNN Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    match_count = len(good_matches)

    # Threshold (adjust after testing)
    fov_change = match_count < 100

    return {
        "fov_change": fov_change,
        "good_matches": match_count
    }