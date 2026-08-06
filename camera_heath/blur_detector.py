import cv2

def detect_blur(frame, baseline):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    current_sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    baseline_sharpness = baseline["sharpness"]

    ratio = current_sharpness / baseline_sharpness

    if ratio < 0.60:
        blur = True
    else:
        blur = False

    return {

        "blur": blur,

        "sharpness": current_sharpness,

        "ratio": ratio

    }