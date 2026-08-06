import cv2
import numpy as np

def detect_obstruction(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Average brightness
    brightness = np.mean(gray)

    # Edge detection
    edges = cv2.Canny(gray, 100, 200)

    edge_count = np.count_nonzero(edges)

    # Pixel variance
    variance = np.var(gray)

    obstruction = False

    # Camera is probably covered
    if brightness < 40 and edge_count < 800 and variance < 200:
        obstruction = True

    return {

        "obstruction": obstruction,

        "brightness": brightness,

        "edge_count": edge_count,

        "variance": variance

    }