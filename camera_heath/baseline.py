import cv2
import numpy as np

def learn_baseline(cap):

    brightness_list = []
    sharpness_list = []
    hist_list = []

    print("Learning Camera Baseline...")

    for i in range(150):

        ret, frame = cap.read()

        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness = np.mean(gray)

        sharpness = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        
        hist = cv2.calcHist(
            [hsv],
            [0,1],
            None,
            [50,60],
            [0,180,0,256]
        )

        cv2.normalize(hist,hist)

        brightness_list.append(brightness)
        sharpness_list.append(sharpness)
        hist_list.append(hist)

    baseline = {

        "brightness": np.mean(brightness_list),

        "sharpness": np.mean(sharpness_list),

        "histogram": np.mean(hist_list,axis=0),

        "reference_frame": frame

    }

    print("Baseline Learned")

    return baseline