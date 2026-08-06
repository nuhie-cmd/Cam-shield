import cv2

from stream_reader import cap
from baseline import learn_baseline
from blur_detector import detect_blur
from brightness_detector import detect_brightness
from color_detector import detect_color
from freeze_detector import detect_freeze
from obstruction_detector import detect_obstruction
from tilt_detector import detect_tilt
from fov_detector import detect_fov
from health_score import calculate_health_score
from json_output import generate_json

# -------------------------------
# Learn Camera Baseline
# -------------------------------

baseline = learn_baseline(cap)

print("Camera Health Monitoring Started...")

# -------------------------------
# Live Monitoring
# -------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Live Feed Lost")
        break

    # Blur Detection
    blur_result = detect_blur(frame, baseline)
    #Brightness Detection
    brightness_result = detect_brightness(frame, baseline)
    color_result = detect_color(frame, baseline)
    freeze_result = detect_freeze(frame)
    obstruction_result = detect_obstruction(frame)
    tilt_result = detect_tilt(frame, baseline)
    fov_result = detect_fov(frame, baseline)
    health_result = calculate_health_score(
        blur_result,
        brightness_result,
        color_result,
        freeze_result,
        obstruction_result,
        tilt_result,
        fov_result
    )
    json_output = generate_json(
        blur_result,
        brightness_result,
        color_result,
        freeze_result,
        obstruction_result,
        tilt_result,
        fov_result,
        health_result
    )
    print("Health Score :",health_result)
    print("JSON Output :",json_output)
    print("FOV :",fov_result)
    print("Tilt :",tilt_result)
    print("Obstruction :",obstruction_result)
    print("Freeze :",freeze_result)
    print("Color:",color_result)
    print("Brightness :",brightness_result)

    # Display Blur Status
    if blur_result["blur"]:
        blur_color = (0, 0, 255)      # Red
    else:
        blur_color = (0, 255, 0)      # Green

    cv2.putText(
        frame,
        f"Blur : {blur_result['blur']}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        blur_color,
        2
    )

    cv2.putText(
        frame,
        f"Sharpness : {blur_result['sharpness']:.2f}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Sharpness Ratio : {blur_result['ratio']:.2f}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
    frame,
    f"Health Score : {health_result['score']}",
    (20,420),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0,255,0) if health_result["score"] >= 75 else (0,0,255),
    2
)

cv2.putText(
    frame,
    f"Status : {health_result['status']}",
    (20,455),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0,255,0) if health_result["score"] >= 75 else (0,0,255),
    2
)

cv2.imshow("Camera Health Monitor", frame)

if cv2.waitKey(1) & 0xFF == ord('q'):  
 exit(0)

cap.release()
cv2.destroyAllWindows()