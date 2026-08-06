import cv2

# Camera Credentials
ip = "172.16.10.235"
user = "camera2"
password = "Camera%40235"

url = f"rtsp://{user}:{password}@{ip}:554/cam/realmonitor?channel=1&subtype=0"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Failed Connecting")
    exit()

print("Connected Successfully")