import cv2

ip='172.16.11.156'
user='camera3'
password='Camera@156'

url=f"rtsp://{'camera3'}:{'Camera@156'}@{'172.16.11.156'}:554/cam/realmonitor?channel=1&subtype=0"

cap=cv2.VideoCapture(url)

if not cap.isOpened():
    print("Failed Connecting")
    exit()
print("Connected ")

while True:
    ret,frame=cap.read()
    if not ret:
        print('Failed Live Feed')
        break
    cv2.imshow("CamShield Live Feed",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
         
