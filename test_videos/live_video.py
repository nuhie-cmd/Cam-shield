import cv2
from detection.viewblock import detect_viewblock

ip='172.16.10.235'
user='camera2'
password='Camera@235'

url=f"rtsp://{'camera2'}:{'Camera@235'}@{'172.16.10.235'}:554/cam/realmonitor?channel=1&subtype=0"

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
    blocked,ratio=detect_viewblock(frame)
    print(blocked)
    print(ratio)
    if blocked:
        cv2.putText(frame,"CAMERA VIEW BLOCKED",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    cv2.imshow("CamShield Live Feed",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
         
