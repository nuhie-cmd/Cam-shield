import cv2
from detection.viewblock import detect_viewblock
from evidence.evidence_manager import EvidenceManager
from evidence.alert_service import AlertService

ip='172.16.10.235'
user='camera2'
password='Camera@235'

url=f"rtsp://{'camera2'}:{'Camera@235'}@{'172.16.10.235'}:554/cam/realmonitor?channel=1&subtype=0"

cap=cv2.VideoCapture(url)

if not cap.isOpened():
    print("Failed Connecting")
    exit()
print("Connected")

evidence=EvidenceManager(fps=30)
alert=AlertService()
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

    evidence.add_frame(frame)
    if blocked:
        snapshot = evidence.save_snapshot(frame)
        video = evidence.save_video()

        alert.send_alert(
        event="Camera View Blocked",
        threat_score=95,
        snapshot_path=snapshot
        )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
<<<<<<< HEAD
cv2.destroyAllWindows()
=======
cv2.destroyAllWindows()
>>>>>>> ec2c5b8005117b8a1b3d1c211eafa1f64cd31940
