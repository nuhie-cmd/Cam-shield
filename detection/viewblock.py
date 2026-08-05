import cv2
import numpy as np

def detect_viewblock(frame, threshold=25):
    grey=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contrast=np.std(grey)
    blocked = contrast < threshold
    return blocked,contrast
<<<<<<< HEAD
 
=======
>>>>>>> ec2c5b8005117b8a1b3d1c211eafa1f64cd31940
