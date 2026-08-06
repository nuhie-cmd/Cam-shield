from collections import deque
from datetime import datetime
import os
import cv2

class EvidenceManager:
    def __init__(self, fps=30, buffer_sec=5):
        self.fps=fps
        self.buffer=deque(maxlen=fps*buffer_sec)

        os.makedirs("evidence/snapshots",exist_ok=True)
        os.makedirs("evidence/videos",exist_ok=True)


    def add_frame(self, frame):
        self.buffer.append(frame.copy())   

    def save_snapshot(self, frame):
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        filename=f"evidence/snapshots/{timestamp}.jpg"

        cv2.imwrite(filename, frame)

        return filename

    def save_video(self):
        if len(self.buffer)==0:
            return None
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        filename=f"evidence/videos/{timestamp}.mp4"
        height, width = self.buffer[0].shape[:2]

        fourcc=cv2.VideoWriter_fourcc(*"mp4v")

        writer=cv2.VideoWriter( filename, fourcc, self.fps, (width,height))
        for frame in self.buffer:
            writer.write(frame)
        writer.release()
        return filename    