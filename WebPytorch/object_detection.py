import os
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO


class ObjectDetection:
    def __init__(self):
        PROJECT_PATH = os.path.abspath(os.getcwd())
        MODELS_PATH = os.path.join(PROJECT_PATH, "models")

        # Load the YOLOv8 model using Ultralytics and PyTorch
        self.MODEL = YOLO(os.path.join(MODELS_PATH, "best.pt"))

        # Get class names from the model
        self.CLASSES = self.MODEL.names

        # Generate random colors for each class
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        self.COLORS /= (np.sum(self.COLORS**2, axis=1)**0.5/255)[np.newaxis].T

    def detectObj(self, snap):
        height, width, channels = snap.shape

        # Convert the image to a tensor and run inference
        results = self.MODEL(snap)

        # Parse detection results
        boxes = results[0].boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            label = self.CLASSES[class_id]

            # Draw the bounding box and label on the image
            color = self.COLORS[class_id]
            cv2.rectangle(snap, (x1, y1), (x2, y2), color, 2)
            cv2.putText(snap, f'{label} {confidence:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        return snap


class VideoStreaming(object):
    def __init__(self):
        super(VideoStreaming, self).__init__()
        self.VIDEO = cv2.VideoCapture(0)

        self.MODEL = ObjectDetection()

        self._preview = True
        self._flipH = False
        self._detect = False
        self._exposure = self.VIDEO.get(cv2.CAP_PROP_EXPOSURE)
        self._contrast = self.VIDEO.get(cv2.CAP_PROP_CONTRAST)

    @property
    def preview(self):
        return self._preview

    @preview.setter
    def preview(self, value):
        self._preview = bool(value)

    @property
    def flipH(self):
        return self._flipH

    @flipH.setter
    def flipH(self, value):
        self._flipH = bool(value)

    @property
    def detect(self):
        return self._detect

    @detect.setter
    def detect(self, value):
        self._detect = bool(value)

    @property
    def exposure(self):
        return self._exposure

    @exposure.setter
    def exposure(self, value):
        self._exposure = value
        self.VIDEO.set(cv2.CAP_PROP_EXPOSURE, self._exposure)

    @property
    def contrast(self):
        return self._contrast

    @contrast.setter
    def contrast(self, value):
        self._contrast = value
        self.VIDEO.set(cv2.CAP_PROP_CONTRAST, self._contrast)

    def show(self):
        while self.VIDEO.isOpened():
            ret, snap = self.VIDEO.read()
            if self.flipH:
                snap = cv2.flip(snap, 1)

            if ret:
                if self._preview:
                    if self.detect:
                        snap = self.MODEL.detectObj(snap)

                else:
                    snap = np.zeros((
                        int(self.VIDEO.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        int(self.VIDEO.get(cv2.CAP_PROP_FRAME_WIDTH))
                    ), np.uint8)
                    label = "camera disabled"
                    H, W = snap.shape
                    font = cv2.FONT_HERSHEY_PLAIN
                    color = (255, 255, 255)
                    cv2.putText(snap, label, (W // 2 - 100, H // 2),
                                font, 2, color, 2)

                frame = cv2.imencode(".jpg", snap)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.01)

            else:
                break
        print("off")


# Usage example
if __name__ == "__main__":
    video_stream = VideoStreaming()
    for frame in video_stream.show():
        pass