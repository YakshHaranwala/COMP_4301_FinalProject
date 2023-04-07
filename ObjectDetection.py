"""
    This module contains the code for object detection in a given
    frame. The frame can be either a video or an image.
    Credits: https://github.com/ultralytics/ultralytics

    @author: Yaksh J Haranwala
"""
import warnings

import cv2 as cv
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator
import torch
import numpy as np


class ObjectDetection:
    def __init__(self):
        self.net = YOLO("yolov8n.pt")

    def detect_objects(self, frame, depth_array):
        """
            Given the current frame in the video, detect all the objects that
            the neural net can.

            Parameters
            ----------
                frame: np.ndarray
                    The current video frame
                depth_array: np.array
                    The array containing depth values for each pixel that
                    we calculated using MiDaS.

        """
        # Predict using the YoloV8 model.
        results = self.net.predict(frame, verbose=False, conf=0.25)

        # Used to draw bounding boxes on the detected objects.
        annotator = Annotator(frame)

        # These are the bounding boxes that we will pass to the distance calculator.
        bounding_boxes = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b_box = np.array(box.xyxy[0].cpu()).astype(np.int16)
                depth_box = depth_array[(b_box[1]+b_box[3])//2, (b_box[0]+b_box[2])//2]
                threshold = np.average(depth_box) / 1000

                if threshold < 0.35:
                    annotator.box_label(b_box, f"far {threshold}")
                else:
                    annotator.box_label(b_box, f"near")
                    cv.putText(frame, f'Dur reh lavde', (int(b_box[0]), int(b_box[3] - 25)),
                               cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 25), 2)


