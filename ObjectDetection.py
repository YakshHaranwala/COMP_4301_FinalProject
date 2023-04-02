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

    def detect_objects(self, frame):
        """
            Given the current frame in the video, detect all the objects that
            the neural net can.

            Parameters
            ----------
                frame: np.ndarray
                    The current video frame

            Returns
            -------
                np.array:
                    Array with bounding boxes.
        """
        # Predict using the YoloV8 model.
        results = self.net.predict(frame, verbose=False)

        # These are the bounding boxes that we will pass to the distance calculator.
        bounding_boxes = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get the bounding boxes for distance calculation purpose.
                b_box = np.array(box.xyxy[0].cpu()).astype(np.int16)
                bounding_boxes.append([
                    (b_box[0], b_box[1]),
                    (b_box[2], b_box[1]),
                    (b_box[2], b_box[3]),
                    (b_box[0], b_box[3]),
                ])

        return np.array(bounding_boxes)
