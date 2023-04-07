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
        self.messages = {"12": "Move Right!", "23": "Move Left!", "123":  "Warning: Stop!"}

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
        rows, cols = frame.shape[:2]
        rect_top_left = [int(cols * 0.15), int(rows * 0.05)]
        rect_bottom_right = [int(cols * 0.85), int(rows * 0.95)]
        vertices = np.array([[rect_top_left, rect_bottom_right]], dtype=np.int32)
        cv.rectangle(frame, rect_top_left, rect_bottom_right, (0, 0, 255), 3)
        # calculate the x-coordinates of the two vertical lines

        third_width = (rect_bottom_right[0]-rect_top_left[0])//3
        x1 = rect_top_left[0] + third_width
        x1_bottom = [x1, rect_bottom_right[1]]
        x2 = rect_bottom_right[0] - third_width
        x2_bottom = [x2, rect_bottom_right[1]]


        # draw the two vertical lines
        cv.line(frame, (x1, rect_top_left[1]), (x1, rect_bottom_right[1]), (0, 0, 255), 3)
        cv.line(frame, (x2, rect_top_left[1]), (x2, rect_bottom_right[1]), (0, 0, 255), 3)
        # These are the bounding boxes that we will pass to the distance calculator.
        bounding_boxes = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b_box = np.array(box.xyxy[0].cpu()).astype(np.int16)
                depth_box = depth_array[(b_box[1] + b_box[3]) // 2, (b_box[0] + b_box[2]) // 2]
                threshold = np.average(depth_box) / 1000

                if threshold < 0.55:
                    continue

                obj_top_left = (b_box[0], b_box[1])
                obj_bottom_right = (b_box[2], b_box[3])

                # stopping all bboxes not overlapping with roi
                if not self.overlap(rect_top_left, rect_bottom_right, obj_top_left, obj_bottom_right):
                    continue

                verify = []
                regions = [ [rect_top_left, x1_bottom],
                            [(rect_top_left[0] + third_width, rect_top_left[1]), x2_bottom],
                            [(rect_top_left[0] +2*third_width, rect_top_left[1]), rect_bottom_right]
                          ]
                for i in range(3):
                    if self.overlap(regions[i][0], regions[i][1], obj_top_left, obj_bottom_right):
                        verify.append(str(i+1))

                verify = ''.join(verify)
                annotator.box_label(b_box, "near")
                cv.putText(frame, f'{self.messages.get(verify, "")}',
                           (int(rect_top_left[0]+rect_bottom_right[0]/2),
                           int(rect_top_left[1]+rect_bottom_right[1]/2)),
                           cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 25), 2)
        


    def overlap(self, topleft1, bottomright1, topleft2, bottomright2):
        if topleft1[0] < bottomright2[0] and topleft2[0] < bottomright1[0]:
            # Check if the rectangles overlap on the y-axis
            if topleft1[1] < bottomright2[1] and topleft2[1] < bottomright1[1]:
                # The rectangles overlap
                return True
        # The rectangles do not overlap
        return False


