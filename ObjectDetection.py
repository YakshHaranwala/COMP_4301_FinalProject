"""
    This module contains the code for object detection in a given
    frame. The frame can be either a video or an image.
    Credits: https://github.com/ultralytics/ultralytics

    @author: Yaksh J Haranwala, Abdul Shaji, Salman Haidri, Mohammad Shoaib
"""
import warnings

import cv2 as cv
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator
import torch
import numpy as np


class ObjectDetection:
    # Messages that we will put on the frame once we determine where to go.
    _MESSAGES = {"12": "Move Right!", "23": "Move Left!", "123": "Warning: Stop!"}

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
        detected_objects = self.net.predict(frame, verbose=False, conf=0.25)

        # Used to draw bounding boxes on the detected objects.
        annotator = Annotator(frame)

        # Here, we are calculating the dimensions of a rectangle within which
        # we will provide guidance to the object. This rectangle will be essentially
        # a scaled down version of the entire frame.
        width, height = frame.shape[:2]
        guidance_rect_tl = [int(height * 0.15), int(width * 0.05)]
        guidance_rect_br = [int(height * 0.85), int(width * 0.95)]
        cv.rectangle(frame, guidance_rect_tl, guidance_rect_br, (128, 0, 128), 2)

        # Now, divide the guidance rectangle into 3 equal parts so that
        # we can provide the object as to which sector to go towards.
        third_width = (guidance_rect_br[0] - guidance_rect_tl[0]) // 3
        x1 = guidance_rect_tl[0] + third_width
        x1_bottom = [x1, guidance_rect_br[1]]
        x2 = guidance_rect_br[0] - third_width
        x2_bottom = [x2, guidance_rect_br[1]]

        # draw the two vertical lines
        cv.line(frame, (x1, guidance_rect_tl[1]), (x1, guidance_rect_br[1]), (128, 0, 128), 2)
        cv.line(frame, (x2, guidance_rect_tl[1]), (x2, guidance_rect_br[1]), (128, 0, 128), 2)

        # Now, we will let our Yolo model detect objects and provide guidance.
        for obj in detected_objects:
            boxes = obj.boxes
            for box in boxes:
                # First, get the bounding box of each object, and check the
                # inverse depth values of the bounding box, calculate the average
                # and determine the threshold of whether the object is nearby or far.
                b_box = np.array(box.xyxy[0].cpu()).astype(np.int16)
                depth_box = depth_array[(b_box[1] + b_box[3]) // 2, (b_box[0] + b_box[2]) // 2]
                threshold = np.average(depth_box) / 1000

                # If the object detected is far off, we can assume that it is not going
                # to bother us in our path, and we can ignore it.
                if threshold < 0.4:
                    continue

                # Now if the object is close, then calculate what the object's bounding box
                # top left and bottom right are.
                obj_top_left = (b_box[0], b_box[1])
                obj_bottom_right = (b_box[2], b_box[3])

                # These are the coordinates for the 3 regions in our guidance rectangle.
                regions = [
                    [guidance_rect_tl, x1_bottom],
                    [(guidance_rect_tl[0] + third_width, guidance_rect_tl[1]), x2_bottom],
                    [(guidance_rect_tl[0] + 2 * third_width, guidance_rect_tl[1]), guidance_rect_br]
                ]

                # Check whether the object's bounding box overlaps with any part
                # of each of our regions.
                blocked = []
                for i in range(len(regions)):
                    if self.overlap(regions[i][0], regions[i][1], obj_top_left, obj_bottom_right):
                        blocked.append(str(i + 1))

                # join the numbers of the regions that are blocked, this will help
                # us in getting the required navigation value from our MESSAGES dictionary.
                blocked = ''.join(blocked)
                annotator.box_label(b_box, "near", color=(0, 128, 128), txt_color=(0, 0, 0))
                cv.putText(frame, f'{ObjectDetection._MESSAGES.get(blocked, "")}',
                           (int(guidance_rect_tl[0] + guidance_rect_br[0] / 2 - 125),
                            int(guidance_rect_tl[1] + guidance_rect_br[1] / 2)),
                           cv.FONT_HERSHEY_DUPLEX, 1.25, (0, 255, 0), 2)

    def overlap(self, tl_one, br_one, tl_two, br_two):
        """
            Check whether two given bounding boxes overlap at all or not.

            Parameters
            ----------
                tl_one: List or Tuple
                    The coordinates of the top left of the first bounding box.
                br_one: List or Tuple
                    The coordinates of the bottom right of the first bounding box.
                tl_two: List or Tuple
                    The coordinates of the top left of the second bounding box.
                br_two: List or Tuple
                    The coordinates of the bottom right of the second bounding box.

            Returns
            -------
                bool:
                    Indication of whether the two bounding boxes are overlapping or not.

        """
        if (tl_one[1] < br_two[1] and tl_two[1] < br_one[1]) and (tl_one[0] < br_two[0] and tl_two[0] < br_one[0]):
            return True

        return False
