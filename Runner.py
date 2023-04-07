"""
    This file contains the code for executing the program.
    It takes care of instantiating all the modules (Object Detection, Distance Calculation and Voice Command).
    It starts the video capture and runs all the processes and handles
    any errors if they occur.

    @author: Yaksh J Haranwala
"""
# Credits: https://github.com/jaimin-k/Monocular-Depth-Estimation-using-MiDaS/blob/main/MiDaS%20Depth%20Sensing.ipynb
# MiDaS: https://pytorch.org/hub/intelisl_midas_v2/
# YoloV8: https://github.com/ultralytics/ultralytics

import cv2
import requests
import imutils
import numpy as np
from ObjectDetection import ObjectDetection
from DepthCalculator import MidasDepth


if __name__ == '__main__':
    detector = ObjectDetection()
    depthCalculator = MidasDepth("MiDaS_small")

    url = "http://192.168.2.56:8080/shot.jpg"
    # Start the video capture.
    # cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    # if not cap.isOpened():
    #     print("Error opening video capture.")

    # While the video capture is on, detect the objects and
    # provide guidance to the object.
    while True:
        # grab next frame
        # ret, frame = cap.read()
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        frame = imutils.resize(img, width=640, height=640)

        # Check if the frame was read successfully
        # if not ret:
        #     print("Error reading frame from video stream.")
        #     break

        # Display the frame
        output = depthCalculator.calculate_depth(frame)
        detector.detect_objects(frame, output)
        detector.draw_roi(frame)
        depth_map = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        depth_map = (depth_map * 255).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
        cv2.imshow("Live Depth Map", depth_map)
        cv2.imshow("Live Prediction", frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # cap.release()
    cv2.destroyAllWindows()
