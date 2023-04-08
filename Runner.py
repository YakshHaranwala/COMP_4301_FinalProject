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
# Inspiration: https://kananvyas.medium.com/obstacle-detection-and-navigation-through-tensorflow-api-943728c33243

import cv2
import requests
import imutils
import numpy as np
from ObjectDetection import ObjectDetection
from DepthCalculator import MidasDepth


if __name__ == '__main__':
    detector = ObjectDetection()
    depthCalculator = MidasDepth("MiDaS_small")

    try:
        url = "http://192.168.2.93:8080/shot.jpg"

        # While the video capture is on, detect the objects and
        # provide guidance to the object.
        while True:
            # grab next frame
            img_resp = requests.get(url)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            frame = cv2.imdecode(img_arr, -1)

            # Calculate the depth, detect the objects and provide guidance.
            inv_depth, disp_depth_map = depthCalculator.calculate_depth(frame)
            detector.detect_objects(frame, inv_depth)

            # Display the depth map and the guidance frame.
            cv2.imshow("Live Depth Map", disp_depth_map)
            cv2.imshow("Live Prediction", frame)

            # Exit if the 'q' key is pressed
            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()

    except:
        print("Could not find phone camera, using device webcam instead.")
        # Start the video capture.
        cap = cv2.VideoCapture(0)

        # Check if the camera is opened successfully
        if not cap.isOpened():
            print("Error opening video capture.")

        while True:
            # grab next frame
            ret, frame = cap.read()

            # Check if the frame was read successfully
            if not ret:
                print("Error reading frame from video stream.")
                break

            # Calculate the depth, detect the objects and provide guidance.
            inv_depth, disp_depth_map = depthCalculator.calculate_depth(frame)
            detector.detect_objects(frame, inv_depth)

            # Display the depth map and the guidance frame.
            cv2.imshow("Live Depth Map", disp_depth_map)
            cv2.imshow("Live Prediction", frame)

            # Exit if the 'q' key is pressed
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
