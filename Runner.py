"""
    This file contains the code for executing the program.
    It takes care of instantiating all the modules (Object Detection, Distance Calculation and Voice Command).
    It starts the video capture and runs all the processes and handles
    any errors if they occur.

    @author: Yaksh J Haranwala
"""
import cv2 as cv
from ObjectDetection import ObjectDetection
from DistanceCalculator import DistanceCalculator

if __name__ == '__main__':
    detector = ObjectDetection()
    distance = DistanceCalculator()

    # Start the video capture.
    cap = cv.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error opening video capture.")

    # While the video capture is on, detect the objects and
    # provide guidance to the object.
    while True:
        # grab next frame
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            print("Error reading frame from video stream.")
            break

        # Display the frame
        bboxes = detector.detect_objects(frame)
        frame = distance.calculate_distances(frame, bboxes, 36)
        cv.imshow("Live Prediction", frame)

        # Exit if the 'q' key is pressed
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
