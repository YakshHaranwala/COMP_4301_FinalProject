"""
    This module contains the code that calculates distance between bounding boxes.

    @author: Yaksh J Haranwala
"""
# TODO: At the moment, the code is taking the left most object as the reference object for calculating the distance
#  between bounding boxes. Furthermore, the distances seem inaccurate at the moment. Need to meet and discuss about
#  this.

import numpy as np
import cv2 as cv
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours


class DistanceCalculator:
    def __init__(self):
        pass

    def midpoint(self, ptA, ptB):
        return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5

    def calculate_distances(self, image, bboxes, guided_object_width):
        colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),
                  (255, 0, 255))
        refObj, orig = None, None
        refCoords, objCoords = [], []

        # loop over the contours individually
        for box in bboxes:
            # compute the center of the bounding box
            cX = np.average(box[:, 0])
            cY = np.average(box[:, 1])

            # Now we need to set a bounding box as a reference bounding box.
            # So we take the first bounding box as the reference bounding box
            # and then the distances are calculated accordingly.
            if refObj is None:
                # unpack the ordered bounding box, then compute the
                # midpoint between the top-left and top-right points,
                # followed by the midpoint between the top-right and
                # bottom-right
                (tl, tr, br, bl) = box
                (tlblX, tlblY) = self.midpoint(tl, bl)
                (trbrX, trbrY) = self.midpoint(tr, br)
                # compute the Euclidean distance between the midpoints,
                # then construct the reference object
                D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                refObj = (box, (cX, cY), D / guided_object_width)
                continue

            # draw the contours on the image
            orig = image.copy()
            cv.rectangle(orig, box[0], box[2], (255, 0, 0), 2)
            cv.rectangle(orig, refObj[0][0], refObj[0][2], (255, 0, 0), 2)

            # stack the reference coordinates and the object coordinates
            # to include the object center
            refCoords = np.vstack([refObj[0][1], refObj[0][2]])
            objCoords = np.vstack([box[0], box[3]])

        # loop over the original points
        for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
            # compute the Euclidean distance between the coordinates,
            # and then convert the distance in pixels to distance in
            # units
            D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]

            # draw circles corresponding to the current points and
            # connect them with a line
            cv.circle(orig, (int(xA), int(yA)), 5, color, -1)
            cv.circle(orig, (int(xB), int(yB)), 5, color, -1)
            cv.line(orig, (int(xA), int(yA)), (int(xB), int(yB)),
                    color, 2)
            (mX, mY) = self.midpoint((xA, yA), (xB, yB))
            cv.putText(orig, "{:.1f}in".format(D), (int(mX), int(mY - 10)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        return orig if orig is not None else image
