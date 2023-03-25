"""
    This module contains the code for object detection in a given
    frame. The frame can be either a video or an image.
    The code in this file is inspired from: https://github.com/amdegroot/ssd.pytorch
    Credits: ssd.pytorch authors

    @author: Yaksh J Haranwala
"""
import warnings

import cv2 as cv
import numpy as np
import torch

from ssd.ssd import build_ssd
from ssd.data.__init__ import BaseTransform
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

warnings.simplefilter("ignore", category=UserWarning)


class ObjectDetection:
    def __init__(self):
        self.net = build_ssd('test')
        self.net.load_weights('./ssd300_mAP_77.43_v2.pth')
        self.net.load_state_dict(torch.load('./ssd300_mAP_77.43_v2.pth'))
        self.COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        self.transform = BaseTransform(self.net.size, (104 / 256.0, 117 / 256.0, 123 / 256.0))

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
                frame: np.ndarray
                    The current video frame with bbox drawn around detected objects.
        """
        height, width = frame.shape[:2]
        x = torch.from_numpy(self.transform(frame)[0]).permute(2, 0, 1)
        if torch.cuda.is_available():
            x = x.cuda()
        x = x.unsqueeze(0)
        y = self.net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        bboxes = []
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.1:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                if (len(pt)) > 0:
                    t_l = pt[0], pt[1]
                    t_r = pt[2], pt[1]
                    b_l = pt[0], pt[3]
                    b_r = pt[2], pt[3]
                    bboxes.append(np.array([t_l, t_r, b_r, b_l], dtype=np.int32))
                cv.rectangle(frame,
                             (int(pt[0]), int(pt[1])),
                             (int(pt[2]), int(pt[3])),
                             self.COLORS[i % 3], 2)

                j += 1
        return frame, bboxes
