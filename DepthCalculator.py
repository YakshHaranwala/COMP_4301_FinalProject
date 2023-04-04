"""
    Using the MiDaS model from PyTorch, calculate the monocular
    depth map for the current video/image.
    Credits: https://pytorch.org/hub/intelisl_midas_v2/

    @authors: Yaksh J Haranwala, Salman Haidri
"""
import torch
import numpy as np
import cv2 as cv


class MidasDepth:
    def __init__(self, model_type):
        self.model_type = "MiDaS_small"
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def calculate_depth(self, frame):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        input_batch = self.transform(frame).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()

        return output