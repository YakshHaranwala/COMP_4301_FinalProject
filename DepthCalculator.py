"""
    Using the MiDaS model from PyTorch, calculate the monocular
    depth map for the current video/image.
    Credits: https://pytorch.org/hub/intelisl_midas_v2/

    @authors: Yaksh J Haranwala, Salman Haidri
"""
import torch
import numpy as np
import cv2 as cv2


class MidasDepth:
    def __init__(self, model_type):
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def calculate_depth(self, frame):
        """
            Using the MiDaS model, estimate the inverse depth of the current frame.

            Parameters
            ----------
                frame: np.ndarray
                    The frame of which the depth map is to be calculated.

            Returns
            -------
                np.ndarray:
                    The actual depth map values.
                np.ndarray:
                    The same depth map but modified to display.
        """
        # Convert to RGB since opencv uses the BGR format and torch uses RGB.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

        depth_map = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        depth_map = (depth_map * 255).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

        return output, depth_map
