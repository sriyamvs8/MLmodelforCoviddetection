# import torch
# import torch.nn as nn
# from torchvision import models

# def load_model():

#     model = models.densenet121(
#         weights=None
#     )

#     model.classifier = nn.Linear(
#         model.classifier.in_features,
#         3
#     )

#     model.load_state_dict(
#         torch.load(
#             "final_model.pth",
#             map_location="cpu"
#         )
#     )

#     model.eval()

#     return model

import torch
import torch.nn as nn
import numpy as np
import cv2

from torchvision import models
from pytorch_grad_cam import GradCAM


def load_model():

    model = models.densenet121(weights=None)

    model.classifier = nn.Linear(
        model.classifier.in_features,
        3
    )

    model.load_state_dict(
        torch.load(
            "final_model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


def generate_gradcam(model, img_tensor):

    target_layers = [model.features.denseblock4]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=img_tensor
    )[0]

    return grayscale_cam


def get_hotspot(cam):

    heatmap = np.uint8(cam * 255)

    _, thresh = cv2.threshold(
        heatmap,
        180,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    largest = max(
        contours,
        key=cv2.contourArea
    )

    x, y, w, h = cv2.boundingRect(
        largest
    )

    return x, y, w, h


def region_name(x, y):

    side = "Left Lung" if x < 112 else "Right Lung"

    if y < 75:
        zone = "Upper"
    elif y < 150:
        zone = "Middle"
    else:
        zone = "Lower"

    return f"{zone} {side}"


def severity_score(cam):

    severity = float(cam.mean() * 100)

    if severity < 20:
        level = "Mild"
    elif severity < 40:
        level = "Moderate"
    else:
        level = "Severe"

    return round(severity, 2), level