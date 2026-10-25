"""Step 5: lane curvature and car position"""

import cv2
import numpy as np

from lane_detection import LaneFit

XM_PER_PIX = 3.7 / 640
YM_PER_PIX = 12.2 / 300


def curvature_radius(fit: np.ndarray, y: float) -> float:
    a = fit[0] * XM_PER_PIX / YM_PER_PIX**2
    b = fit[1] * XM_PER_PIX / YM_PER_PIX
    y_m = y * YM_PER_PIX
    if a == 0:
        return float("inf")
    return float((1 + (2 * a * y_m + b) ** 2) ** 1.5 / abs(2 * a))


def lane_curvature(lane: LaneFit, image_height: int) -> float:
    return curvature_radius(lane.left_fit, image_height - 1)


def vehicle_offset(
    lane: LaneFit, image_size: tuple[int, int], matrix: np.ndarray
) -> float:
    width, height = image_size
    car = cv2.perspectiveTransform(
        np.float32([[[width / 2, height - 1]]]), matrix
    )[0, 0]
    lane_center = (
        np.polyval(lane.left_fit, car[1]) + np.polyval(lane.right_fit, car[1])
    ) / 2
    return float((car[0] - lane_center) * XM_PER_PIX)
