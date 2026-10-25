"""Step 5: lane curvature and vehicle position in meters."""

import cv2
import numpy as np

from lane_detection import LaneFit

# Meters per pixel in the bird's-eye view.
# x: US lane width 3.7 m is about 640 px on the straight-road images.
# y: dashed lines repeat every 40 ft (12.2 m), which is about 300 px.
XM_PER_PIX = 3.7 / 640
YM_PER_PIX = 12.2 / 300


def curvature_radius(fit: np.ndarray, y: float) -> float:
    # Convert x = A*y**2 + B*y + C from pixels to meters, then use
    # R = (1 + (2*A*y + B)**2)**1.5 / |2*A|.
    a = fit[0] * XM_PER_PIX / YM_PER_PIX**2
    b = fit[1] * XM_PER_PIX / YM_PER_PIX
    y_m = y * YM_PER_PIX
    if a == 0:
        return float("inf")
    return float((1 + (2 * a * y_m + b) ** 2) ** 1.5 / abs(2 * a))


def lane_curvature(lane: LaneFit, image_height: int) -> float:
    # Measured at the bottom of the image, closest to the car. Both lines
    # share A and B, so they have the same radius at the same y.
    return curvature_radius(lane.left_fit, image_height - 1)


def vehicle_offset(
    lane: LaneFit, image_size: tuple[int, int], matrix: np.ndarray
) -> float:
    # The camera is assumed to be in the middle of the car, so the car is at
    # the bottom center of the original image. That point is mapped to the
    # bird's-eye view, where the lane lines are measured.
    # Positive: the car is to the right of the lane center.
    width, height = image_size
    car = cv2.perspectiveTransform(
        np.float32([[[width / 2, height - 1]]]), matrix
    )[0, 0]
    lane_center = (
        np.polyval(lane.left_fit, car[1]) + np.polyval(lane.right_fit, car[1])
    ) / 2
    return float((car[0] - lane_center) * XM_PER_PIX)
