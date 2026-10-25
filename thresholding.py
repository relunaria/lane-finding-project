"""Step 2: color and gradient thresholds -> binary image."""

import cv2
import numpy as np


def create_yellow_mask(image: np.ndarray) -> np.ndarray:
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    return cv2.inRange(hls, (15, 50, 90), (40, 255, 255))


def create_road_yellow_mask(image: np.ndarray) -> np.ndarray:
    # Yellow markings on the road only: trees and grass are cut off.
    region = create_region_mask((image.shape[1], image.shape[0]))
    return cv2.bitwise_and(create_yellow_mask(image), region)


def create_color_binary(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    yellow = create_yellow_mask(image)

    height, width = image.shape[:2]
    road_sample = hsv[
        int(height * 0.60):int(height * 0.90),
        int(width * 0.25):int(width * 0.75),
        2,
    ]
    white_min_value = min(
        255,
        max(160, int(np.percentile(road_sample, 97)) + 5),
    )
    white = cv2.inRange(hsv, (0, 0, white_min_value), (179, 60, 255))

    # Retain thin bright markings, suppressing broad bright regions.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    local_contrast = cv2.morphologyEx(
        hsv[:, :, 2], cv2.MORPH_TOPHAT, kernel
    )
    white = cv2.bitwise_and(
        white,
        cv2.inRange(local_contrast, 20, 255),
    )

    return cv2.bitwise_or(yellow, white)


def scale_to_uint8(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    if maximum == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.uint8(255 * values / maximum)


def create_gradient_binary(
    image: np.ndarray,
    sobel_kernel: int = 5,
    min_x_gradient: int = 30,
    direction_threshold: tuple[float, float] = (0.6, 1.3),
    min_local_contrast: int = 10,
) -> np.ndarray:
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    lightness = cv2.GaussianBlur(hls[:, :, 1], (5, 5), 0)

    sobel_x = cv2.Sobel(lightness, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(lightness, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_x = np.absolute(sobel_x)
    abs_y = np.absolute(sobel_y)

    # Strong horizontal intensity changes: lane edges are steep in the image.
    scaled_x = scale_to_uint8(abs_x)
    x_mask = scaled_x >= min_x_gradient

    # Keep diagonal edges typical for lane lines in perspective,
    # rejecting near-horizontal and near-vertical edges.
    direction = np.arctan2(abs_y, abs_x)
    direction_mask = (
        (direction >= direction_threshold[0])
        & (direction <= direction_threshold[1])
    )

    # Keep only edges of thin structures brighter than their surroundings.
    # This rejects single edges such as the road/barrier boundary or
    # shadow borders, which are not lane markings.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    local_contrast = cv2.morphologyEx(lightness, cv2.MORPH_TOPHAT, kernel)
    ridge_mask = local_contrast >= min_local_contrast

    binary = np.zeros_like(lightness, dtype=np.uint8)
    binary[x_mask & direction_mask & ridge_mask] = 255
    return binary


def create_region_mask(
    image_size: tuple[int, int],
    bottom: float = 0.92,
    top: float = 0.62,
    bottom_margin: float = 0.05,
    top_left: float = 0.44,
    top_right: float = 0.58,
) -> np.ndarray:
    # Trapezoid covering the road ahead; excludes sky, roadside and the hood.
    width, height = image_size
    vertices = np.array(
        [[
            (int(width * bottom_margin), int(height * bottom)),
            (int(width * top_left), int(height * top)),
            (int(width * top_right), int(height * top)),
            (int(width * (1 - bottom_margin)), int(height * bottom)),
        ]],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, vertices, 255)
    return mask


def remove_small_components(
    binary: np.ndarray, min_area: int = 30
) -> np.ndarray:
    # Drop isolated specks (road texture, debris) smaller than min_area.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def create_thresholded_binary(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the combined binary mask and its color and gradient parts."""
    region = create_region_mask((image.shape[1], image.shape[0]))
    color = cv2.bitwise_and(create_color_binary(image), region)
    gradient = cv2.bitwise_and(create_gradient_binary(image), region)
    combined = remove_small_components(cv2.bitwise_or(color, gradient))
    return combined, color, gradient
