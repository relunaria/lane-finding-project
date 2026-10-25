"""Step 3: perspective transform to the bird's-eye view."""

import cv2
import numpy as np


def fit_straight_line(
    xs: np.ndarray, ys: np.ndarray, keep_fraction: float = 0.8
) -> tuple[float, float]:
    # Fit x = slope * y + intercept, then refit without the worst outliers.
    slope, intercept = np.polyfit(ys, xs, 1)
    residuals = np.abs(xs - (slope * ys + intercept))
    inliers = residuals <= np.quantile(residuals, keep_fraction)
    slope, intercept = np.polyfit(ys[inliers], xs[inliers], 1)
    return float(slope), float(intercept)


Line = tuple[float, float]  # x = slope * y + intercept


def fit_lane_lines(
    binary: np.ndarray, min_y: int = 0, min_pixels: int = 100
) -> tuple[Line, Line] | None:
    # Straight lines through the left and right lane pixels below min_y.
    ys, xs = np.nonzero(binary[min_y:])
    ys = ys + min_y
    width = binary.shape[1]
    lines = []
    for side in (xs < width // 2, xs >= width // 2):
        if side.sum() < min_pixels:
            return None
        lines.append(fit_straight_line(xs[side], ys[side]))
    return lines[0], lines[1]


def vanishing_point(lines: tuple[Line, Line]) -> tuple[float, float] | None:
    (left_slope, left_c), (right_slope, right_c) = lines
    if abs(left_slope - right_slope) < 1e-3:
        return None
    y = (right_c - left_c) / (left_slope - right_slope)
    return left_slope * y + left_c, y


def trapezoid_from_lines(
    lines: tuple[Line, Line], top_y: float, bottom_y: float
) -> np.ndarray:
    (left_slope, left_c), (right_slope, right_c) = lines
    return np.float32([
        (left_slope * top_y + left_c, top_y),
        (right_slope * top_y + right_c, top_y),
        (right_slope * bottom_y + right_c, bottom_y),
        (left_slope * bottom_y + left_c, bottom_y),
    ])


def estimate_source_points(
    binaries: list[np.ndarray], top_ratio: float = 455 / 720
) -> np.ndarray:
    # Fit both lane lines on binary images of a straight road and take
    # their x-positions at the top and bottom of the trapezoid, averaged
    # over all images.
    height = binaries[0].shape[0]
    corners = []
    for binary in binaries:
        lines = fit_lane_lines(binary)
        if lines is None:
            raise ValueError("Too few lane pixels to fit a line")
        corners.append(
            trapezoid_from_lines(lines, height * top_ratio, float(height))
        )
    return np.float32(np.mean(corners, axis=0))


def get_destination_points(image_size: tuple[int, int]) -> np.ndarray:
    # Rectangle in the middle half of the image: lane lines become vertical.
    width, height = image_size
    return np.float32([
        (width * 0.25, 0),
        (width * 0.75, 0),
        (width * 0.75, height),
        (width * 0.25, height),
    ])


def warp_image(
    image: np.ndarray,
    matrix: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.warpPerspective(
        image, matrix, (width, height), flags=interpolation
    )


def measure_lane_widths(
    warped_binary: np.ndarray, bands: int = 3
) -> list[float]:
    # Distance between the median x of left and right lane pixels in
    # horizontal bands (top to bottom). Similar values mean parallel lines.
    height, width = warped_binary.shape[:2]
    widths = []
    for index in range(bands):
        band = warped_binary[
            index * height // bands:(index + 1) * height // bands
        ]
        xs = np.nonzero(band)[1]
        left = xs[xs < width // 2]
        right = xs[xs >= width // 2]
        if len(left) == 0 or len(right) == 0:
            widths.append(float("nan"))
        else:
            widths.append(float(np.median(right) - np.median(left)))
    return widths
