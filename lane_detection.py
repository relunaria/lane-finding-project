"""Step 4: find lane-line pixels and fit polynomials (bird's-eye view)."""

from dataclasses import dataclass

import numpy as np

Window = tuple[int, int, int, int]


@dataclass
class LanePixels:
    xs: np.ndarray
    ys: np.ndarray


@dataclass
class LaneFit:
    # Polynomials x = A * y**2 + B * y + C in warped-image pixels.
    left_fit: np.ndarray
    right_fit: np.ndarray
    left_pixels: LanePixels
    right_pixels: LanePixels
    # Search windows (x1, y1, x2, y2), drawn only for visualization.
    windows: list[Window]


def find_line_bases(
    warped_binary: np.ndarray,
    warped_yellow: np.ndarray | None = None,
    min_yellow_pixels: int = 500,
) -> tuple[int, int]:
    # Lane lines are nearly vertical near the car, so the column histogram
    # of the bottom half peaks at their x-positions.
    height, width = warped_binary.shape[:2]
    histogram = np.sum(warped_binary[height // 2:] > 0, axis=0)
    midpoint = width // 2
    left_base = int(np.argmax(histogram[:midpoint]))
    right_base = int(np.argmax(histogram[midpoint:])) + midpoint

    # The yellow center line is the left border of the lane. If it is
    # clearly visible, it is a more reliable start than the brightest
    # column, which on curvy roads can be the road edge or a shadow.
    if warped_yellow is not None:
        yellow = np.sum(warped_yellow[height // 2:, :midpoint] > 0, axis=0)
        if yellow.sum() >= min_yellow_pixels:
            left_base = int(np.argmax(yellow))
    return left_base, right_base


def find_pixels_sliding_window(
    warped_binary: np.ndarray,
    warped_yellow: np.ndarray | None = None,
    window_count: int = 9,
    margin: int = 100,
    min_pixels_to_recenter: int = 50,
) -> tuple[LanePixels, LanePixels, list[Window]]:
    height = warped_binary.shape[0]
    window_height = height // window_count
    ys, xs = np.nonzero(warped_binary)

    centers = list(find_line_bases(warped_binary, warped_yellow))
    selected: list[list[np.ndarray]] = [[], []]
    windows = []

    # Move windows upward from the bottom; each window follows the mean x
    # of the pixels found in it, so the search follows curved lines.
    for index in range(window_count):
        y_high = height - index * window_height
        y_low = y_high - window_height
        for side in range(2):
            x_low = centers[side] - margin
            x_high = centers[side] + margin
            windows.append((x_low, y_low, x_high, y_high))
            inside = np.nonzero(
                (ys >= y_low) & (ys < y_high)
                & (xs >= x_low) & (xs < x_high)
            )[0]
            selected[side].append(inside)
            if len(inside) >= min_pixels_to_recenter:
                centers[side] = int(np.mean(xs[inside]))

    left, right = (np.concatenate(indices) for indices in selected)
    return (
        LanePixels(xs[left], ys[left]),
        LanePixels(xs[right], ys[right]),
        windows,
    )


def find_pixels_around_fit(
    warped_binary: np.ndarray,
    left_fit: np.ndarray,
    right_fit: np.ndarray,
    margin: int = 80,
) -> tuple[LanePixels, LanePixels]:
    # For video: search only near the lines found in the previous frame.
    ys, xs = np.nonzero(warped_binary)
    pixels = []
    for fit in (left_fit, right_fit):
        inside = np.abs(xs - np.polyval(fit, ys)) < margin
        pixels.append(LanePixels(xs[inside], ys[inside]))
    return pixels[0], pixels[1]


def fit_parallel_lines(
    left: LanePixels, right: LanePixels, min_pixels: int = 200
) -> tuple[np.ndarray, np.ndarray] | None:
    # Fit x = A * y**2 + B * y + C for both lines at once, with A and B
    # shared and a separate offset C per line. Lane lines are parallel in
    # the bird's-eye view, so the solid line stabilizes the shape of the
    # dashed one. Each side is weighted equally regardless of pixel count.
    if len(left.xs) < min_pixels or len(right.xs) < min_pixels:
        return None

    ys = np.concatenate([left.ys, right.ys]).astype(np.float64)
    xs = np.concatenate([left.xs, right.xs]).astype(np.float64)
    is_left = np.concatenate(
        [np.ones(len(left.xs)), np.zeros(len(right.xs))]
    )
    design = np.column_stack([ys**2, ys, is_left, 1 - is_left])
    weights = np.sqrt(
        np.where(is_left == 1, 1 / len(left.xs), 1 / len(right.xs))
    )

    coefficients, *_ = np.linalg.lstsq(
        design * weights[:, None], xs * weights, rcond=None
    )
    a, b, left_c, right_c = coefficients
    return np.array([a, b, left_c]), np.array([a, b, right_c])


def find_lane(
    warped_binary: np.ndarray,
    previous: LaneFit | None = None,
    warped_yellow: np.ndarray | None = None,
) -> LaneFit | None:
    """Find lane pixels and fit both lines; None if a line is missing."""
    windows: list[Window] = []
    if previous is not None:
        left_pixels, right_pixels = find_pixels_around_fit(
            warped_binary, previous.left_fit, previous.right_fit
        )
    else:
        left_pixels, right_pixels, windows = find_pixels_sliding_window(
            warped_binary, warped_yellow
        )

    fits = fit_parallel_lines(left_pixels, right_pixels)
    if fits is None:
        return None
    return LaneFit(fits[0], fits[1], left_pixels, right_pixels, windows)
