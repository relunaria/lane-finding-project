"""Video only: checks and tracking of the lane between frames"""

from collections import deque

import numpy as np

from curvature import XM_PER_PIX, lane_curvature, vehicle_offset
from lane_detection import LaneFit, find_lane
from pipeline import Setup

MIN_LANE_WIDTH_M = 3.0
MAX_LANE_WIDTH_M = 4.5
MIN_RADIUS_M = 30
MAX_OFFSET_M = 1.0
MAX_JUMP_M = 0.5
JUMP_PER_MISSED_FRAME_M = 0.05
MAX_CURVATURE_CHANGE = 0.005
MAX_YELLOW_INSIDE_PX = 1000
YELLOW_MARGIN = 0.25
YELLOW_MIN_Y_RATIO = 0.5


def yellow_inside(
    lane: LaneFit, warped_yellow: np.ndarray, y_low: int, y_high: int
) -> int:
    ys, xs = np.nonzero(warped_yellow[y_low:y_high])
    ys = ys + y_low
    left = np.polyval(lane.left_fit, ys)
    right = np.polyval(lane.right_fit, ys)
    margin = YELLOW_MARGIN * (right - left)
    return int(((xs > left + margin) & (xs < right - margin)).sum())


def find_clear_top(
    lane: LaneFit, warped_yellow: np.ndarray, band: int = 40
) -> int:
    height = warped_yellow.shape[0]
    max_per_band = MAX_YELLOW_INSIDE_PX * band / height
    for y_high in range(height, band - 1, -band):
        if yellow_inside(lane, warped_yellow, y_high - band, y_high) > (
            max_per_band
        ):
            return y_high
    return 0


def is_plausible(
    lane: LaneFit, setup: Setup, warped_yellow: np.ndarray
) -> bool:
    height = setup.image_size[1]
    y = height - 1
    width_px = np.polyval(lane.right_fit, y) - np.polyval(lane.left_fit, y)
    width_m = width_px * XM_PER_PIX
    radius = lane_curvature(lane, height)
    offset = vehicle_offset(lane, setup.image_size, setup.matrix)
    yellow = yellow_inside(
        lane, warped_yellow, int(height * YELLOW_MIN_Y_RATIO), height
    )
    return (
        MIN_LANE_WIDTH_M <= width_m <= MAX_LANE_WIDTH_M
        and radius >= MIN_RADIUS_M
        and abs(offset) <= MAX_OFFSET_M
        and yellow <= MAX_YELLOW_INSIDE_PX
    )


def signed_curvature(lane: LaneFit, image_height: int) -> float:
    radius = lane_curvature(lane, image_height)
    return float(np.sign(lane.left_fit[0]) / radius)


def is_close(
    lane: LaneFit, previous: LaneFit, image_height: int, max_shift_m: float
) -> bool:
    curvature_change = abs(
        signed_curvature(lane, image_height)
        - signed_curvature(previous, image_height)
    )
    if curvature_change > MAX_CURVATURE_CHANGE:
        return False
    for y in (image_height - 1, image_height // 2):
        for fit, previous_fit in (
            (lane.left_fit, previous.left_fit),
            (lane.right_fit, previous.right_fit),
        ):
            shift_px = np.polyval(fit, y) - np.polyval(previous_fit, y)
            if abs(shift_px) * XM_PER_PIX > max_shift_m:
                return False
    return True


class LaneTracker:
    def __init__(
        self,
        setup: Setup,
        history: int = 5,
        max_missed: int = 10,
        radius_history: int = 30,
    ) -> None:
        self.setup = setup
        self.fits: deque[tuple[np.ndarray, np.ndarray]] = deque(
            maxlen=history
        )
        self.last_lane: LaneFit | None = None
        self.smoothed: LaneFit | None = None
        self.missed = 0
        self.max_missed = max_missed
        self.curvatures: deque[float] = deque(maxlen=radius_history)

    def update(
        self, warped_binary: np.ndarray, warped_yellow: np.ndarray
    ) -> tuple[LaneFit | None, str]:
        lane, status = None, "lost"
        if self.last_lane is not None:
            lane = find_lane(warped_binary, self.last_lane)
            if self.accepts(lane, warped_yellow):
                status = "tracked"
            else:
                lane = None
        if lane is None:
            lane = find_lane(warped_binary, warped_yellow=warped_yellow)
            if self.accepts(lane, warped_yellow):
                status = "searched"
            else:
                lane = None

        if lane is not None:
            self.fits.append((lane.left_fit, lane.right_fit))
            self.last_lane = lane
            self.missed = 0
        else:
            self.missed += 1
            if self.missed > self.max_missed:
                self.reset()
            elif self.fits:
                status = "reused"

        if not self.fits:
            return None, "lost"
        left = np.mean([fit[0] for fit in self.fits], axis=0)
        right = np.mean([fit[1] for fit in self.fits], axis=0)
        latest = lane if lane is not None else self.last_lane
        self.smoothed = LaneFit(
            left, right, latest.left_pixels, latest.right_pixels, []
        )
        self.curvatures.append(
            signed_curvature(self.smoothed, warped_binary.shape[0])
        )
        return self.smoothed, status

    def reset(self) -> None:
        self.fits.clear()
        self.curvatures.clear()
        self.last_lane = None
        self.smoothed = None

    def radius(self) -> float:
        mean_curvature = float(np.mean(self.curvatures))
        if mean_curvature == 0:
            return float("inf")
        return 1 / abs(mean_curvature)

    def accepts(
        self, lane: LaneFit | None, warped_yellow: np.ndarray
    ) -> bool:
        if lane is None or not is_plausible(lane, self.setup, warped_yellow):
            return False
        if self.smoothed is None:
            return True
        max_shift = MAX_JUMP_M + JUMP_PER_MISSED_FRAME_M * self.missed
        height = self.setup.image_size[1]
        return is_close(lane, self.smoothed, height, max_shift)
