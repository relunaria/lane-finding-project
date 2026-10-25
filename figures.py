"""Figures for the README"""

import cv2
import numpy as np

from lane_detection import LaneFit
from pipeline import Frame, Setup


def to_bgr(binary: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def binary_row(frame: Frame) -> list[np.ndarray]:
    # Step 2
    contribution = cv2.merge([frame.gradient, frame.color, frame.gradient])
    return [frame.corrected, to_bgr(frame.binary), contribution]


def perspective_row(frame: Frame, setup: Setup) -> list[np.ndarray]:
    # Step 3
    source_view = frame.corrected.copy()
    cv2.polylines(
        source_view, [setup.source.astype(np.int32)], True, (0, 0, 255), 3
    )
    height, width = frame.corrected.shape[:2]
    warped_view = cv2.warpPerspective(
        frame.corrected, setup.matrix, (width, height)
    )
    cv2.polylines(
        warped_view,
        [setup.destination.astype(np.int32)],
        True,
        (0, 0, 255),
        3,
    )
    return [source_view, warped_view, to_bgr(frame.warped_binary)]


def lane_fit_row(frame: Frame, lane: LaneFit | None) -> list[np.ndarray]:
    # Step 4
    preview = to_bgr(frame.warped_binary // 3)
    if lane is not None:
        for x1, y1, x2, y2 in lane.windows:
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        preview[lane.left_pixels.ys, lane.left_pixels.xs] = (0, 0, 255)
        preview[lane.right_pixels.ys, lane.right_pixels.xs] = (255, 0, 0)
        plot_y = np.arange(preview.shape[0], dtype=np.float64)
        for fit in (lane.left_fit, lane.right_fit):
            points = np.column_stack([np.polyval(fit, plot_y), plot_y])
            cv2.polylines(
                preview, [points.astype(np.int32)], False, (0, 255, 255), 4
            )
    return [frame.corrected, preview]


def stack_rows(
    rows: list[list[np.ndarray]], tile_size: tuple[int, int] = (640, 360)
) -> np.ndarray:
    columns = max(len(row) for row in rows)
    blank = np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
    return cv2.vconcat([
        cv2.hconcat(
            [cv2.resize(tile, tile_size) for tile in row]
            + [blank] * (columns - len(row))
        )
        for row in rows
    ])
