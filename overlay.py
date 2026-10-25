"""Step 6: drawing the lane back onto the road"""

import cv2
import numpy as np

from lane_detection import LaneFit

STRAIGHT_RADIUS = 5000


def draw_lane_area(
    image: np.ndarray,
    lane: LaneFit,
    inverse_matrix: np.ndarray,
    top_y: int = 0,
) -> np.ndarray:
    height, width = image.shape[:2]
    if top_y >= height - 1:
        return image.copy()
    plot_y = np.arange(top_y, height, dtype=np.float64)
    left = np.column_stack([np.polyval(lane.left_fit, plot_y), plot_y])
    right = np.column_stack([np.polyval(lane.right_fit, plot_y), plot_y])
    polygon = np.vstack([left, right[::-1]]).astype(np.int32)

    warped_overlay = np.zeros_like(image)
    cv2.fillPoly(warped_overlay, [polygon], (0, 255, 0))
    for line, color in ((left, (0, 0, 255)), (right, (255, 0, 0))):
        cv2.polylines(
            warped_overlay, [line.astype(np.int32)], False, color, 20
        )

    overlay = cv2.warpPerspective(
        warped_overlay, inverse_matrix, (width, height)
    )
    return cv2.addWeighted(image, 1.0, overlay, 0.3, 0)


def draw_measurements(
    image: np.ndarray, radius: float, offset: float
) -> np.ndarray:
    if radius > STRAIGHT_RADIUS:
        radius_text = "Radius of curvature: straight"
    else:
        radius_text = f"Radius of curvature: {radius:.0f} m"
    side = "right" if offset > 0 else "left"
    offset_text = f"Vehicle is {abs(offset):.2f} m {side} of center"

    result = image.copy()
    for row, text in enumerate((radius_text, offset_text)):
        for color, thickness in (((0, 0, 0), 8), ((255, 255, 255), 3)):
            cv2.putText(
                result,
                text,
                (40, 60 + row * 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                color,
                thickness,
                cv2.LINE_AA,
            )
    return result


def draw_lane_not_detected(image: np.ndarray) -> np.ndarray:
    result = image.copy()
    cv2.putText(
        result,
        "Lane not detected",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )
    return result
