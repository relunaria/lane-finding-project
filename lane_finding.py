"""Runs all steps on test_images/ and saves the figures to docs/"""

import cv2

from curvature import lane_curvature, vehicle_offset
from figures import binary_row, lane_fit_row, perspective_row, stack_rows
from lane_detection import find_lane
from overlay import draw_lane_area, draw_measurements
from perspective import measure_lane_widths
from pipeline import (
    IMAGES_DIR,
    PROJECT_DIR,
    load_image,
    load_setup,
    prepare_frame,
    save_image,
)

DOCS_DIR = PROJECT_DIR / "docs"


def main() -> None:
    setup = load_setup()
    print("Perspective source points:")
    for x, y in setup.source:
        print(f"  ({x:.0f}, {y:.0f})")

    paths = [
        *sorted(IMAGES_DIR.glob("straight_lines*.jpg")),
        *sorted(IMAGES_DIR.glob("test*.jpg")),
        *sorted(IMAGES_DIR.glob("challange*.jpg")),
    ]

    binary_rows, perspective_rows, lane_fit_rows, results = [], [], [], []
    for path in paths:
        # Steps 1-3
        frame = prepare_frame(load_image(path), setup)
        # Step 4
        lane = find_lane(frame.warped_binary)

        result = frame.corrected
        if lane is None:
            print(f"{path.name}: lane lines not found")
        else:
            # Steps 5 and 6
            radius = lane_curvature(lane, setup.image_size[1])
            offset = vehicle_offset(lane, setup.image_size, setup.matrix)
            result = draw_lane_area(
                frame.corrected, lane, setup.inverse_matrix
            )
            result = draw_measurements(result, radius, offset)
            print(
                f"{path.name}: radius {radius:.0f} m, "
                f"offset {offset:+.2f} m"
            )

        # Step 3 check: lane width at the top, middle and bottom
        widths = measure_lane_widths(frame.warped_binary)
        widths_text = ", ".join(f"{width:.0f}" for width in widths)
        print(f"{path.name}: lane width top to bottom {widths_text} px")

        if path.name == "test1.jpg":
            save_image(
                DOCS_DIR / "road_undistortion.jpg",
                cv2.hconcat([frame.image, frame.corrected]),
            )
        binary_rows.append(binary_row(frame))
        perspective_rows.append(perspective_row(frame, setup))
        lane_fit_rows.append(lane_fit_row(frame, lane))
        results.append(result)

    result_rows = [results[i:i + 2] for i in range(0, len(results), 2)]
    save_image(DOCS_DIR / "binary_examples.jpg", stack_rows(binary_rows))
    save_image(
        DOCS_DIR / "perspective_examples.jpg", stack_rows(perspective_rows)
    )
    save_image(DOCS_DIR / "lane_fit_examples.jpg", stack_rows(lane_fit_rows))
    save_image(DOCS_DIR / "result_examples.jpg", stack_rows(result_rows))
    print(f"Saved figures to {DOCS_DIR}")


if __name__ == "__main__":
    main()
