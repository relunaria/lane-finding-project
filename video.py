"""Runs the pipeline on videos, with tracking between frames.

Usage:
    python3 video.py                                   # all test_videos/*.mp4
    python3 video.py test_videos/project_video01.mp4   # selected videos
Results are saved to output_videos/<name>_output.mp4.
"""

import sys
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from curvature import vehicle_offset
from overlay import draw_lane_area, draw_lane_not_detected, draw_measurements
from perspective import fit_lane_lines, trapezoid_from_lines, vanishing_point
from pipeline import PROJECT_DIR, Setup, load_setup, prepare_frame
from tracking import LaneTracker, find_clear_top

VIDEOS_DIR = PROJECT_DIR / "test_videos"
OUTPUT_DIR = PROJECT_DIR / "output_videos"


def read_frames(path: Path, step: int = 1) -> Iterator[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            yield frame
        index += 1
    capture.release()


def setup_vanishing_y(source: np.ndarray) -> float:
    # Where the sides of the source trapezoid meet.
    lines = []
    for bottom, top in ((source[3], source[0]), (source[2], source[1])):
        slope = (top[0] - bottom[0]) / (top[1] - bottom[1])
        lines.append((slope, bottom[0] - slope * bottom[1]))
    return vanishing_point(tuple(lines))[1]


def adapt_perspective(
    path: Path, setup: Setup, sample_count: int = 40
) -> Setup:
    """Re-estimate the perspective if the camera is mounted differently.

    The perspective from straight_lines*.jpg fits the project videos. In
    other videos the camera can be tilted or mounted at another height,
    which moves the horizon. Straight lines are fitted to the lane pixels
    in the lower third of sampled frames; their median vanishing point
    shows where the horizon is. If it is more than 5 px away from the one
    in the image setup, or the lane at the bottom of the image is more than
    10% wider or narrower, a new trapezoid is built from the median line
    positions.
    """
    width, height = setup.image_size
    total = int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // sample_count)

    all_lines = []
    for image in read_frames(path, step):
        binary = prepare_frame(image, setup).binary
        lines = fit_lane_lines(binary, min_y=height * 2 // 3)
        if lines is not None and vanishing_point(lines) is not None:
            all_lines.append(lines)
    if len(all_lines) < 5:
        return setup

    video_vp_y = float(
        np.median([vanishing_point(lines)[1] for lines in all_lines])
    )
    setup_vp_y = setup_vanishing_y(setup.source)
    # Keep the same distance between the horizon and the trapezoid top.
    top_y = video_vp_y + (setup.source[0][1] - setup_vp_y)
    source = np.float32(np.median(
        [trapezoid_from_lines(lines, top_y, height) for lines in all_lines],
        axis=0,
    ))
    width_ratio = (source[2][0] - source[3][0]) / (
        setup.source[2][0] - setup.source[3][0]
    )

    print(
        f"{path.name}: horizon at y={video_vp_y:.0f} "
        f"(images: y={setup_vp_y:.0f}), "
        f"lane width {width_ratio:.0%} of the images"
    )
    if abs(video_vp_y - setup_vp_y) <= 5 and abs(width_ratio - 1) <= 0.1:
        return setup
    print(f"{path.name}: using a video-specific perspective")
    return setup.with_source(source)


def process_frame(
    image: np.ndarray, setup: Setup, tracker: LaneTracker
) -> tuple[np.ndarray, str]:
    frame = prepare_frame(image, setup)  # steps 1-3
    lane, status = tracker.update(  # step 4 with tracking
        frame.warped_binary, frame.warped_yellow
    )
    if lane is None:
        return draw_lane_not_detected(frame.corrected), status

    # Steps 5 and 6.
    offset = vehicle_offset(lane, setup.image_size, setup.matrix)
    top_y = find_clear_top(lane, frame.warped_yellow)
    result = draw_lane_area(
        frame.corrected, lane, setup.inverse_matrix, top_y
    )
    return draw_measurements(result, tracker.radius(), offset), status


def open_writer(
    path: Path, fps: float, size: tuple[int, int]
) -> cv2.VideoWriter:
    # H.264 (avc1) plays in QuickTime, browsers and on GitHub, but is not
    # available in every OpenCV build; mp4v is the fallback.
    path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec), fps, size
        )
        if writer.isOpened():
            print(f"{path.name}: writing with codec {codec}")
            return writer
    raise OSError(f"Cannot open video writer for {path}")


def process_video(path: Path, setup: Setup) -> None:
    setup = adapt_perspective(path, setup)
    fps = cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FPS) or 25
    output_path = OUTPUT_DIR / f"{path.stem}_output.mp4"
    writer = open_writer(output_path, fps, setup.image_size)

    tracker = LaneTracker(setup)
    counts = {"tracked": 0, "searched": 0, "reused": 0, "lost": 0}
    for image in read_frames(path):
        result, status = process_frame(image, setup, tracker)
        counts[status] += 1
        writer.write(result)
    writer.release()

    summary = ", ".join(f"{name} {count}" for name, count in counts.items())
    print(f"{path.name}: {sum(counts.values())} frames ({summary})")
    print(f"Saved: {output_path}")


def main() -> None:
    if len(sys.argv) > 1:
        paths = [Path(argument) for argument in sys.argv[1:]]
    else:
        paths = sorted(VIDEOS_DIR.glob("*.mp4"))
    if not paths:
        raise FileNotFoundError(f"No videos in {VIDEOS_DIR}")

    setup = load_setup()
    for path in paths:
        process_video(path, setup)


if __name__ == "__main__":
    main()
