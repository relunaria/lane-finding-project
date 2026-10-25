"""Steps 1-3 for one frame, and the setup computed once"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from camera_calibration import undistort_image
from perspective import (
    estimate_source_points,
    get_destination_points,
    warp_image,
)
from thresholding import create_road_yellow_mask, create_thresholded_binary

PROJECT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = PROJECT_DIR / "test_images"


@dataclass(frozen=True)
class Setup:
    camera_matrix: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]
    source: np.ndarray
    destination: np.ndarray
    matrix: np.ndarray
    inverse_matrix: np.ndarray

    def with_source(self, source: np.ndarray) -> "Setup":
        return Setup(
            camera_matrix=self.camera_matrix,
            distortion=self.distortion,
            image_size=self.image_size,
            source=source,
            destination=self.destination,
            matrix=cv2.getPerspectiveTransform(source, self.destination),
            inverse_matrix=cv2.getPerspectiveTransform(
                self.destination, source
            ),
        )


@dataclass
class Frame:
    image: np.ndarray
    corrected: np.ndarray
    binary: np.ndarray
    color: np.ndarray
    gradient: np.ndarray
    warped_binary: np.ndarray
    warped_yellow: np.ndarray


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to save: {path}")


def load_setup() -> Setup:
    # Step 1: camera parameters
    with np.load(PROJECT_DIR / "camera_parameters.npz") as parameters:
        camera_matrix = parameters["camera_matrix"]
        distortion = parameters["distortion_coefficients"]
        image_size = tuple(int(v) for v in parameters["image_size"])

    # Step 3: perspective from the straight-road images
    straight_paths = sorted(IMAGES_DIR.glob("straight_lines*.jpg"))
    if not straight_paths:
        raise FileNotFoundError(f"No straight_lines*.jpg in {IMAGES_DIR}")
    straight_binaries = [
        create_thresholded_binary(
            undistort_image(load_image(path), camera_matrix, distortion)
        )[0]
        for path in straight_paths
    ]
    source = estimate_source_points(straight_binaries)
    destination = get_destination_points(image_size)
    return Setup(
        camera_matrix=camera_matrix,
        distortion=distortion,
        image_size=image_size,
        source=source,
        destination=destination,
        matrix=cv2.getPerspectiveTransform(source, destination),
        inverse_matrix=cv2.getPerspectiveTransform(destination, source),
    )


def prepare_frame(image: np.ndarray, setup: Setup) -> Frame:
    if (image.shape[1], image.shape[0]) != setup.image_size:
        image = cv2.resize(image, setup.image_size)

    # Step 1: distortion correction
    corrected = undistort_image(image, setup.camera_matrix, setup.distortion)
    # Step 2: binary image
    binary, color, gradient = create_thresholded_binary(corrected)
    # Step 3: bird's-eye view
    warped_binary = warp_image(binary, setup.matrix, cv2.INTER_NEAREST)
    warped_yellow = warp_image(
        create_road_yellow_mask(corrected), setup.matrix, cv2.INTER_NEAREST
    )
    return Frame(
        image, corrected, binary, color, gradient, warped_binary, warped_yellow
    )
