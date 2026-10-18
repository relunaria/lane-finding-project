import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def natural_sort_key(path: Path) -> list[tuple[int, int | str]]:
    return [
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", path.name)
    ]


def create_object_points(pattern_size: tuple[int, int]) -> np.ndarray:
    columns, rows = pattern_size
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    return points


def find_chessboard_corners(
    image: np.ndarray,
    min_pattern_size: tuple[int, int],
    max_pattern_size: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int]] | None:
    if any(
        minimum < 3 or minimum > maximum
        for minimum, maximum in zip(
            min_pattern_size, max_pattern_size, strict=True
        )
    ):
        raise ValueError("Invalid pattern size range")

    candidates = sorted(
        (
            (columns, rows)
            for columns in range(min_pattern_size[0], max_pattern_size[0] + 1)
            for rows in range(min_pattern_size[1], max_pattern_size[1] + 1)
        ),
        key=lambda size: size[0] * size[1],
        reverse=True,
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Try the fast search first; use the exhaustive search only if it fails.
    for flags in (0, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY):
        for pattern_size in candidates:
            found, corners = cv2.findChessboardCornersSB(
                gray, pattern_size, flags=flags
            )
            if not found or corners is None:
                continue

            if flags == 0:
                refined, accurate_corners = cv2.findChessboardCornersSB(
                    gray, pattern_size, flags=cv2.CALIB_CB_ACCURACY
                )
                if refined and accurate_corners is not None:
                    corners = accurate_corners

            return corners, pattern_size

    return None


def collect_calibration_points(
    image_paths: list[Path],
    min_pattern_size: tuple[int, int],
    max_pattern_size: tuple[int, int],
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[tuple[int, int]],
    list[Path],
]:
    object_points_list: list[np.ndarray] = []
    image_points_list: list[np.ndarray] = []
    image_sizes: list[tuple[int, int]] = []
    used_image_paths: list[Path] = []

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Failed to load: {path.name}")
            continue

        result = find_chessboard_corners(
            image, min_pattern_size, max_pattern_size
        )
        if result is None:
            print(f"Corners not found: {path.name}")
            continue

        corners, pattern_size = result
        object_points = create_object_points(pattern_size)
        if len(corners) != len(object_points) or not np.isfinite(corners).all():
            raise ValueError(f"Invalid calibration points: {path.name}")

        object_points_list.append(object_points)
        image_points_list.append(corners)
        image_sizes.append((image.shape[1], image.shape[0]))
        used_image_paths.append(path)
        print(f"{path.name}: {pattern_size}, {len(corners)} corners")

    if not image_points_list:
        raise ValueError("No calibration points found")

    return object_points_list, image_points_list, image_sizes, used_image_paths


def scale_image_points(
    image_points_list: list[np.ndarray],
    image_sizes: list[tuple[int, int]],
    target_size: tuple[int, int],
) -> list[np.ndarray]:
    scaled_points = []
    for corners, (width, height) in zip(
        image_points_list, image_sizes, strict=True
    ):
        scale = np.array(
            [target_size[0] / width, target_size[1] / height],
            dtype=np.float32,
        )
        scaled_points.append((corners * scale).astype(np.float32))
    return scaled_points


def calibrate_camera(
    object_points_list: list[np.ndarray],
    image_points_list: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    rms, camera_matrix, distortion, _, _, _, _, per_view_errors = (
        cv2.calibrateCameraExtended(
            object_points_list, image_points_list, image_size, None, None
        )
    )
    return rms, camera_matrix, distortion, per_view_errors


def undistort_image(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
) -> np.ndarray:
    return cv2.undistort(image, camera_matrix, distortion_coefficients)


def save_undistortion_comparison(
    image_path: Path,
    output_path: Path,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    corrected = undistort_image(image, camera_matrix, distortion_coefficients)
    comparison = cv2.hconcat([image, corrected])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), comparison):
        raise OSError(f"Failed to save: {output_path}")


def save_camera_parameters(
    output_path: Path,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    image_size: tuple[int, int],
) -> None:
    np.savez(
        output_path,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
        image_size=np.asarray(image_size, dtype=np.int32),
    )


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    calibration_dir = project_dir / "camera_cal"
    image_paths = sorted(calibration_dir.glob("*.jpg"), key=natural_sort_key)
    if not image_paths:
        raise FileNotFoundError(calibration_dir)

    object_points, image_points, image_sizes, used_paths = (
        collect_calibration_points(image_paths, (4, 4), (10, 10))
    )
    print(f"Successful detections: {len(used_paths)}/{len(image_paths)}")

    reference_size = Counter(image_sizes).most_common(1)[0][0]
    scaled_points = scale_image_points(image_points, image_sizes, reference_size)
    rms, camera_matrix, distortion, errors = calibrate_camera(
        object_points, scaled_points, reference_size
    )

    parameters_path = project_dir / "camera_parameters.npz"
    save_camera_parameters(
        parameters_path, camera_matrix, distortion, reference_size
    )
    print(f"\nRMS error: {rms:.4f} px")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients:\n{distortion}")
    print("Per-image errors:")
    for path, error in zip(used_paths, errors.ravel(), strict=True):
        print(f"{path.name}: {error:.4f} px")
    print(f"Saved camera parameters: {parameters_path}")

    reference_image = next(
        path for path, size in zip(used_paths, image_sizes, strict=True)
        if size == reference_size
    )
    comparison_path = project_dir / "docs" / "undistortion_comparison.jpg"
    save_undistortion_comparison(
        reference_image, comparison_path, camera_matrix, distortion
    )
    print(f"Saved comparison: {comparison_path}")


if __name__ == "__main__":
    main()
