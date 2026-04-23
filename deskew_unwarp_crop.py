#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps

try:
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    from skimage import exposure, feature, filters, measure, transform
except ImportError as exc:
    raise SystemExit("Install numpy, scipy, and scikit-image to run this script.") from exc


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    ".webp",
}
DEFAULT_MAX_DETECTION_DIM = 3000
DEFAULT_MAX_WORKERS = 8
AXIS_TOLERANCE_DEGREES = 10
MAX_SKEW_DEGREES = 15
HORIZONTAL_OUTER_BAND_FRACTION = 0.1
VERTICAL_OUTER_BAND_FRACTION = 0.01
MIN_CLUSTER_FRACTION = 0.9


@dataclass(frozen=True)
class LineSegment:
    p0: np.ndarray
    p1: np.ndarray
    length: float
    angle: float
    x: float
    y: float


@dataclass(frozen=True)
class BoundaryCluster:
    position: float
    total_length: float
    segments: tuple[LineSegment, ...]


@dataclass(frozen=True)
class ProcessingResult:
    image_path: Path
    output_path: Path
    skew_angle: float
    used_unwarp: bool


def iter_image_paths(images_dir: Path, output_dir: Path) -> Iterable[Path]:
    for path in sorted(images_dir.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if output_dir in path.parents:
            continue

        yield path


def normalize_image_mode(image: Image.Image) -> Image.Image:
    if image.mode in {"L", "RGB"}:
        return image

    if image.mode in {"RGBA", "LA"}:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(background, rgba)
        return composited.convert("RGB")

    return image.convert("RGB")


def resize_for_detection(image: Image.Image, max_dim: int) -> Image.Image:
    scale = min(1.0, max_dim / max(image.size))
    if scale >= 1.0:
        return image.copy()

    resized_width = max(1, round(image.width * scale))
    resized_height = max(1, round(image.height * scale))
    return image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)


def weighted_median(values: list[float], weights: list[float]) -> float:
    order = np.argsort(values)
    ordered_values = np.asarray(values, dtype=np.float64)[order]
    ordered_weights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(ordered_weights)
    cutoff = ordered_weights.sum() / 2.0
    return float(ordered_values[np.searchsorted(cumulative, cutoff)])


def normalize_line_angle(angle: float) -> float:
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return angle


def nearest_axis_deviation(angle: float) -> float:
    normalized = normalize_line_angle(angle)
    if abs(normalized) <= 45.0:
        return normalized

    return normalized - math.copysign(90.0, normalized)


def build_edge_map(gray: np.ndarray) -> np.ndarray:
    equalized = exposure.equalize_adapthist(gray, clip_limit=0.03)
    smoothed = filters.gaussian(equalized, sigma=1.2)
    return feature.canny(smoothed, sigma=1.0)


def detect_line_segments(gray: np.ndarray) -> list[LineSegment]:
    edges = build_edge_map(gray)
    line_length = max(40, int(min(gray.shape) * 0.08))
    line_gap = max(10, int(min(gray.shape) * 0.015))
    segments: list[LineSegment] = []

    for p0, p1 in transform.probabilistic_hough_line(
        edges,
        threshold=10,
        line_length=line_length,
        line_gap=line_gap,
    ):
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        angle = normalize_line_angle(math.degrees(math.atan2(dy, dx)))
        segments.append(
            LineSegment(
                p0=np.asarray(p0, dtype=np.float64),
                p1=np.asarray(p1, dtype=np.float64),
                length=math.hypot(dx, dy),
                angle=angle,
                x=(p0[0] + p1[0]) / 2.0,
                y=(p0[1] + p1[1]) / 2.0,
            )
        )

    return segments


def estimate_skew_angle(gray: np.ndarray) -> float:
    deviations: list[float] = []
    lengths: list[float] = []

    for segment in detect_line_segments(gray):
        deviation = nearest_axis_deviation(segment.angle)
        if abs(deviation) > MAX_SKEW_DEGREES:
            continue

        deviations.append(deviation)
        lengths.append(segment.length)

    if not deviations:
        return 0.0

    return weighted_median(deviations, lengths)


def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    fill = 255 if image.mode == "L" else (255, 255, 255)
    return image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=fill,
    )


def cluster_segments(
    segments: list[LineSegment],
    axis: str,
    tolerance: float,
) -> list[BoundaryCluster]:
    if not segments:
        return []

    sorted_segments = sorted(segments, key=lambda segment: getattr(segment, axis))
    current_cluster = [sorted_segments[0]]
    clusters: list[BoundaryCluster] = []

    for segment in sorted_segments[1:]:
        center = np.average(
            [getattr(item, axis) for item in current_cluster],
            weights=[item.length for item in current_cluster],
        )
        if abs(getattr(segment, axis) - center) <= tolerance:
            current_cluster.append(segment)
            continue

        clusters.append(build_boundary_cluster(current_cluster, axis))
        current_cluster = [segment]

    clusters.append(build_boundary_cluster(current_cluster, axis))
    return clusters


def build_boundary_cluster(
    segments: list[LineSegment],
    axis: str,
) -> BoundaryCluster:
    total_length = sum(segment.length for segment in segments)
    position = float(
        np.average(
            [getattr(segment, axis) for segment in segments],
            weights=[segment.length for segment in segments],
        )
    )
    return BoundaryCluster(
        position=position,
        total_length=total_length,
        segments=tuple(segments),
    )


def select_boundary_cluster(
    clusters: list[BoundaryCluster],
    side: str,
    image_extent: int,
) -> BoundaryCluster | None:
    if not clusters:
        return None

    if side in {"top", "bottom"}:
        outer_band = image_extent * HORIZONTAL_OUTER_BAND_FRACTION
    else:
        outer_band = image_extent * VERTICAL_OUTER_BAND_FRACTION

    if side in {"top", "left"}:
        candidates = [cluster for cluster in clusters if cluster.position <= outer_band]
        if not candidates:
            return None
        return min(candidates, key=lambda cluster: cluster.position)

    candidates = [
        cluster
        for cluster in clusters
        if cluster.position >= image_extent - outer_band
    ]
    if not candidates:
        return None

    return max(candidates, key=lambda cluster: cluster.position)


def fit_line(segments: tuple[LineSegment, ...]) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    for segment in segments:
        points.append(segment.p0)
        points.append(segment.p1)

    line_model = measure.LineModelND.from_estimate(np.vstack(points))
    direction = np.asarray(line_model.direction, dtype=np.float64)
    if direction[0] < 0:
        direction *= -1
    return np.asarray(line_model.origin, dtype=np.float64), direction


def line_intersection(
    line_a: tuple[np.ndarray, np.ndarray],
    line_b: tuple[np.ndarray, np.ndarray],
) -> np.ndarray | None:
    origin_a, direction_a = line_a
    origin_b, direction_b = line_b
    matrix = np.array(
        [
            [direction_a[0], -direction_b[0]],
            [direction_a[1], -direction_b[1]],
        ],
        dtype=np.float64,
    )

    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-6:
        return None

    rhs = origin_b - origin_a
    offset_a, _ = np.linalg.solve(matrix, rhs)
    return origin_a + offset_a * direction_a


def polygon_area(corners: np.ndarray) -> float:
    x = corners[:, 0]
    y = corners[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def detect_page_corners(gray: np.ndarray) -> np.ndarray | None:
    height, width = gray.shape
    all_segments = detect_line_segments(gray)
    horizontal_segments = [
        segment
        for segment in all_segments
        if abs(segment.angle) <= AXIS_TOLERANCE_DEGREES
    ]
    vertical_segments = [
        segment
        for segment in all_segments
        if abs(abs(segment.angle) - 90.0) <= AXIS_TOLERANCE_DEGREES
    ]

    horizontal_clusters = [
        cluster
        for cluster in cluster_segments(
            horizontal_segments,
            axis="y",
            tolerance=max(10.0, height * 0.015),
        )
        if cluster.total_length >= width * MIN_CLUSTER_FRACTION
    ]
    vertical_clusters = [
        cluster
        for cluster in cluster_segments(
            vertical_segments,
            axis="x",
            tolerance=max(10.0, width * 0.015),
        )
        if cluster.total_length >= height * MIN_CLUSTER_FRACTION
    ]

    top_cluster = select_boundary_cluster(horizontal_clusters, "top", height)
    bottom_cluster = select_boundary_cluster(horizontal_clusters, "bottom", height)
    left_cluster = select_boundary_cluster(vertical_clusters, "left", width)
    right_cluster = select_boundary_cluster(vertical_clusters, "right", width)
    if not all((top_cluster, bottom_cluster, left_cluster, right_cluster)):
        return None

    top_line = fit_line(top_cluster.segments)
    bottom_line = fit_line(bottom_cluster.segments)
    left_line = fit_line(left_cluster.segments)
    right_line = fit_line(right_cluster.segments)

    corners = [
        line_intersection(top_line, left_line),
        line_intersection(top_line, right_line),
        line_intersection(bottom_line, right_line),
        line_intersection(bottom_line, left_line),
    ]
    if any(corner is None for corner in corners):
        return None

    corners_array = np.vstack(corners)
    if np.any(corners_array[:, 0] < -0.05 * width):
        return None
    if np.any(corners_array[:, 0] > 1.05 * width):
        return None
    if np.any(corners_array[:, 1] < -0.05 * height):
        return None
    if np.any(corners_array[:, 1] > 1.05 * height):
        return None

    if polygon_area(corners_array) < 0.25 * width * height:
        return None

    return corners_array


def warp_channel(
    channel: np.ndarray,
    projective_transform: transform.ProjectiveTransform,
    output_shape: tuple[int, int],
) -> np.ndarray:
    warped = transform.warp(
        channel,
        projective_transform.inverse,
        output_shape=output_shape,
        preserve_range=True,
        mode="edge",
    )
    return np.clip(warped, 0, 255).astype(np.uint8)


def unwarp_image(image: Image.Image, corners: np.ndarray) -> Image.Image:
    top_width = np.linalg.norm(corners[1] - corners[0])
    bottom_width = np.linalg.norm(corners[2] - corners[3])
    left_height = np.linalg.norm(corners[3] - corners[0])
    right_height = np.linalg.norm(corners[2] - corners[1])
    output_width = max(1, round(max(top_width, bottom_width)))
    output_height = max(1, round(max(left_height, right_height)))

    destination = np.array(
        [
            [0.0, 0.0],
            [output_width - 1.0, 0.0],
            [output_width - 1.0, output_height - 1.0],
            [0.0, output_height - 1.0],
        ],
        dtype=np.float64,
    )
    projective_transform = transform.ProjectiveTransform.from_estimate(
        corners,
        destination,
    )
    if projective_transform is None:
        return image.copy()

    image_array = np.asarray(image)
    if image_array.ndim == 2:
        warped = warp_channel(image_array, projective_transform, (output_height, output_width))
        return Image.fromarray(warped, mode="L")

    channels = [
        warp_channel(image_array[..., channel_index], projective_transform, (output_height, output_width))
        for channel_index in range(image_array.shape[2])
    ]
    return Image.fromarray(np.stack(channels, axis=-1), mode="RGB")


def detect_trim_box(gray: np.ndarray) -> tuple[int, int, int, int]:
    height, width = gray.shape
    row_profile = gaussian_filter1d(np.percentile(gray, 85, axis=1), sigma=6)
    col_profile = gaussian_filter1d(np.percentile(gray, 85, axis=0), sigma=6)
    row_gradient = np.gradient(row_profile)
    col_gradient = np.gradient(col_profile)

    row_band = min(max(20, int(height * 0.2)), max(20, height // 2))
    col_band = min(max(20, int(width * 0.2)), max(20, width // 2))

    top = int(np.argmax(row_gradient[:row_band]))
    bottom = int(np.argmin(row_gradient[-row_band:]) + (height - row_band))
    left = int(np.argmax(col_gradient[:col_band]))
    right = int(np.argmin(col_gradient[-col_band:]) + (width - col_band))

    top = max(0, min(top + 2, height - 2))
    bottom = max(top + 1, min(bottom - 2, height - 1))
    left = max(0, min(left + 2, width - 2))
    right = max(left + 1, min(right - 2, width - 1))

    if bottom - top < height * 0.5:
        top = 0
        bottom = height - 1

    if right - left < width * 0.5:
        left = 0
        right = width - 1

    return left, top, right, bottom


def trim_image(image: Image.Image) -> Image.Image:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    left, top, right, bottom = detect_trim_box(gray)
    return image.crop((left, top, right + 1, bottom + 1))


def build_output_path(image_path: Path, images_dir: Path, output_dir: Path) -> Path:
    relative_path = image_path.relative_to(images_dir)
    relative_parent = relative_path.parent
    target_dir = output_dir / relative_parent
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{image_path.stem}_cropped.jpg"


def recommended_worker_count(image_count: int) -> int:
    if image_count <= 1:
        return 1

    # Each worker keeps multiple large arrays alive during detection and warping,
    # so cap concurrency by default to avoid wasting time in memory pressure.
    available_cores = os.cpu_count() or 1
    return max(1, min(DEFAULT_MAX_WORKERS, available_cores, image_count))


def process_image(image_path: Path, output_path: Path, max_detection_dim: int) -> tuple[float, bool]:
    with Image.open(image_path) as image:
        normalized = normalize_image_mode(ImageOps.exif_transpose(image))

    detection_source = resize_for_detection(normalized.convert("L"), max_detection_dim)
    detection_array = np.asarray(detection_source, dtype=np.float32) / 255.0
    skew_angle = estimate_skew_angle(detection_array)

    rotated_image = rotate_image(normalized, -skew_angle)
    rotated_detection = resize_for_detection(rotated_image.convert("L"), max_detection_dim)
    rotated_detection_array = np.asarray(rotated_detection, dtype=np.float32) / 255.0

    used_unwarp = False
    page_corners = detect_page_corners(rotated_detection_array)
    processed_image = rotated_image
    if page_corners is not None:
        scale_x = rotated_image.width / rotated_detection.width
        scale_y = rotated_image.height / rotated_detection.height
        full_resolution_corners = page_corners.copy()
        full_resolution_corners[:, 0] *= scale_x
        full_resolution_corners[:, 1] *= scale_y
        processed_image = unwarp_image(rotated_image, full_resolution_corners)
        used_unwarp = True

    cropped_image = trim_image(processed_image)
    save_kwargs = {"quality": 95}
    cropped_image.save(output_path, **save_kwargs)
    return skew_angle, used_unwarp


def process_image_task(
    image_path: Path,
    images_dir: Path,
    output_dir: Path,
    max_detection_dim: int,
) -> ProcessingResult:
    output_path = build_output_path(image_path, images_dir, output_dir)
    skew_angle, used_unwarp = process_image(
        image_path,
        output_path,
        max_detection_dim=max_detection_dim,
    )
    return ProcessingResult(
        image_path=image_path,
        output_path=output_path,
        skew_angle=skew_angle,
        used_unwarp=used_unwarp,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Deskew book photos, apply a perspective unwarp when possible, "
            "and crop out the page border."
        )
    )
    parser.add_argument(
        "--images-dir",
        default=Path(__file__).resolve().parent / "images",
        type=Path,
        help="Directory to scan for images. Defaults to ./images next to this script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write processed images. Defaults to <images-dir>/cropped.",
    )
    parser.add_argument(
        "--max-detection-dim",
        default=DEFAULT_MAX_DETECTION_DIM,
        type=int,
        help=(
            "Maximum width or height for the downscaled detection image. "
            f"Defaults to {DEFAULT_MAX_DETECTION_DIM}."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        help=(
            "Number of worker threads to use. Defaults to an automatic value "
            f"up to {DEFAULT_MAX_WORKERS}."
        ),
    )
    args = parser.parse_args()

    images_dir = args.images_dir.resolve()
    output_dir = (args.output_dir or (images_dir / "cropped")).resolve()

    if not images_dir.exists():
        print(f"Images directory does not exist: {images_dir}")
        return 1

    if not images_dir.is_dir():
        print(f"Path is not a directory: {images_dir}")
        return 1

    if args.max_detection_dim <= 0:
        print(f"Max detection dimension must be positive: {args.max_detection_dim}")
        return 1

    if args.workers is not None and args.workers <= 0:
        print(f"Workers must be positive: {args.workers}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(iter_image_paths(images_dir, output_dir))
    if not image_paths:
        print(f"No supported images found in {images_dir}")
        return 0

    worker_count = args.workers or recommended_worker_count(len(image_paths))
    print(
        f"Processing {len(image_paths)} image(s) with {worker_count} worker(s). "
        f"Output: {output_dir}"
    )

    processed_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_path = {
            executor.submit(
                process_image_task,
                image_path,
                images_dir,
                output_dir,
                args.max_detection_dim,
            ): image_path
            for image_path in image_paths
        }

        for future in as_completed(future_to_path):
            image_path = future_to_path[future]
            try:
                result = future.result()
            except Exception as exc:
                failed_count += 1
                print(f"Failed: {image_path} ({exc})")
                continue

            processed_count += 1
            unwarp_status = "yes" if result.used_unwarp else "no"
            print(
                f"Processed: {result.image_path} -> {result.output_path} "
                f"(deskew={result.skew_angle:.2f} deg, unwarp={unwarp_status})"
            )

    print(
        f"Finished. Processed {processed_count} image(s), failed {failed_count}. "
        f"Output: {output_dir}"
    )
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
