#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps


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
DEFAULT_MARGIN_PERCENT = 5.0


def iter_image_paths(images_dir: Path, output_dir: Path) -> Iterable[Path]:
    for path in sorted(images_dir.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if output_dir in path.parents:
            continue

        yield path


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image

    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGB", image.size, "white")
        alpha = image.getchannel("A")
        background.paste(image.convert("RGBA"), mask=alpha)
        return background

    return image.convert("RGB")


def split_image(image_path: Path, output_dir: Path, margin_percent: float) -> tuple[Path, Path]:
    with Image.open(image_path) as image:
        normalized = ImageOps.exif_transpose(image)
        width, height = normalized.size
        margin_pixels = round(width * (margin_percent / 100.0))
        center = width / 2.0

        left_right_edge = min(width, math.ceil(center + margin_pixels))
        right_left_edge = max(0, math.floor(center - margin_pixels))

        left_half = ensure_rgb(normalized.crop((0, 0, left_right_edge, height)))
        right_half = ensure_rgb(normalized.crop((right_left_edge, 0, width, height)))

        left_path = output_dir / f"{image_path.stem}_left.jpg"
        right_path = output_dir / f"{image_path.stem}_right.jpg"

        left_half.save(left_path, quality=95)
        right_half.save(right_path, quality=95)

        return left_path, right_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Split images into left and right halves with an overlap around the center."
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
        help="Directory to write split images. Defaults to <images-dir>/split.",
    )
    parser.add_argument(
        "--margin-percent",
        default=DEFAULT_MARGIN_PERCENT,
        type=float,
        help=(
            "Extra width to include on both halves across the center line. "
            f"Defaults to {DEFAULT_MARGIN_PERCENT}."
        ),
    )
    args = parser.parse_args()

    images_dir = args.images_dir.resolve()
    output_dir = (args.output_dir or (images_dir / "split")).resolve()

    if not images_dir.exists():
        print(f"Images directory does not exist: {images_dir}")
        return 1

    if not images_dir.is_dir():
        print(f"Path is not a directory: {images_dir}")
        return 1

    if args.margin_percent < 0:
        print(f"Margin percent must be non-negative: {args.margin_percent}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    image_paths = list(iter_image_paths(images_dir, output_dir))

    for image_path in image_paths:
        left_path, right_path = split_image(image_path, output_dir, args.margin_percent)
        processed_count += 1
        print(f"Split: {image_path} -> {left_path.name}, {right_path.name}")

    print(f"Finished. Processed {processed_count} image(s). Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
