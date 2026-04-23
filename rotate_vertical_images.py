#!/usr/bin/env python3

from __future__ import annotations

import argparse
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


def iter_image_paths(images_dir: Path) -> Iterable[Path]:
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def rotate_if_vertical(image_path: Path) -> bool:
    with Image.open(image_path) as image:
        normalized = ImageOps.exif_transpose(image)
        width, height = normalized.size

        if height <= width:
            return False

        rotated = normalized.rotate(90, expand=True)
        exif = rotated.getexif()
        if exif:
            exif[274] = 1
            rotated.save(image_path, exif=exif.tobytes())
        else:
            rotated.save(image_path)

        return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rotate portrait images 90 degrees counterclockwise in the images directory."
        )
    )
    parser.add_argument(
        "--images-dir",
        default=Path(__file__).resolve().parent / "images",
        type=Path,
        help="Directory to scan for images. Defaults to ./images next to this script.",
    )
    args = parser.parse_args()

    images_dir = args.images_dir.resolve()
    if not images_dir.exists():
        print(f"Images directory does not exist: {images_dir}")
        return 1

    if not images_dir.is_dir():
        print(f"Path is not a directory: {images_dir}")
        return 1

    rotated_count = 0
    scanned_count = 0

    for image_path in iter_image_paths(images_dir):
        scanned_count += 1
        if rotate_if_vertical(image_path):
            rotated_count += 1
            print(f"Rotated: {image_path}")
        else:
            print(f"Skipped: {image_path}")

    print(f"Finished. Scanned {scanned_count} image(s), rotated {rotated_count}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
