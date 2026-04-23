#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps

try:
    import numpy as np
    from skimage import filters
except ImportError as exc:
    raise SystemExit("Install numpy and scikit-image to run this script.") from exc

try:
    from PyQt6.QtCore import QObject, QRunnable, QThread, QThreadPool, Qt, pyqtSignal
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )
    from PIL.ImageQt import ImageQt

    HORIZONTAL = Qt.Orientation.Horizontal
    ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
    KEEP_ASPECT = Qt.AspectRatioMode.KeepAspectRatio
    SMOOTH_TRANSFORMATION = Qt.TransformationMode.SmoothTransformation
    MESSAGEBOX_YES = QMessageBox.StandardButton.Yes
    MESSAGEBOX_NO = QMessageBox.StandardButton.No
    Signal = pyqtSignal
except ImportError:
    try:
        from PyQt5.QtCore import QObject, QRunnable, QThread, QThreadPool, Qt, pyqtSignal
        from PyQt5.QtGui import QPixmap
        from PyQt5.QtWidgets import (
            QApplication,
            QHBoxLayout,
            QLabel,
            QMessageBox,
            QPushButton,
            QSlider,
            QVBoxLayout,
            QWidget,
        )
        from PIL.ImageQt import ImageQt

        HORIZONTAL = Qt.Horizontal
        ALIGN_CENTER = Qt.AlignCenter
        KEEP_ASPECT = Qt.KeepAspectRatio
        SMOOTH_TRANSFORMATION = Qt.SmoothTransformation
        MESSAGEBOX_YES = QMessageBox.Yes
        MESSAGEBOX_NO = QMessageBox.No
        Signal = pyqtSignal
    except ImportError as exc:
        raise SystemExit("Install PyQt6 or PyQt5 to run this script.") from exc


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
MAX_CONVERSION_WORKERS = 4


@dataclass(frozen=True)
class ProcessingSettings:
    black_level: int
    white_level: int
    gradient_radius_percent: int
    flatten_strength: int


class PreviewWorker(QObject):
    finished = Signal(object, object)
    failed = Signal(object, str)
    done = Signal()

    def __init__(self, image_path: Path, settings: ProcessingSettings) -> None:
        super().__init__()
        self.image_path = image_path
        self.settings = settings

    def run(self) -> None:
        try:
            rendered, _ = render_image_file(self.image_path, self.settings)
        except Exception as exc:
            self.failed.emit(self.image_path, str(exc))
        else:
            self.finished.emit(self.image_path, rendered)
        finally:
            self.done.emit()


class ConversionTaskSignals(QObject):
    finished = Signal(object)
    failed = Signal(object, str)


class ConversionTask(QRunnable):
    def __init__(self, image_path: Path, settings: ProcessingSettings) -> None:
        super().__init__()
        self.image_path = image_path
        self.settings = settings
        self.signals = ConversionTaskSignals()

    def run(self) -> None:
        try:
            rendered, exif = render_image_file(self.image_path, self.settings)
            save_rendered_image(self.image_path, rendered, exif)
        except Exception as exc:
            self.signals.failed.emit(self.image_path, str(exc))
        else:
            self.signals.finished.emit(self.image_path)


def iter_image_paths(images_dir: Path) -> Iterable[Path]:
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def recommended_worker_count(image_count: int) -> int:
    if image_count <= 1:
        return 1

    # Each worker holds large float arrays during illumination flattening,
    # so cap the pool to keep memory use reasonable.
    available_cores = os.cpu_count() or 1
    return max(1, min(MAX_CONVERSION_WORKERS, available_cores, image_count))


def build_levels_lut(black_level: int, white_level: int) -> list[int]:
    span = max(1, white_level - black_level)
    lut: list[int] = []

    for value in range(256):
        if value <= black_level:
            lut.append(0)
        elif value >= white_level:
            lut.append(255)
        else:
            lut.append(round((value - black_level) * 255 / span))

    return lut


def apply_levels_to_image(image: Image.Image, lut: list[int]) -> Image.Image:
    grayscale = image if image.mode == "L" else ImageOps.grayscale(image)
    return grayscale.point(lut)


def flatten_illumination(
    image: Image.Image, gradient_radius_percent: int, flatten_strength: int
) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    if flatten_strength <= 0:
        return grayscale

    grayscale_array = np.asarray(grayscale, dtype=np.float32) / 255.0
    sigma = max(1.0, max(grayscale_array.shape) * gradient_radius_percent / 100.0)

    # Estimate the large-scale lighting field, then flatten it while keeping
    # the page close to its original overall brightness.
    illumination = filters.gaussian(grayscale_array, sigma=sigma, preserve_range=True)
    illumination = np.clip(illumination, 1e-3, None)
    reference_level = float(np.median(illumination))
    corrected = np.clip(grayscale_array * (reference_level / illumination), 0.0, 1.0)

    blend = flatten_strength / 100.0
    flattened = np.clip(
        grayscale_array * (1.0 - blend) + corrected * blend,
        0.0,
        1.0,
    )
    return Image.fromarray(np.rint(flattened * 255.0).astype(np.uint8), mode="L")


def render_image_file(
    image_path: Path, settings: ProcessingSettings
) -> tuple[Image.Image, Image.Exif]:
    lut = build_levels_lut(settings.black_level, settings.white_level)

    with Image.open(image_path) as image:
        normalized = ImageOps.exif_transpose(image)
        flattened = flatten_illumination(
            normalized,
            settings.gradient_radius_percent,
            settings.flatten_strength,
        )
        rendered = apply_levels_to_image(flattened, lut)
        return rendered.copy(), normalized.getexif()


def save_rendered_image(image_path: Path, rendered: Image.Image, exif: Image.Exif) -> None:
    save_kwargs = {}
    if exif:
        exif[274] = 1
        save_kwargs["exif"] = exif.tobytes()

    try:
        rendered.save(image_path, **save_kwargs)
    except TypeError:
        rendered.save(image_path)


class LevelsWindow(QWidget):
    def __init__(self, images_dir: Path) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.image_paths = list(iter_image_paths(images_dir))
        self.current_image_path = self.image_paths[0] if self.image_paths else None
        self.current_preview_image: Image.Image | None = None
        self.preview_thread: QThread | None = None
        self.preview_worker: PreviewWorker | None = None
        self.preview_active = False
        self.batch_running = False
        self.batch_total = 0
        self.batch_completed = 0
        self.batch_successes = 0
        self.batch_failures: list[str] = []
        self.thread_pool = QThreadPool(self)
        self.thread_pool.setMaxThreadCount(recommended_worker_count(len(self.image_paths)))

        self.setWindowTitle("B&W Levels + Lighting Correction")
        self.resize(1400, 900)

        self.black_slider = QSlider(HORIZONTAL)
        self.white_slider = QSlider(HORIZONTAL)
        self.gradient_radius_slider = QSlider(HORIZONTAL)
        self.flatten_strength_slider = QSlider(HORIZONTAL)
        self.black_label = QLabel()
        self.white_label = QLabel()
        self.gradient_radius_label = QLabel()
        self.flatten_strength_label = QLabel()
        self.image_name_label = QLabel()
        self.image_label = QLabel()
        self.preview_button = QPushButton("Preview")
        self.convert_button = QPushButton("Convert")
        self.status_label = QLabel()

        self._build_ui()
        self._sync_slider_labels()
        self._initialize_state()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(18)

        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(12)

        intro_label = QLabel(
            "Adjust the lighting-flattening pass and black/white points, preview the first image, then convert the full images folder in place."
        )
        intro_label.setWordWrap(True)

        self.black_slider.setRange(0, 254)
        self.black_slider.setValue(0)
        self.black_slider.valueChanged.connect(self._on_black_slider_changed)

        self.white_slider.setRange(1, 255)
        self.white_slider.setValue(255)
        self.white_slider.valueChanged.connect(self._on_white_slider_changed)

        self.gradient_radius_slider.setRange(1, 25)
        self.gradient_radius_slider.setValue(8)
        self.gradient_radius_slider.valueChanged.connect(self._sync_slider_labels)

        self.flatten_strength_slider.setRange(0, 100)
        self.flatten_strength_slider.setValue(80)
        self.flatten_strength_slider.valueChanged.connect(self._sync_slider_labels)

        self.preview_button.clicked.connect(self.preview_current_image)
        self.convert_button.clicked.connect(self.convert_all_images)

        controls_layout.addWidget(intro_label)
        controls_layout.addWidget(self.black_label)
        controls_layout.addWidget(self.black_slider)
        controls_layout.addWidget(self.white_label)
        controls_layout.addWidget(self.white_slider)
        controls_layout.addWidget(self.gradient_radius_label)
        controls_layout.addWidget(self.gradient_radius_slider)
        controls_layout.addWidget(self.flatten_strength_label)
        controls_layout.addWidget(self.flatten_strength_slider)
        controls_layout.addWidget(self.preview_button)
        controls_layout.addWidget(self.convert_button)
        controls_layout.addStretch()
        controls_layout.addWidget(self.status_label)

        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)
        controls_widget.setMaximumWidth(320)

        self.image_name_label.setAlignment(ALIGN_CENTER)
        self.image_name_label.setWordWrap(True)

        self.image_label.setAlignment(ALIGN_CENTER)
        self.image_label.setMinimumSize(700, 700)
        self.image_label.setStyleSheet(
            "background-color: #1a1a1a; color: #efefef; border: 1px solid #444;"
        )

        preview_layout = QVBoxLayout()
        preview_layout.setSpacing(10)
        preview_layout.addWidget(self.image_name_label)
        preview_layout.addWidget(self.image_label, 1)

        preview_widget = QWidget()
        preview_widget.setLayout(preview_layout)

        layout.addWidget(controls_widget, 0)
        layout.addWidget(preview_widget, 1)

    def _initialize_state(self) -> None:
        if not self.image_paths:
            self.image_name_label.setText("No supported images found.")
            self.image_label.setText(f"No images found in {self.images_dir}")
            self.status_label.setText("Add files to the images folder and relaunch the tool.")
            self._set_controls_enabled(False)
            return

        self.image_name_label.setText(str(self.current_image_path.name))
        self.preview_current_image()

    def _sync_slider_labels(self) -> None:
        self.black_label.setText(f"Black level: {self.black_slider.value()}")
        self.white_label.setText(f"White level: {self.white_slider.value()}")
        self.gradient_radius_label.setText(
            f"Gradient radius: {self.gradient_radius_slider.value()}% of image size"
        )
        self.flatten_strength_label.setText(
            f"Flatten strength: {self.flatten_strength_slider.value()}%"
        )

    def _current_settings(self) -> ProcessingSettings:
        return ProcessingSettings(
            black_level=self.black_slider.value(),
            white_level=self.white_slider.value(),
            gradient_radius_percent=self.gradient_radius_slider.value(),
            flatten_strength=self.flatten_strength_slider.value(),
        )

    def _set_controls_enabled(self, enabled: bool) -> None:
        has_images = bool(self.image_paths)
        active = enabled and has_images
        self.preview_button.setEnabled(active)
        self.convert_button.setEnabled(active)
        self.black_slider.setEnabled(active)
        self.white_slider.setEnabled(active)
        self.gradient_radius_slider.setEnabled(active)
        self.flatten_strength_slider.setEnabled(active)

    def _start_preview_worker(self, image_path: Path, settings: ProcessingSettings) -> None:
        self.preview_active = True
        self._set_controls_enabled(False)

        self.preview_thread = QThread(self)
        self.preview_worker = PreviewWorker(image_path, settings)
        self.preview_worker.moveToThread(self.preview_thread)

        self.preview_thread.started.connect(self.preview_worker.run)
        self.preview_worker.finished.connect(self._handle_preview_success)
        self.preview_worker.failed.connect(self._handle_preview_failure)
        self.preview_worker.done.connect(self.preview_thread.quit)
        self.preview_worker.done.connect(self.preview_worker.deleteLater)
        self.preview_thread.finished.connect(self._cleanup_preview_worker)
        self.preview_thread.finished.connect(self.preview_thread.deleteLater)
        self.preview_thread.start()

    def _cleanup_preview_worker(self) -> None:
        self.preview_active = False
        self.preview_worker = None
        self.preview_thread = None

        if not self.batch_running:
            self._set_controls_enabled(True)

    def _handle_preview_success(self, image_path: Path, rendered: Image.Image) -> None:
        self.current_preview_image = rendered
        self.image_name_label.setText(str(image_path.name))
        self._update_preview_label()
        self.status_label.setText(f"Preview ready: {image_path.name}")

    def _handle_preview_failure(self, image_path: Path, error_message: str) -> None:
        QMessageBox.critical(
            self,
            "Preview Failed",
            f"Could not render {image_path}.\n\n{error_message}",
        )
        self.status_label.setText("Preview failed.")

    def _record_conversion_result(self, image_path: Path) -> None:
        self.batch_completed += 1
        failures = len(self.batch_failures)
        failure_suffix = f", {failures} failed" if failures else ""
        self.status_label.setText(
            f"Processed {self.batch_completed}/{self.batch_total}: {image_path.name}"
            f"{failure_suffix}"
        )

        if self.batch_completed == self.batch_total:
            self._finish_conversion()

    def _handle_conversion_success(self, image_path: Path) -> None:
        self.batch_successes += 1
        self._record_conversion_result(image_path)

    def _handle_conversion_failure(self, image_path: Path, error_message: str) -> None:
        self.batch_failures.append(f"{image_path.name}: {error_message}")
        self._record_conversion_result(image_path)

    def _finish_conversion(self) -> None:
        self.batch_running = False
        self._set_controls_enabled(True)

        converted_count = self.batch_successes
        failures = self.batch_failures

        if failures:
            self.status_label.setText(
                f"Converted {converted_count} image(s), {len(failures)} failed."
            )
            QMessageBox.warning(
                self,
                "Conversion Finished With Errors",
                "Some images could not be converted:\n\n" + "\n".join(failures[:20]),
            )
        else:
            self.status_label.setText(f"Converted {converted_count} image(s).")
            QMessageBox.information(
                self,
                "Conversion Complete",
                f"Converted {converted_count} image(s) in place.",
            )

        if self.current_image_path is not None:
            self.preview_current_image()

    def _on_black_slider_changed(self, value: int) -> None:
        if value >= self.white_slider.value():
            self.white_slider.blockSignals(True)
            self.white_slider.setValue(min(255, value + 1))
            self.white_slider.blockSignals(False)
        self._sync_slider_labels()

    def _on_white_slider_changed(self, value: int) -> None:
        if value <= self.black_slider.value():
            self.black_slider.blockSignals(True)
            self.black_slider.setValue(max(0, value - 1))
            self.black_slider.blockSignals(False)
        self._sync_slider_labels()

    def _update_preview_label(self) -> None:
        if self.current_preview_image is None:
            return

        preview_image = self.current_preview_image.convert("RGBA")
        pixmap = QPixmap.fromImage(ImageQt(preview_image))
        target_size = self.image_label.size()

        if target_size.width() > 0 and target_size.height() > 0:
            pixmap = pixmap.scaled(target_size, KEEP_ASPECT, SMOOTH_TRANSFORMATION)

        self.image_label.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_preview_label()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.preview_active or self.batch_running:
            QMessageBox.information(
                self,
                "Processing In Progress",
                "Wait for the current preview or conversion to finish before closing the window.",
            )
            event.ignore()
            return

        super().closeEvent(event)

    def preview_current_image(self) -> None:
        if self.current_image_path is None or self.preview_active or self.batch_running:
            return

        settings = self._current_settings()
        self.status_label.setText(
            "Previewing "
            f"{self.current_image_path.name} with black {settings.black_level}, "
            f"white {settings.white_level}, radius {settings.gradient_radius_percent}%, "
            f"and flatten {settings.flatten_strength}%."
        )
        self._start_preview_worker(self.current_image_path, settings)

    def convert_all_images(self) -> None:
        if not self.image_paths or self.preview_active or self.batch_running:
            return

        settings = self._current_settings()

        response = QMessageBox.question(
            self,
            "Convert All Images",
            (
                f"This will overwrite {len(self.image_paths)} image(s) in:\n"
                f"{self.images_dir}\n\n"
                f"Black level: {settings.black_level}\n"
                f"White level: {settings.white_level}\n"
                f"Gradient radius: {settings.gradient_radius_percent}%\n"
                f"Flatten strength: {settings.flatten_strength}%\n\n"
                "Continue?"
            ),
            MESSAGEBOX_YES | MESSAGEBOX_NO,
            MESSAGEBOX_NO,
        )
        if response != MESSAGEBOX_YES:
            return

        self.batch_running = True
        self.batch_total = len(self.image_paths)
        self.batch_completed = 0
        self.batch_successes = 0
        self.batch_failures = []
        self._set_controls_enabled(False)

        worker_count = self.thread_pool.maxThreadCount()
        worker_label = "worker" if worker_count == 1 else "workers"
        self.status_label.setText(
            f"Converting {self.batch_total} image(s) with {worker_count} {worker_label}."
        )

        for image_path in self.image_paths:
            task = ConversionTask(image_path, settings)
            task.signals.finished.connect(self._handle_conversion_success)
            task.signals.failed.connect(self._handle_conversion_failure)
            self.thread_pool.start(task)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Preview and apply lighting correction plus black/white level conversion "
            "to images."
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

    app = QApplication(sys.argv)
    window = LevelsWindow(images_dir)
    window.show()
    exec_method = getattr(app, "exec", None)
    if exec_method is not None:
        return exec_method()

    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
