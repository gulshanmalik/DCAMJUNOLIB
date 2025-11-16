#!/usr/bin/env python3
"""PySide6 GUI that mirrors the GTK side-panel interface for Hamamatsu cameras."""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from typing import Optional


import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

try:
    from . import filters as img_filters
except Exception:  # pragma: no cover
    import filters as img_filters  # type: ignore

try:
    from camera import CameraDevice
except Exception:  # pragma: no cover
    from .camera import CameraDevice


class FrameWorker(QtCore.QObject):
    """Background thread that fetches camera frames and emits them via a Qt signal."""

    frameReady = QtCore.Signal(object, object, object)  # (img_rgb, img16, histogram)

    def __init__(self, camera: CameraDevice, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.camera = camera
        self._running = False
        self._preview_interval = 1.0 / 15.0
        self._last_emit = 0.0
        self._thread: Optional[threading.Thread] = None
        self._display_min = 0
        self._display_max = 65535
        self._levels_lock = threading.Lock()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._last_emit = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def set_preview_fps(self, fps: float):
        if fps < 1:
            fps = 1.0
        self._preview_interval = 1.0 / float(fps)

    def set_display_levels(self, low: int, high: int):
        low = max(0, min(int(low), 65535))
        high = max(low + 1, min(int(high), 65535))
        with self._levels_lock:
            self._display_min = low
            self._display_max = high

    def _apply_levels(self, img16: np.ndarray):
        if img16 is None or img16.size == 0:
            return None
        with self._levels_lock:
            low = self._display_min
            high = self._display_max
        if high <= low:
            high = low + 1
        img = img16.astype(np.float32, copy=False)
        np.clip(img, low, high, out=img)
        img -= low
        img *= 255.0 / float(high - low)
        return img.astype(np.uint8)

    def _run(self):
        while self._running:
            try:
                res = self.camera.get_frame(timeout_ms=2000)
            except Exception:
                try:
                    self.camera.hard_reset()
                except Exception:
                    self.frameReady.emit(None, None, None)
                    time.sleep(0.5)
                    continue
                try:
                    res = self.camera.get_frame(timeout_ms=2000)
                except Exception:
                    self.frameReady.emit(None, None, None)
                    time.sleep(0.5)
                    continue

            if not res:
                continue

            img8, img16, idx, fr = res
            now = time.time()
            if now - self._last_emit < self._preview_interval:
                continue
            self._last_emit = now
            hist = None
            if img16 is not None and img16.size:
                hist = np.histogram(img16, bins=256, range=(0, 65535))[0]
            display = self._apply_levels(img16)
            if display is None:
                self.frameReady.emit(None, None, hist)
                continue
            img_rgb = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)
            self.frameReady.emit(img_rgb, img16, hist)


class SaveWorker(QtCore.QObject):
    """Background worker that records raw frames to disk and emits progress."""

    progress = QtCore.Signal(int, float)  # frames saved, elapsed seconds
    finished = QtCore.Signal(bool, str, int)  # success, error message, frames saved

    def __init__(self, camera: CameraDevice, path: str, frame_limit: int, duration_limit: float,
                 parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.camera = camera
        self.path = path
        self.frame_limit = int(frame_limit)
        self.duration_limit = float(duration_limit)
        self._abort = False
        self.metadata_path = self._derive_metadata_path(path)
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _derive_metadata_path(bin_path: str) -> str:
        base, _ = os.path.splitext(bin_path)
        if base:
            return f"{base}.json"
        return f"{bin_path}.json"

    def _initial_metadata(self) -> dict:
        meta = {
            'format': 'hamamatsu_raw_v1',
            'binary_file': os.path.abspath(self.path),
            'created_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            'frame_limit': self.frame_limit,
            'duration_limit_s': self.duration_limit,
            'dtype': 'uint16',
            'endianness': 'little',
            'frame_shape': None,
            'bytes_per_frame': None,
            'frame_stride_bytes': None,
            'bytes_per_pixel': 2,
        }
        try:
            meta['camera_status'] = self.camera.dump_status()
        except Exception as exc:
            meta['camera_status_error'] = str(exc)
        try:
            meta['roi'] = self.camera.get_subarray_info()
        except Exception as exc:
            meta['roi_error'] = str(exc)
        return meta

    @staticmethod
    def _metadata_from_frame(arr: np.ndarray) -> dict:
        h, w = arr.shape
        return {
            'frame_shape': [int(h), int(w)],
            'bytes_per_frame': int(arr.nbytes),
            'frame_stride_bytes': int(arr.strides[0]),
            'bytes_per_pixel': int(arr.dtype.itemsize),
            'dtype': str(arr.dtype),
            'numpy_dtype': arr.dtype.str,
        }

    def _finalize_metadata(self, metadata: dict) -> dict:
        if metadata.get('frame_shape') is None:
            try:
                width = int(getattr(self.camera, 'width', 0) or 0)
                height = int(getattr(self.camera, 'height', 0) or 0)
                if width and height:
                    metadata['frame_shape'] = [height, width]
                    metadata['bytes_per_frame'] = int(width * height * 2)
                    metadata['frame_stride_bytes'] = int(width * 2)
            except Exception:
                pass
        return metadata

    def _write_metadata_file(self, metadata: dict):
        finalized = self._finalize_metadata(dict(metadata))
        try:
            with open(self.metadata_path, 'w', encoding='utf-8') as meta_file:
                json.dump(finalized, meta_file, indent=2, sort_keys=True)
        except Exception as exc:
            print(f'Warning: failed to write metadata file {self.metadata_path}: {exc}', file=sys.stderr)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._abort = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def request_stop(self):
        self._abort = True

    def _run(self):
        saved = 0
        start_time = time.time()
        metadata = self._initial_metadata()
        stop_reason = 'completed'
        success = False
        error_message = ''
        try:
            self.camera.start()
            with open(self.path, 'wb') as f:
                while not self._abort:
                    if self.frame_limit > 0 and saved >= self.frame_limit:
                        stop_reason = 'frame_limit'
                        break
                    if self.duration_limit > 0 and (time.time() - start_time) >= self.duration_limit:
                        stop_reason = 'duration_limit'
                        break
                    res = self.camera.get_frame(timeout_ms=2000)
                    if not res:
                        continue
                    img8, img16, idx, fr = res
                    arr = np.ascontiguousarray(img16, dtype=np.uint16)
                    if metadata.get('frame_shape') is None:
                        metadata.update(self._metadata_from_frame(arr))
                    f.write(arr.astype('<u2').tobytes())
                    saved += 1
                    self.progress.emit(saved, time.time() - start_time)
            if self._abort and stop_reason == 'completed':
                stop_reason = 'user_abort'
            success = True
        except Exception as exc:
            error_message = str(exc) or 'Unknown error'
            metadata['error'] = error_message
            stop_reason = 'error'
        finally:
            metadata['status'] = 'success' if success else 'error'
            metadata['frames_saved'] = saved
            metadata['elapsed_s'] = time.time() - start_time
            metadata['finished_at'] = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
            metadata['stop_reason'] = stop_reason
            metadata['aborted'] = bool(self._abort)
            self._write_metadata_file(metadata)
            try:
                self.camera.stop()
            except Exception:
                pass

        self.finished.emit(success, error_message, saved)


class HistogramWidget(QtWidgets.QWidget):
    """Vertical histogram widget with draggable low/high level handles."""

    levelsChanged = QtCore.Signal(int, int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, bins: int = 256):
        super().__init__(parent)
        self._bins = int(bins)
        self._hist = np.zeros(self._bins, dtype=np.int64)
        self._min_level = 0
        self._max_level = 65535
        self._view_low = 0
        self._view_high = 65535
        self._active_handle: Optional[str] = None
        self.setMinimumWidth(160)
        self.setMinimumHeight(200)

    def sizeHint(self):  # noqa: D401
        return QtCore.QSize(180, 220)

    def set_histogram(self, hist):
        if hist is None:
            self._hist = np.zeros(self._bins, dtype=np.int64)
        else:
            arr = np.asarray(hist).flatten()
            if arr.size != self._bins:
                arr = np.resize(arr, self._bins)
            self._hist = arr.astype(np.int64, copy=False)
        self._auto_zoom_view()
        self.update()

    def set_level_range(self, low: int, high: int, emit_signal: bool = False):
        low = max(0, min(int(low), 65535))
        high = max(low + 1, min(int(high), 65535))
        if low == self._min_level and high == self._max_level:
            return
        self._min_level = low
        self._max_level = high
        self._ensure_view_contains_levels()
        self.update()
        if emit_signal:
            self.levelsChanged.emit(self._min_level, self._max_level)

    def _hist_rect(self) -> QtCore.QRectF:
        rect = self.rect().adjusted(12, 10, -12, -12)
        if rect.width() <= 0 or rect.height() <= 0:
            rect = QtCore.QRectF(rect.left(), rect.top(), max(1, rect.width()), max(1, rect.height()))
        return rect

    def _level_to_y(self, level: int, rect: QtCore.QRectF) -> float:
        level = max(0, min(65535, int(level)))
        span = max(1, self._view_high - self._view_low)
        t = (level - self._view_low) / span
        t = max(0.0, min(1.0, t))
        return rect.top() + rect.height() - t * rect.height()

    def _y_to_level(self, ypos: float, rect: QtCore.QRectF) -> int:
        if rect.height() <= 0:
            return 0
        t = (rect.top() + rect.height() - ypos) / rect.height()
        t = max(0.0, min(1.0, t))
        level = self._view_low + t * (self._view_high - self._view_low)
        return int(round(max(0, min(65535, level))))

    def _auto_zoom_view(self):
        arr = self._hist
        total = int(arr.sum())
        if total <= 0:
            self._view_low = 0
            self._view_high = 65535
            return
        cumsum = np.cumsum(arr)
        low_idx = int(np.searchsorted(cumsum, total * 0.01))
        high_idx = int(np.searchsorted(cumsum, total * 0.99))
        bin_width = 65535 / max(1, self._bins)
        start = low_idx * bin_width
        end = (high_idx + 1) * bin_width
        if end <= start:
            end = start + bin_width
        span = end - start
        margin = max(100.0, span * 0.5)
        self._view_low = int(max(0.0, start - margin))
        self._view_high = int(min(65535.0, end + margin))
        if self._view_high <= self._view_low:
            self._view_high = min(65535, self._view_low + 1)
        self._ensure_view_contains_levels()

    def _ensure_view_contains_levels(self):
        changed = False
        if self._min_level < self._view_low:
            self._view_low = max(0, self._min_level - 100)
            changed = True
        if self._max_level > self._view_high:
            self._view_high = min(65535, self._max_level + 100)
            changed = True
        if changed and self._view_high <= self._view_low:
            self._view_high = min(65535, self._view_low + 1)

    def paintEvent(self, _event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(30, 30, 30))

        rect = self._hist_rect()
        painter.fillRect(rect, QtGui.QColor(40, 40, 40))

        max_count = int(self._hist.max()) if self._hist.size else 0
        max_count = max(1, max_count)
        bin_width_level = 65535 / max(1, self._bins)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(120, 180, 255, 210))

        for i in range(self._bins):
            count = self._hist[i]
            if count <= 0:
                continue
            bin_low = i * bin_width_level
            bin_high = (i + 1) * bin_width_level
            if bin_high < self._view_low or bin_low > self._view_high:
                continue
            y_high = self._level_to_y(bin_low, rect)
            y_low = self._level_to_y(bin_high, rect)
            y_top = min(y_high, y_low)
            bar_height = abs(y_high - y_low)
            bar_width = rect.width() * (count / max_count)
            painter.drawRect(QtCore.QRectF(rect.left(), y_top, bar_width, bar_height))

        handle_pen = QtGui.QPen(QtGui.QColor(255, 200, 0), 2)
        painter.setPen(handle_pen)
        painter.setBrush(QtGui.QColor(255, 200, 0))
        for level in (self._min_level, self._max_level):
            y_pos = self._level_to_y(level, rect)
            painter.drawLine(QtCore.QPointF(rect.left(), y_pos), QtCore.QPointF(rect.right(), y_pos))
            handle_rect = QtCore.QRectF(rect.right() - 12, y_pos - 4, 12, 8)
            painter.drawRect(handle_rect)

        painter.setPen(QtGui.QColor(200, 200, 200))
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(QtCore.QPointF(rect.left() + 4, rect.top() + 12), f"{self._view_high}")
        painter.drawText(QtCore.QPointF(rect.left() + 4, rect.bottom() - 4), f"{self._view_low}")

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() != QtCore.Qt.LeftButton:
            return
        rect = self._hist_rect()
        if not rect.contains(event.pos()):
            return
        y_min = self._level_to_y(self._min_level, rect)
        y_max = self._level_to_y(self._max_level, rect)
        self._active_handle = 'min' if abs(event.y() - y_min) <= abs(event.y() - y_max) else 'max'
        self._update_handle_from_pos(event.y(), rect)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if not self._active_handle:
            return
        rect = self._hist_rect()
        self._update_handle_from_pos(event.y(), rect)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            self._active_handle = None

    def _update_handle_from_pos(self, ypos: float, rect: QtCore.QRectF):
        level = self._y_to_level(ypos, rect)
        if self._active_handle == 'min':
            self.set_level_range(level, self._max_level, emit_signal=True)
        elif self._active_handle == 'max':
            self.set_level_range(self._min_level, level, emit_signal=True)


class PreviewArea(QtWidgets.QWidget):
    """Widget that paints the latest QPixmap, optionally preserving aspect ratio."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._keep_aspect = True
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def set_pixmap(self, pixmap: Optional[QtGui.QPixmap]):
        self._pixmap = pixmap
        self.update()

    def set_keep_aspect(self, keep: bool):
        self._keep_aspect = bool(keep)
        self.update()

    def paintEvent(self, _event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtCore.Qt.black)
        if not self._pixmap or self._pixmap.isNull():
            return
        target = self.rect()
        if self._keep_aspect:
            scaled = self._pixmap.scaled(target.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            x = target.x() + (target.width() - scaled.width()) // 2
            y = target.y() + (target.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            scaled = self._pixmap.scaled(target.size(), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
            painter.drawPixmap(target.topLeft(), scaled)


class CameraWindow(QtWidgets.QMainWindow):
    """Main window that mirrors the GTK side-panel layout using PySide6 widgets."""

    def __init__(self, camera: CameraDevice):
        super().__init__()
        self.camera = camera
        self.setWindowTitle('JUNO - Icp Control')
        self.resize(1200, 760)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        content_layout = QtWidgets.QHBoxLayout()
        content_layout.setSpacing(8)
        root_layout.addLayout(content_layout, stretch=1)

        # Histogram column
        hist_container = QtWidgets.QFrame()
        hist_container.setMinimumWidth(210)
        hist_layout = QtWidgets.QVBoxLayout(hist_container)
        hist_layout.setContentsMargins(4, 4, 4, 4)
        hist_layout.setSpacing(6)

        hist_group = QtWidgets.QGroupBox('Histogram / Display Range')
        hist_group_layout = QtWidgets.QVBoxLayout(hist_group)
        hist_group_layout.setContentsMargins(6, 6, 6, 6)
        hist_group_layout.setSpacing(6)

        self.auto_levels_check = QtWidgets.QCheckBox('Auto display range')
        self.auto_levels_check.setChecked(True)
        self.auto_levels_check.toggled.connect(self.on_auto_levels_toggled)
        hist_group_layout.addWidget(self.auto_levels_check)

        self.hist_widget = HistogramWidget()
        self.hist_widget.levelsChanged.connect(self.on_hist_levels_changed)
        hist_group_layout.addWidget(self.hist_widget, stretch=1)

        levels_box = QtWidgets.QWidget()
        levels_layout = QtWidgets.QVBoxLayout(levels_box)
        levels_layout.setContentsMargins(0, 0, 0, 0)
        levels_layout.setSpacing(6)

        black_row = QtWidgets.QHBoxLayout()
        black_label = QtWidgets.QLabel('Black')
        self.spin_black = QtWidgets.QSpinBox()
        self.spin_black.setRange(0, 65534)
        self.spin_black.valueChanged.connect(self.on_black_level_changed)
        black_row.addWidget(black_label)
        black_row.addWidget(self.spin_black, stretch=1)
        levels_layout.addLayout(black_row)

        white_row = QtWidgets.QHBoxLayout()
        white_label = QtWidgets.QLabel('White')
        self.spin_white = QtWidgets.QSpinBox()
        self.spin_white.setRange(1, 65535)
        self.spin_white.valueChanged.connect(self.on_white_level_changed)
        white_row.addWidget(white_label)
        white_row.addWidget(self.spin_white, stretch=1)
        levels_layout.addLayout(white_row)

        hist_group_layout.addWidget(levels_box)
        hist_layout.addWidget(hist_group, stretch=1)
        content_layout.addWidget(hist_container, stretch=0)

        # Center preview panel
        center_widget = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(6)

        self.preview_area = PreviewArea()
        center_layout.addWidget(self.preview_area, stretch=1)

        self.fps_label = QtWidgets.QLabel('FPS: --')
        self.fps_label.setAlignment(QtCore.Qt.AlignCenter)
        center_layout.addWidget(self.fps_label)
        content_layout.addWidget(center_widget, stretch=1)

        # Right panel with tabs
        right_panel = QtWidgets.QFrame()
        right_panel.setMinimumWidth(380)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(6)

        tabs = QtWidgets.QTabWidget()
        right_layout.addWidget(tabs, stretch=1)

        # Settings tab
        settings_widget = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(settings_widget)
        settings_layout.setSpacing(10)
        settings_layout.setContentsMargins(8, 8, 8, 8)
        tabs.addTab(settings_widget, 'Settings')

        button_row = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton('Start')
        self.btn_start.clicked.connect(self.on_start_stop)
        self.btn_capture = QtWidgets.QPushButton('Save Burst')
        self.btn_capture.clicked.connect(self.on_save_burst)
        button_row.addWidget(self.btn_start)
        button_row.addWidget(self.btn_capture)
        settings_layout.addLayout(button_row)

        exposure_group = QtWidgets.QGroupBox('Exposure')
        exp_layout = QtWidgets.QVBoxLayout(exposure_group)
        exp_layout.setSpacing(6)
        exp_layout.setContentsMargins(6, 6, 6, 6)
        settings_layout.addWidget(exposure_group)

        self.SMAX = 1000
        self.exp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.exp_slider.setRange(0, self.SMAX)
        self.exp_slider.valueChanged.connect(self.on_exp_slider_changed)
        exp_layout.addWidget(self.exp_slider)

        exp_entry_row = QtWidgets.QHBoxLayout()
        exp_entry_row.addWidget(QtWidgets.QLabel('Seconds:'))
        self.exp_entry = QtWidgets.QLineEdit('1')
        self.exp_entry.setFixedWidth(100)
        self.exp_entry.returnPressed.connect(self.on_exp_entry_activated)
        self.exp_entry.editingFinished.connect(self.on_exp_entry_editing_finished)
        exp_entry_row.addWidget(self.exp_entry)
        exp_entry_row.addStretch(1)
        exp_layout.addLayout(exp_entry_row)

        roi_group = QtWidgets.QGroupBox('ROI / Acquisition')
        roi_layout = QtWidgets.QGridLayout(roi_group)
        roi_layout.setHorizontalSpacing(8)
        roi_layout.setVerticalSpacing(4)
        settings_layout.addWidget(roi_group)

        self.spin_hpos = QtWidgets.QSpinBox()
        self.spin_hpos.setRange(0, 16384)
        roi_layout.addWidget(QtWidgets.QLabel('HPOS'), 0, 0)
        roi_layout.addWidget(self.spin_hpos, 0, 1)

        self.spin_vpos = QtWidgets.QSpinBox()
        self.spin_vpos.setRange(0, 16384)
        roi_layout.addWidget(QtWidgets.QLabel('VPOS'), 1, 0)
        roi_layout.addWidget(self.spin_vpos, 1, 1)

        self.spin_hsize = QtWidgets.QSpinBox()
        self.spin_hsize.setRange(2, 16384)
        self.spin_hsize.setSingleStep(2)
        roi_layout.addWidget(QtWidgets.QLabel('HSIZE'), 2, 0)
        roi_layout.addWidget(self.spin_hsize, 2, 1)

        self.spin_vsize = QtWidgets.QSpinBox()
        self.spin_vsize.setRange(2, 16384)
        self.spin_vsize.setSingleStep(2)
        roi_layout.addWidget(QtWidgets.QLabel('VSIZE'), 3, 0)
        roi_layout.addWidget(self.spin_vsize, 3, 1)

        self.btn_maximize_roi = QtWidgets.QPushButton('Maximize')
        self.btn_maximize_roi.setToolTip('Reset ROI to full sensor resolution')
        self.btn_maximize_roi.clicked.connect(self.on_maximize_roi)
        roi_layout.addWidget(self.btn_maximize_roi, 4, 0, 1, 2)

        self.btn_apply_roi = QtWidgets.QPushButton('Apply ROI')
        self.btn_apply_roi.setToolTip('Apply ROI using CameraDevice.set_subarray()')
        self.btn_apply_roi.clicked.connect(self.on_apply_roi)
        roi_layout.addWidget(self.btn_apply_roi, 5, 0, 1, 2)

        roi_layout.addWidget(QtWidgets.QLabel('Preview FPS'), 6, 0)
        self.spin_preview_fps = QtWidgets.QSpinBox()
        self.spin_preview_fps.setRange(1, 60)
        self.spin_preview_fps.setValue(15)
        self.spin_preview_fps.valueChanged.connect(self.on_preview_fps_changed)
        roi_layout.addWidget(self.spin_preview_fps, 6, 1)

        roi_layout.addWidget(QtWidgets.QLabel('Frames to save'), 7, 0)
        self.spin_save_frames = QtWidgets.QSpinBox()
        self.spin_save_frames.setRange(0, 1_000_000)
        self.spin_save_frames.setValue(1000)
        roi_layout.addWidget(self.spin_save_frames, 7, 1)

        roi_layout.addWidget(QtWidgets.QLabel('Duration (s)'), 8, 0)
        self.spin_save_secs = QtWidgets.QDoubleSpinBox()
        self.spin_save_secs.setRange(0.0, 3600.0)
        self.spin_save_secs.setDecimals(1)
        self.spin_save_secs.setSingleStep(0.1)
        roi_layout.addWidget(self.spin_save_secs, 8, 1)

        settings_layout.addStretch(1)

        # Filters tab
        filters_widget = QtWidgets.QWidget()
        filters_layout = QtWidgets.QVBoxLayout(filters_widget)
        filters_layout.setSpacing(6)
        filters_layout.setContentsMargins(8, 8, 8, 8)
        tabs.addTab(filters_widget, 'Filters')

        self.chk_filter_smooth = QtWidgets.QCheckBox('Smooth preview (Gaussian)')
        self.chk_filter_smooth.toggled.connect(self.on_filter_smooth_toggled)
        filters_layout.addWidget(self.chk_filter_smooth)

        self.chk_filter_sharpen = QtWidgets.QCheckBox('Sharpen preview')
        self.chk_filter_sharpen.toggled.connect(self.on_filter_sharpen_toggled)
        filters_layout.addWidget(self.chk_filter_sharpen)

        self.chk_filter_mavg = QtWidgets.QCheckBox('Moving average (5 frames)')
        self.chk_filter_mavg.toggled.connect(self.on_filter_mavg_toggled)
        filters_layout.addWidget(self.chk_filter_mavg)

        self.chk_filter_mavg_sub = QtWidgets.QCheckBox('Rolling avg diff')
        self.chk_filter_mavg_sub.toggled.connect(self.on_filter_mavg_sub_toggled)
        filters_layout.addWidget(self.chk_filter_mavg_sub)

        sub_params_layout = QtWidgets.QHBoxLayout()
        sub_params_layout.addWidget(QtWidgets.QLabel('Length'))
        self.spin_mavg_length = QtWidgets.QSpinBox()
        self.spin_mavg_length.setRange(2, 200)
        self.spin_mavg_length.setValue(20)
        self.spin_mavg_length.valueChanged.connect(self.on_filter_mavg_sub_params_changed)
        sub_params_layout.addWidget(self.spin_mavg_length)
        sub_params_layout.addWidget(QtWidgets.QLabel('Normalize'))
        self.chk_mavg_sub_norm = QtWidgets.QCheckBox()
        self.chk_mavg_sub_norm.setChecked(True)
        self.chk_mavg_sub_norm.toggled.connect(self.on_filter_mavg_sub_norm_toggled)
        sub_params_layout.addWidget(self.chk_mavg_sub_norm)
        sub_params_layout.addStretch(1)
        filters_layout.addLayout(sub_params_layout)
        filters_layout.addStretch(1)

        # Display tab
        display_widget = QtWidgets.QWidget()
        display_layout = QtWidgets.QVBoxLayout(display_widget)
        display_layout.setSpacing(6)
        display_layout.setContentsMargins(8, 8, 8, 8)
        tabs.addTab(display_widget, 'Display')

        self.chk_show_fps = QtWidgets.QCheckBox('Show FPS label')
        self.chk_show_fps.setChecked(True)
        self.chk_show_fps.toggled.connect(self.on_display_show_fps_toggled)
        display_layout.addWidget(self.chk_show_fps)

        self.chk_keep_aspect = QtWidgets.QCheckBox('Preserve aspect ratio')
        self.chk_keep_aspect.setChecked(True)
        self.chk_keep_aspect.toggled.connect(self.on_display_keep_aspect_toggled)
        display_layout.addWidget(self.chk_keep_aspect)
        display_layout.addStretch(1)

        content_layout.addWidget(right_panel, stretch=0)

        # Status bar
        self.status = self.statusBar()
        self._status_timer = QtCore.QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(self._clear_status)

        # Worker + runtime state init
        self.worker = FrameWorker(self.camera)
        self.worker.frameReady.connect(self.on_frame_ready)
        self.display_min = 0
        self.display_max = 65535
        self.hist_widget.set_level_range(self.display_min, self.display_max, emit_signal=False)
        self._block_black_spin = False
        self._block_white_spin = False
        self._set_spin_value(self.spin_black, self.display_min, '_block_black_spin')
        self._set_spin_value(self.spin_white, self.display_max, '_block_white_spin')
        self.worker.set_display_levels(self.display_min, self.display_max)
        self.worker.set_preview_fps(self.spin_preview_fps.value())

        self.last_rgb = None
        self.last_16 = None
        self.last_hist = None
        self._latest_pixmap: Optional[QtGui.QPixmap] = None
        self._latest_image_bytes = None
        self.running = False
        self._frame_times: list[float] = []
        self.save_worker: Optional[SaveWorker] = None
        self._resume_preview_after_save = False
        self._last_save_path: Optional[str] = None
        self._last_metadata_path: Optional[str] = None
        self._frame_error_reported = False
        self.auto_levels_enabled = True
        self.filter_smoothing = False
        self.filter_sharpen = False
        self.filter_mavg = img_filters.MovingAverageFilter(window=5)
        self.filter_mavg_enabled = False
        self.filter_mavg_sub = img_filters.RollingMovingAverageDiff(length=10)
        self.filter_mavg_sub_enabled = False
        self.preview_keep_aspect = True
        self.show_fps_overlay = True
        self.fps_label.setVisible(self.show_fps_overlay)

        self._updating_exp_slider = False
        self._updating_exp_entry = False

        # Exposure mapping setup
        try:
            emin, emax, estep, edef = self.camera.get_exposure_range()
            self.exp_min = max(emin, 1e-12)
            self.exp_max = max(emax, self.exp_min)
            self.exp_step = estep
            self.exp_default = edef
        except Exception:
            self.exp_min = 1e-6
            self.exp_max = 10.0
            self.exp_step = 1e-6
            self.exp_default = 1.0

        def exp_to_pos(exp: float):
            exp = max(self.exp_min, min(self.exp_max, max(exp, 1e-12)))
            span = math.log(self.exp_max) - math.log(self.exp_min)
            if span <= 0:
                return 0
            return int(round((math.log(exp) - math.log(self.exp_min)) / span * self.SMAX))

        def pos_to_exp(pos: float):
            t = float(pos) / float(self.SMAX)
            val = math.exp(math.log(self.exp_min) + t * (math.log(self.exp_max) - math.log(self.exp_min)))
            if self.exp_step and self.exp_step > 0:
                val = round(val / self.exp_step) * self.exp_step
            return max(self.exp_min, min(self.exp_max, float(val)))

        self._exp_to_pos = exp_to_pos
        self._pos_to_exp = pos_to_exp

        try:
            status = self.camera.dump_status()
            exp = status.get('exposure')
            if exp is None:
                exp = self.exp_default
        except Exception:
            exp = self.exp_default

        self.current_exposure = float(exp)
        self._set_slider_value(self._exp_to_pos(self.current_exposure))
        self._set_exp_entry(self.current_exposure)

        try:
            width = int(getattr(self.camera, 'width', 0) or 0)
            height = int(getattr(self.camera, 'height', 0) or 0)
            if width > 0:
                self.spin_hsize.setValue(width)
            if height > 0:
                self.spin_vsize.setValue(height)
        except Exception:
            pass

        self.preview_area.set_keep_aspect(self.preview_keep_aspect)
        self.show_status('Ready')

    # ------------------------------------------------------------------
    # UI helpers
    def _set_spin_value(self, spin: QtWidgets.QSpinBox, value: int, attr: str):
        flag = getattr(self, attr)
        setattr(self, attr, True)
        spin.setValue(int(value))
        setattr(self, attr, flag)

    def _set_slider_value(self, value: int):
        self._updating_exp_slider = True
        self.exp_slider.setValue(int(value))
        self._updating_exp_slider = False

    def _set_exp_entry(self, value: float):
        self._updating_exp_entry = True
        self.exp_entry.setText(f"{value:.6g}")
        self._updating_exp_entry = False

    def show_status(self, message: str, timeout_ms: int = 0):
        if self._status_timer.isActive():
            self._status_timer.stop()
        self.status.showMessage(message)
        if timeout_ms > 0:
            self._status_timer.start(timeout_ms)

    def _clear_status(self):
        self.status.clearMessage()

    # ------------------------------------------------------------------
    # Qt life-cycle
    def closeEvent(self, event: QtGui.QCloseEvent):
        if self.save_worker:
            self.save_worker.request_stop()
        if self.worker:
            self.worker.stop()
        try:
            self.camera.stop()
        except Exception:
            pass
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Button + widget callbacks
    def on_start_stop(self):
        if not self.running:
            try:
                self.camera.start()
            except Exception as exc:
                self.show_error('Error', f'Failed to start: {exc}')
                return
            self.worker.start()
            self.btn_start.setText('Stop')
            self.running = True
            self.show_status('Running')
        else:
            self.worker.stop()
            try:
                self.camera.stop()
            except Exception:
                pass
            self.btn_start.setText('Start')
            self.running = False
            self.show_status('Stopped')

    def on_frame_ready(self, img_rgb, img16, hist):
        if img_rgb is None:
            if not self._frame_error_reported:
                self.show_error('Frame Error', 'Frame worker failed to read frames')
                self._frame_error_reported = True
            return
        self._frame_error_reported = False
        display_rgb = img_rgb
        if self.filter_smoothing:
            display_rgb = img_filters.gaussian_blur(display_rgb, ksize=5)
        if self.filter_sharpen:
            display_rgb = img_filters.sharpen(display_rgb)
        if self.filter_mavg_enabled:
            display_rgb = self.filter_mavg.apply(display_rgb)
        if self.filter_mavg_sub_enabled:
            diff = self.filter_mavg_sub.apply(display_rgb)
            if diff is not None:
                avg = self.filter_mavg.apply(display_rgb)
                if avg is None:
                    avg = display_rgb
                blended = cv2.addWeighted(
                    avg.astype(np.float32),
                    0.7,
                    diff.astype(np.float32),
                    0.3,
                    0.0
                )
                display_rgb = np.clip(blended, 0, 255).astype(np.uint8)
        display_rgb = np.clip(display_rgb, 0, 255).astype(np.uint8, copy=False)
        self.last_rgb = display_rgb
        self.last_16 = img16
        if hist is not None:
            self.last_hist = hist
            self.hist_widget.set_histogram(hist)
            if self.auto_levels_enabled:
                self._auto_adjust_levels_from_hist(hist)

        self._latest_pixmap = self._numpy_to_qpixmap(display_rgb)
        self.preview_area.set_pixmap(self._latest_pixmap)

        now = time.time()
        self._frame_times.append(now)
        while self._frame_times and now - self._frame_times[0] > 2.0:
            self._frame_times.pop(0)
        fps = 0.0
        if len(self._frame_times) >= 2:
            dt = self._frame_times[-1] - self._frame_times[0]
            if dt > 0:
                fps = (len(self._frame_times) - 1) / dt
        h, w, _ = display_rgb.shape
        self.fps_label.setText(f'FPS: {fps:.1f}  ({w}x{h})')

    def _numpy_to_qpixmap(self, img_rgb: np.ndarray) -> QtGui.QPixmap:
        arr = np.ascontiguousarray(img_rgb, dtype=np.uint8)
        self._latest_image_bytes = arr  # keep buffer alive
        h, w, _ = arr.shape
        image = QtGui.QImage(arr.data, w, h, w * 3, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(image)

    def on_hist_levels_changed(self, low: int, high: int):
        self._set_display_levels(low, high, source='hist')

    def on_filter_smooth_toggled(self, checked: bool):
        self.filter_smoothing = bool(checked)

    def on_filter_sharpen_toggled(self, checked: bool):
        self.filter_sharpen = bool(checked)

    def on_filter_mavg_toggled(self, checked: bool):
        self.filter_mavg_enabled = bool(checked)
        if not self.filter_mavg_enabled:
            self.filter_mavg.reset()

    def on_display_show_fps_toggled(self, checked: bool):
        self.show_fps_overlay = bool(checked)
        self.fps_label.setVisible(self.show_fps_overlay)

    def on_display_keep_aspect_toggled(self, checked: bool):
        self.preview_keep_aspect = bool(checked)
        self.preview_area.set_keep_aspect(self.preview_keep_aspect)

    def on_filter_mavg_sub_toggled(self, checked: bool):
        self.filter_mavg_sub_enabled = bool(checked)
        if not self.filter_mavg_sub_enabled:
            self.filter_mavg_sub.reset()

    def on_filter_mavg_sub_params_changed(self, _value: int):
        length = int(self.spin_mavg_length.value())
        self.filter_mavg_sub.set_params(length=length)

    def on_filter_mavg_sub_norm_toggled(self, checked: bool):
        self.filter_mavg_sub.set_params(normalize=bool(checked))

    def on_black_level_changed(self, value: int):
        if self._block_black_spin:
            return
        high = self.display_max
        if value >= high:
            high = min(65535, value + 1)
        self._set_display_levels(value, high, source='black_spin')

    def on_white_level_changed(self, value: int):
        if self._block_white_spin:
            return
        low = self.display_min
        if value >= 65535:
            value = 65535
        if value <= low:
            low = max(0, value - 1)
        self._set_display_levels(low, value, source='white_spin')

    def _set_display_levels(self, low: int, high: int, source: Optional[str] = None):
        if source in ('hist', 'black_spin', 'white_spin') and self.auto_levels_enabled:
            self.auto_levels_check.setChecked(False)
        low = max(0, min(int(low), 65535))
        high = max(low + 1, min(int(high), 65535))
        if low == self.display_min and high == self.display_max:
            return
        self.display_min = low
        self.display_max = high
        if source != 'hist':
            self.hist_widget.set_level_range(low, high, emit_signal=False)
        if source != 'black_spin':
            self._set_spin_value(self.spin_black, low, '_block_black_spin')
        if source != 'white_spin':
            self._set_spin_value(self.spin_white, high, '_block_white_spin')
        if self.worker:
            self.worker.set_display_levels(low, high)

    def on_auto_levels_toggled(self, checked: bool):
        self.auto_levels_enabled = bool(checked)
        if self.auto_levels_enabled and self.last_hist is not None:
            self._auto_adjust_levels_from_hist(self.last_hist)

    def _auto_adjust_levels_from_hist(self, hist):
        try:
            arr = np.asarray(hist, dtype=np.int64).flatten()
        except Exception:
            return
        if arr.size == 0:
            return
        total = int(arr.sum())
        if total <= 0:
            return
        cumsum = np.cumsum(arr)
        low_target = max(0, total * 0.01)
        high_target = max(low_target + 1, total * 0.99)
        low_idx = int(np.searchsorted(cumsum, low_target))
        high_idx = int(np.searchsorted(cumsum, high_target))
        if high_idx <= low_idx:
            high_idx = min(arr.size - 1, low_idx + 1)
        bin_width = 65535 / max(1, arr.size - 1)
        low_level = int(max(0, min(65535, round(low_idx * bin_width))))
        high_level = int(max(low_level + 1, min(65535, round((high_idx + 1) * bin_width))))
        self._set_display_levels(low_level, high_level, source='auto')

    def on_preview_fps_changed(self, value: int):
        try:
            self.worker.set_preview_fps(float(value))
            self.show_status(f'Preview FPS limited to {int(value)}', 3000)
        except Exception:
            pass

    def on_save_burst(self):
        if self.save_worker is not None:
            self.show_info('Saving', 'A save operation is already running.')
            return

        frame_limit = int(self.spin_save_frames.value())
        duration_limit = float(self.spin_save_secs.value())
        if frame_limit <= 0 and duration_limit <= 0:
            self.show_error('Invalid settings', 'Set either a positive frame count or duration.')
            return

        default_name = f'burst_{datetime.now().strftime("%Y%m%d_%H%M%S")}.bin'
        fname = self._choose_save_file(default_name)
        if not fname:
            return

        self._last_save_path = fname
        self._last_metadata_path = None
        self._resume_preview_after_save = self.running
        if self.running:
            self.worker.stop()
            try:
                self.camera.stop()
            except Exception:
                pass
            self.running = False
            self.btn_start.setText('Start')
            self.show_status('Preview paused for saving...', 3000)

        self.btn_capture.setEnabled(False)
        self.save_worker = SaveWorker(
            self.camera,
            fname,
            frame_limit,
            duration_limit
        )
        self.save_worker.progress.connect(self.on_save_progress)
        self.save_worker.finished.connect(self.on_save_finished)
        self._last_metadata_path = self.save_worker.metadata_path
        self.save_worker.start()
        self.show_status('Saving frames...', 0)

    def _choose_save_file(self, default_name: str) -> Optional[str]:
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            'Save burst',
            default_name,
            'Binary files (*.bin)'
        )
        return filename or None

    def on_save_progress(self, frames_saved: int, elapsed: float):
        rate = (frames_saved / elapsed) if elapsed > 0 else 0.0
        self.show_status(f'Saving... frames={frames_saved} ({rate:.1f} FPS)', 0)

    def on_save_finished(self, success: bool, error: str, frames_saved: int):
        self.btn_capture.setEnabled(True)
        if success:
            msg = f'Saved {frames_saved} frames to:\n{self._last_save_path}'
            if self._last_metadata_path:
                msg += f'\nMetadata: {self._last_metadata_path}'
            self.show_info('Save complete', msg)
            status_msg = f'Saved {frames_saved} frames'
            if self._last_metadata_path:
                status_msg += ' (+ metadata)'
            self.show_status(status_msg, 5000)
        else:
            self.show_error('Save failed', error or 'Unknown error')
            self.show_status('Save failed', 5000)

        if self._resume_preview_after_save:
            self._resume_preview_after_save = False
            if not self.running:
                self.on_start_stop()

        self.save_worker = None

    def on_exp_slider_changed(self, value: int):
        if self._updating_exp_slider:
            return
        pos = value
        try:
            exp = self._pos_to_exp(pos)
        except Exception as exc:
            self.show_error('Mapping error', f'Failed to map slider to exposure: {exc}')
            return
        self._set_exp_entry(exp)
        try:
            self.camera.set_exposure(float(exp))
            self.current_exposure = float(exp)
            self.show_status(f'Exposure set to {exp:.6g} s', 3000)
        except Exception as exc:
            self._set_slider_value(self._exp_to_pos(self.current_exposure))
            self._set_exp_entry(self.current_exposure)
            self.show_error('Exposure failed', f'Failed to set exposure: {exc}')

    def on_exp_entry_activated(self):
        self._apply_exp_entry()

    def on_exp_entry_editing_finished(self):
        self._apply_exp_entry()

    def _apply_exp_entry(self):
        if self._updating_exp_entry:
            return
        try:
            v = float(self.exp_entry.text())
        except ValueError:
            self.show_error('Exposure error', 'Invalid exposure value.')
            self._set_exp_entry(self.current_exposure)
            return
        v = max(self.exp_min, min(self.exp_max, v))
        if self.exp_step and self.exp_step > 0:
            v = round(v / self.exp_step) * self.exp_step
        try:
            self.camera.set_exposure(v)
            self.current_exposure = float(v)
            self._set_slider_value(self._exp_to_pos(v))
            self._set_exp_entry(v)
            self.show_status(f'Exposure set to {v:.6g} s', 3000)
        except Exception as exc:
            self._set_exp_entry(self.current_exposure)
            self._set_slider_value(self._exp_to_pos(self.current_exposure))
            self.show_error('Exposure failed', f'Failed to set exposure: {exc}')

    def on_apply_roi(self):
        hpos = int(self.spin_hpos.value())
        vpos = int(self.spin_vpos.value())
        hsize = int(self.spin_hsize.value())
        vsize = int(self.spin_vsize.value())
        was_running = self.running
        if was_running:
            self._pause_preview('Pausing preview to apply ROI...')
        try:
            self.camera.set_subarray(hpos, vpos, hsize, vsize, mode=2)
        except Exception as exc:
            self.show_error('ROI failed', f'Failed to set ROI: {exc}')
            if was_running:
                self._resume_preview('Preview resumed (ROI unchanged)')
            return
        msg = f'ROI applied: HPOS={hpos} VPOS={vpos} HSIZE={hsize} VSIZE={vsize}'
        if was_running:
            if not self._resume_preview(msg):
                return
        else:
            self.show_status(msg, 5000)

    def on_maximize_roi(self):
        try:
            info = self.camera.get_subarray_info()
        except Exception as exc:
            self.show_error('ROI failed', f'Failed to query camera info: {exc}')
            return

        def _extract_max(attr_tuple):
            if not attr_tuple:
                return None
            try:
                return int(attr_tuple[1])
            except Exception:
                return None

        max_width = _extract_max(info.get('sub_h_attr'))
        max_height = _extract_max(info.get('sub_v_attr'))
        if not max_width:
            max_width = int(info.get('full_width') or getattr(self.camera, 'width', 0) or 0)
        if not max_height:
            max_height = int(info.get('full_height') or getattr(self.camera, 'height', 0) or 0)
        if max_width <= 0 or max_height <= 0:
            self.show_error('ROI failed', 'Camera did not report maximum dimensions.')
            return

        self.spin_hpos.setValue(0)
        self.spin_vpos.setValue(0)
        self.spin_hsize.setValue(max_width)
        self.spin_vsize.setValue(max_height)
        self.on_apply_roi()

    def _pause_preview(self, message: str = 'Preview paused'):
        if not self.running:
            return False
        self.worker.stop()
        try:
            self.camera.stop()
        except Exception:
            pass
        self.running = False
        self.btn_start.setText('Start')
        self.show_status(message, 3000)
        return True

    def _resume_preview(self, message: str = 'Preview resumed'):
        try:
            self.camera.start()
        except Exception as exc:
            self.show_error('Restart failed', f'Failed to restart camera: {exc}')
            return False
        self.worker.start()
        self.running = True
        self.btn_start.setText('Stop')
        self.show_status(message, 5000)
        return True

    # ------------------------------------------------------------------
    # Message helpers
    def show_error(self, title: str, message: str):
        QtWidgets.QMessageBox.critical(self, title, message)

    def show_info(self, title: str, message: str):
        QtWidgets.QMessageBox.information(self, title, message)


def main():
    app = QtWidgets.QApplication(sys.argv)
    cam = CameraDevice(cam_index=0)
    try:
        cam.init()
    except Exception as exc:
        print('Failed to initialize camera:', exc)
        return
    win = CameraWindow(cam)
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
