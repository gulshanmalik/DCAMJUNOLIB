#!/usr/bin/env python3
"""PyQt5 GUI for Hamamatsu camera: start/stop, exposure, preview, save raw .bin frames.

Dependencies: PyQt5, numpy, opencv-python

Run from the MyHamamatsu folder so it can import `camera` module, or set PYTHONPATH accordingly.
"""
import sys
import os
import threading
from datetime import datetime
import json

import numpy as np
import cv2
import time
import math

from PyQt5 import QtWidgets, QtGui, QtCore

try:
    from camera import CameraDevice
except Exception:
    from .camera import CameraDevice


class FrameWorker(QtCore.QObject):
    frame_ready = QtCore.pyqtSignal(object, object, object)  # (img_rgb, img16, hist)

    def __init__(self, camera: CameraDevice, parent=None):
        super().__init__(parent)
        self.camera = camera
        self._running = False
        self._preview_interval = 1.0 / 15.0  # default 15 FPS preview
        self._last_emit = 0.0
        self._thread = None
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

    def _apply_levels(self, img16: np.ndarray) -> np.ndarray:
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
            except Exception as e:
                # Try one recovery: hard_reset then retry once
                try:
                    self.camera.hard_reset()
                except Exception:
                    # recovery failed: notify GUI and pause briefly
                    self.frame_ready.emit(None, None, None)
                    time.sleep(0.5)
                    continue

                # after recovery try once
                try:
                    res = self.camera.get_frame(timeout_ms=2000)
                except Exception:
                    self.frame_ready.emit(None, None, None)
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
                self.frame_ready.emit(None, None, hist)
                continue
            img_rgb = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)
            self.frame_ready.emit(img_rgb, img16, hist)


class SaveWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, float)
    finished = QtCore.pyqtSignal(bool, str, int)

    def __init__(self, camera: CameraDevice, path: str, frame_limit: int, duration_limit: float):
        super().__init__()
        self.camera = camera
        self.path = path
        self.frame_limit = int(frame_limit)
        self.duration_limit = float(duration_limit)
        self._abort = False
        self.metadata_path = self._derive_metadata_path(path)

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
        except Exception as e:
            meta['camera_status_error'] = str(e)
        try:
            meta['roi'] = self.camera.get_subarray_info()
        except Exception as e:
            meta['roi_error'] = str(e)
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
        except Exception as e:
            print(f'Warning: failed to write metadata file {self.metadata_path}: {e}', file=sys.stderr)

    @QtCore.pyqtSlot()
    def run(self):
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
        except Exception as e:
            error_message = str(e) or 'Unknown error'
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

        if success:
            self.finished.emit(True, '', saved)
        else:
            self.finished.emit(False, error_message, saved)

    def stop(self):
        self._abort = True


class HistogramWidget(QtWidgets.QWidget):
    levelsChanged = QtCore.pyqtSignal(int, int)

    def __init__(self, parent=None, bins: int = 256):
        super().__init__(parent)
        self._bins = int(bins)
        self._hist = np.zeros(self._bins, dtype=np.int64)
        self._min_level = 0
        self._max_level = 65535
        self._active_handle = None
        self.setMinimumHeight(140)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set_histogram(self, hist):
        if hist is None:
            self._hist = np.zeros(self._bins, dtype=np.int64)
        else:
            arr = np.asarray(hist).flatten()
            if arr.size != self._bins:
                arr = np.resize(arr, self._bins)
            self._hist = arr.astype(np.int64, copy=False)
        self.update()

    def set_level_range(self, low: int, high: int, emit_signal: bool = False):
        low = max(0, min(int(low), 65535))
        high = max(low + 1, min(int(high), 65535))
        if low == self._min_level and high == self._max_level:
            return
        self._min_level = low
        self._max_level = high
        self.update()
        if emit_signal:
            self.levelsChanged.emit(self._min_level, self._max_level)

    def _hist_rect(self) -> QtCore.QRect:
        margin_x = 12
        margin_top = 12
        margin_bottom = 28
        rect = self.rect().adjusted(margin_x, margin_top, -margin_x, -margin_bottom)
        if rect.width() <= 0 or rect.height() <= 0:
            rect = QtCore.QRect(0, 0, max(1, rect.width()), max(1, rect.height()))
        return rect

    def _level_to_x(self, level: int, rect: QtCore.QRect) -> float:
        t = float(level) / 65535.0
        return rect.left() + t * rect.width()

    def _x_to_level(self, x: float, rect: QtCore.QRect) -> int:
        if rect.width() <= 0:
            return 0
        t = (x - rect.left()) / rect.width()
        t = max(0.0, min(1.0, t))
        return int(round(t * 65535))

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(30, 30, 30))

        rect = self._hist_rect()
        painter.fillRect(rect, QtGui.QColor(45, 45, 45))

        max_count = int(self._hist.max()) if self._hist.size else 0
        max_count = max(1, max_count)

        bar_width = rect.width() / float(self._bins)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(120, 180, 255, 200))

        for i in range(self._bins):
            count = self._hist[i]
            height = rect.height() * (count / max_count)
            x = rect.left() + i * bar_width
            bar_rect = QtCore.QRectF(x, rect.bottom() - height, bar_width, height)
            painter.drawRect(bar_rect)

        # draw handles
        handle_pen = QtGui.QPen(QtGui.QColor(255, 200, 0), 2)
        painter.setPen(handle_pen)
        painter.setBrush(QtGui.QColor(255, 200, 0))
        for level in (self._min_level, self._max_level):
            x = self._level_to_x(level, rect)
            painter.drawLine(QtCore.QPointF(x, rect.top()), QtCore.QPointF(x, rect.bottom()))
            triangle = QtGui.QPolygonF([
                QtCore.QPointF(x - 6, rect.bottom() + 8),
                QtCore.QPointF(x + 6, rect.bottom() + 8),
                QtCore.QPointF(x, rect.bottom() + 18),
            ])
            painter.drawPolygon(triangle)

        # draw labels
        painter.setPen(QtGui.QColor(200, 200, 200))
        painter.setFont(QtGui.QFont("", 9))
        painter.drawText(rect.adjusted(0, -20, 0, 0), QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, f"{self._min_level}")
        painter.drawText(rect.adjusted(0, -20, 0, 0), QtCore.Qt.AlignRight | QtCore.Qt.AlignTop, f"{self._max_level}")

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.LeftButton:
            return
        rect = self._hist_rect()
        if not rect.contains(event.pos()):
            return
        x = event.pos().x()
        x_min = self._level_to_x(self._min_level, rect)
        x_max = self._level_to_x(self._max_level, rect)
        if abs(x - x_min) <= abs(x - x_max):
            self._active_handle = 'min'
        else:
            self._active_handle = 'max'
        self._update_handle_from_pos(x, rect)

    def mouseMoveEvent(self, event):
        if not self._active_handle:
            return
        rect = self._hist_rect()
        self._update_handle_from_pos(event.pos().x(), rect)

    def mouseReleaseEvent(self, event):
        self._active_handle = None

    def _update_handle_from_pos(self, x: float, rect: QtCore.QRect):
        level = self._x_to_level(x, rect)
        if self._active_handle == 'min':
            self.set_level_range(level, self._max_level, emit_signal=True)
        elif self._active_handle == 'max':
            self.set_level_range(self._min_level, level, emit_signal=True)


class CameraWindow(QtWidgets.QMainWindow):
    def __init__(self, camera: CameraDevice):
        super().__init__()
        self.camera = camera
        self.setWindowTitle('Hamamatsu Camera - PyQt')
        self.resize(900, 700)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Preview label
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 480)
        layout.addWidget(self.preview_label)

        self.fps_label = QtWidgets.QLabel('FPS: --')
        self.fps_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.fps_label)

        # Controls
        ctrl = QtWidgets.QHBoxLayout()

        self.btn_start = QtWidgets.QPushButton('Start')
        self.btn_start.clicked.connect(self.on_start_stop)
        ctrl.addWidget(self.btn_start)

        self.btn_capture = QtWidgets.QPushButton('Save Burst')
        self.btn_capture.clicked.connect(self.on_save_burst)
        ctrl.addWidget(self.btn_capture)

        ctrl.addWidget(QtWidgets.QLabel('Exposure'))
        self.exp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.exp_slider.setMinimum(1)
        self.exp_slider.setMaximum(1000)
        self.exp_slider.setValue(1)
        self.exp_slider.valueChanged.connect(self.on_exp_changed)
        ctrl.addWidget(self.exp_slider)

        self.exp_edit = QtWidgets.QLineEdit('1')
        self.exp_edit.setFixedWidth(80)
        self.exp_edit.editingFinished.connect(self.on_exp_edit)
        ctrl.addWidget(self.exp_edit)

        layout.addLayout(ctrl)

        # ROI controls row
        roi_ctrl = QtWidgets.QHBoxLayout()
        roi_ctrl.addWidget(QtWidgets.QLabel('HPOS'))
        self.spin_hpos = QtWidgets.QSpinBox()
        self.spin_hpos.setRange(0, 16384)
        self.spin_hpos.setValue(0)
        roi_ctrl.addWidget(self.spin_hpos)

        roi_ctrl.addWidget(QtWidgets.QLabel('VPOS'))
        self.spin_vpos = QtWidgets.QSpinBox()
        self.spin_vpos.setRange(0, 16384)
        self.spin_vpos.setValue(0)
        roi_ctrl.addWidget(self.spin_vpos)

        roi_ctrl.addWidget(QtWidgets.QLabel('HSIZE'))
        self.spin_hsize = QtWidgets.QSpinBox()
        self.spin_hsize.setRange(2, 16384)
        self.spin_hsize.setSingleStep(2)
        roi_ctrl.addWidget(self.spin_hsize)

        roi_ctrl.addWidget(QtWidgets.QLabel('VSIZE'))
        self.spin_vsize = QtWidgets.QSpinBox()
        self.spin_vsize.setRange(2, 16384)
        self.spin_vsize.setSingleStep(2)
        roi_ctrl.addWidget(self.spin_vsize)

        self.btn_apply_roi = QtWidgets.QPushButton('Apply ROI')
        self.btn_apply_roi.setToolTip('Apply ROI using CameraDevice.set_subarray()')
        self.btn_apply_roi.clicked.connect(self.on_apply_roi)
        roi_ctrl.addWidget(self.btn_apply_roi)

        roi_ctrl.addWidget(QtWidgets.QLabel('Preview FPS'))
        self.spin_preview_fps = QtWidgets.QSpinBox()
        self.spin_preview_fps.setRange(1, 60)
        self.spin_preview_fps.setValue(15)
        self.spin_preview_fps.valueChanged.connect(self.on_preview_fps_changed)
        roi_ctrl.addWidget(self.spin_preview_fps)

        roi_ctrl.addWidget(QtWidgets.QLabel('Frames to save'))
        self.spin_save_frames = QtWidgets.QSpinBox()
        self.spin_save_frames.setRange(0, 1000000)
        self.spin_save_frames.setValue(1000)
        roi_ctrl.addWidget(self.spin_save_frames)

        roi_ctrl.addWidget(QtWidgets.QLabel('Duration (s)'))
        self.spin_save_secs = QtWidgets.QDoubleSpinBox()
        self.spin_save_secs.setRange(0.0, 3600.0)
        self.spin_save_secs.setDecimals(1)
        self.spin_save_secs.setValue(0.0)
        roi_ctrl.addWidget(self.spin_save_secs)

        layout.addLayout(roi_ctrl)

        # Histogram + display range controls
        hist_group = QtWidgets.QVBoxLayout()
        hist_label = QtWidgets.QLabel('Histogram / Display Range')
        hist_label.setAlignment(QtCore.Qt.AlignCenter)
        hist_group.addWidget(hist_label)
        self.hist_widget = HistogramWidget()
        self.hist_widget.levelsChanged.connect(self.on_hist_levels_changed)
        hist_group.addWidget(self.hist_widget)

        levels_ctrl = QtWidgets.QHBoxLayout()
        levels_ctrl.addWidget(QtWidgets.QLabel('Black level'))
        self.spin_black = QtWidgets.QSpinBox()
        self.spin_black.setRange(0, 65534)
        self.spin_black.valueChanged.connect(self.on_black_level_changed)
        levels_ctrl.addWidget(self.spin_black)
        levels_ctrl.addSpacing(12)
        levels_ctrl.addWidget(QtWidgets.QLabel('White level'))
        self.spin_white = QtWidgets.QSpinBox()
        self.spin_white.setRange(1, 65535)
        self.spin_white.valueChanged.connect(self.on_white_level_changed)
        levels_ctrl.addWidget(self.spin_white)
        levels_ctrl.addStretch(1)
        hist_group.addLayout(levels_ctrl)
        layout.addLayout(hist_group)

        # Status bar
        self.status = self.statusBar()

        # frame worker
        self.worker = FrameWorker(self.camera)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.set_preview_fps(self.spin_preview_fps.value())
        self.display_min = 0
        self.display_max = 65535
        self.hist_widget.set_level_range(self.display_min, self.display_max, emit_signal=False)
        self.spin_black.blockSignals(True)
        self.spin_black.setValue(self.display_min)
        self.spin_black.blockSignals(False)
        self.spin_white.blockSignals(True)
        self.spin_white.setValue(self.display_max)
        self.spin_white.blockSignals(False)
        self.worker.set_display_levels(self.display_min, self.display_max)

        self.last_rgb = None
        self.last_16 = None
        self.last_hist = None
        self.running = False
        self._frame_times = []
        self.save_thread = None
        self.save_worker = None
        self._resume_preview_after_save = False
        self._last_save_path = None
        self._last_metadata_path = None

        # Map slider to exposure range using a logarithmic mapping (slider 0..SMAX)
        self.SMAX = 1000
        try:
            emin, emax, estep, edef = self.camera.get_exposure_range()
            self.exp_min = max(emin, 1e-12)
            self.exp_max = max(emax, self.exp_min * 1.0)
            self.exp_step = estep
            self.exp_default = edef
        except Exception:
            # fallback to sensible defaults
            self.exp_min = 1e-6
            self.exp_max = 10.0
            self.exp_step = 1e-6
            self.exp_default = 1.0

        # set slider range
        self.exp_slider.setMinimum(0)
        self.exp_slider.setMaximum(self.SMAX)

        # helper: convert exposure <-> slider position
        def exp_to_pos(exp):
            # log mapping
            return int(round((math.log(exp) - math.log(self.exp_min)) / (math.log(self.exp_max) - math.log(self.exp_min)) * self.SMAX))

        def pos_to_exp(pos):
            t = float(pos) / float(self.SMAX)
            val = math.exp(math.log(self.exp_min) + t * (math.log(self.exp_max) - math.log(self.exp_min)))
            # quantize to step
            if self.exp_step and self.exp_step > 0:
                q = round(val / self.exp_step) * self.exp_step
                return max(self.exp_min, min(self.exp_max, float(q)))
            return max(self.exp_min, min(self.exp_max, val))

        self._exp_to_pos = exp_to_pos
        self._pos_to_exp = pos_to_exp

        # initialize current_exposure from device if possible
        try:
            status = self.camera.dump_status()
            exp = status.get('exposure')
            if exp is None:
                exp = self.exp_default
        except Exception:
            exp = self.exp_default

        self.current_exposure = float(exp)
        # set slider and edit
        try:
            self.exp_slider.setValue(self._exp_to_pos(self.current_exposure))
            self.exp_edit.setText(str(self.current_exposure))
        except Exception:
            self.exp_slider.setValue(0)
            self.exp_edit.setText(str(self.current_exposure))

        # initialize ROI spin boxes to current camera geometry
        try:
            width = int(getattr(self.camera, 'width', 0) or 0)
            height = int(getattr(self.camera, 'height', 0) or 0)
            if width > 0:
                self.spin_hsize.setValue(width)
            if height > 0:
                self.spin_vsize.setValue(height)
        except Exception:
            pass

    def on_start_stop(self):
        if not self.running:
            try:
                self.camera.start()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to start: {e}')
                return
            self.worker.start()
            self.btn_start.setText('Stop')
            self.running = True
            self.status.showMessage('Running')
        else:
            self.worker.stop()
            try:
                self.camera.stop()
            except Exception:
                pass
            self.btn_start.setText('Start')
            self.running = False
            self.status.showMessage('Stopped')

    def on_frame_ready(self, img_rgb, img16, hist):
        if img_rgb is None:
            QtWidgets.QMessageBox.critical(self, 'Frame Error', 'Frame worker failed to read frames')
            return
        self.last_rgb = img_rgb
        self.last_16 = img16
        if hist is not None:
            self.last_hist = hist
            self.hist_widget.set_histogram(hist)

        h, w, _ = img_rgb.shape
        bytes_per_line = 3 * w
        qimg = QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.preview_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.preview_label.setPixmap(pix)

        # update FPS label
        now = time.time()
        self._frame_times.append(now)
        # keep last 2 seconds of timestamps
        while self._frame_times and now - self._frame_times[0] > 2.0:
            self._frame_times.pop(0)
        fps = 0.0
        if len(self._frame_times) >= 2:
            dt = self._frame_times[-1] - self._frame_times[0]
            if dt > 0:
                fps = (len(self._frame_times) - 1) / dt
        self.fps_label.setText(f'FPS: {fps:.1f}  ({w}x{h})')

    def on_hist_levels_changed(self, low, high):
        self._set_display_levels(low, high, source='hist')

    def on_black_level_changed(self, value):
        value = int(value)
        high = self.display_max
        if value >= high:
            high = min(65535, value + 1)
        self._set_display_levels(value, high, source='black_spin')

    def on_white_level_changed(self, value):
        value = int(value)
        low = self.display_min
        if value >= 65535:
            value = 65535
        if value <= low:
            low = max(0, value - 1)
        self._set_display_levels(low, value, source='white_spin')

    def _set_display_levels(self, low, high, source=None):
        low = max(0, min(int(low), 65535))
        high = max(low + 1, min(int(high), 65535))
        if low == self.display_min and high == self.display_max:
            return
        self.display_min = low
        self.display_max = high
        if source != 'hist':
            self.hist_widget.set_level_range(low, high, emit_signal=False)
        if source != 'black_spin':
            self.spin_black.blockSignals(True)
            self.spin_black.setValue(low)
            self.spin_black.blockSignals(False)
        if source != 'white_spin':
            self.spin_white.blockSignals(True)
            self.spin_white.setValue(high)
            self.spin_white.blockSignals(False)
        if self.worker:
            self.worker.set_display_levels(low, high)

    def on_save_burst(self):
        if self.save_thread is not None:
            QtWidgets.QMessageBox.information(self, 'Saving', 'A save operation is already running.')
            return

        frame_limit = int(self.spin_save_frames.value())
        duration_limit = float(self.spin_save_secs.value())
        if frame_limit <= 0 and duration_limit <= 0:
            QtWidgets.QMessageBox.warning(self, 'Invalid settings', 'Set either a positive frame count or duration.')
            return

        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            'Save burst',
            f'burst_{datetime.now().strftime("%Y%m%d_%H%M%S")}.bin',
            'Binary files (*.bin)'
        )
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
            self.status.showMessage('Preview paused for saving...', 3000)

        self.btn_capture.setEnabled(False)

        self.save_thread = QtCore.QThread()
        self.save_worker = SaveWorker(self.camera, fname, frame_limit, duration_limit)
        self._last_metadata_path = self.save_worker.metadata_path
        self.save_worker.moveToThread(self.save_thread)
        self.save_thread.started.connect(self.save_worker.run)
        self.save_worker.progress.connect(self.on_save_progress)
        self.save_worker.finished.connect(self.on_save_finished)
        self.save_worker.finished.connect(self.save_thread.quit)
        self.save_worker.finished.connect(self.save_worker.deleteLater)
        self.save_thread.finished.connect(self._cleanup_save_thread)
        self.save_thread.start()
        self.status.showMessage('Saving frames...', 0)

    def on_exp_changed(self, val):
        # map slider position -> exposure (log mapping + quantize)
        try:
            exp = self._pos_to_exp(val)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'Mapping error', f'Failed to map slider to exposure: {e}')
            return

        # update edit box to show the actual exposure value (seconds)
        self.exp_edit.setText(f"{exp:.6g}")
        try:
            self.camera.set_exposure(float(exp))
            self.current_exposure = float(exp)
            self.status.showMessage(f'Exposure set to {exp:.6g} s')
        except Exception as e:
            # revert slider to last known good value (use mapping to position)
            self.exp_slider.blockSignals(True)
            try:
                self.exp_slider.setValue(self._exp_to_pos(self.current_exposure))
            except Exception:
                pass
            self.exp_slider.blockSignals(False)
            self.exp_edit.setText(f"{self.current_exposure:.6g}")
            QtWidgets.QMessageBox.warning(self, 'Exposure failed', f'Failed to set exposure: {e}')

    def on_exp_edit(self):
        try:
            v = float(self.exp_edit.text())
            # clamp to device range
            v = max(self.exp_min, min(self.exp_max, v))
            # quantize to step
            if self.exp_step and self.exp_step > 0:
                q = round(v / self.exp_step) * self.exp_step
            else:
                q = v

            # move slider to corresponding position
            try:
                self.exp_slider.setValue(self._exp_to_pos(q))
            except Exception:
                # if mapping fails, don't crash; fallback to current exposure
                pass

            try:
                self.camera.set_exposure(q)
                self.current_exposure = float(q)
                self.exp_edit.setText(f"{self.current_exposure:.6g}")
            except Exception as e:
                # revert edit and slider
                self.exp_edit.setText(f"{self.current_exposure:.6g}")
                try:
                    self.exp_slider.setValue(self._exp_to_pos(self.current_exposure))
                except Exception:
                    pass
                QtWidgets.QMessageBox.warning(self, 'Exposure failed', f'Failed to set exposure: {e}')
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'Exposure error', f'{e}')

    def on_apply_roi(self):
        hpos = int(self.spin_hpos.value())
        vpos = int(self.spin_vpos.value())
        hsize = int(self.spin_hsize.value())
        vsize = int(self.spin_vsize.value())
        try:
            self.camera.set_subarray(hpos, vpos, hsize, vsize, mode=2)
            self.status.showMessage(f'ROI applied: HPOS={hpos} VPOS={vpos} HSIZE={hsize} VSIZE={vsize}', 5000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'ROI failed', f'Failed to set ROI: {e}')

    def on_preview_fps_changed(self, value: int):
        try:
            self.worker.set_preview_fps(float(value))
            self.status.showMessage(f'Preview FPS limited to {value}', 3000)
        except Exception:
            pass

    def on_save_progress(self, frames_saved: int, elapsed: float):
        rate = 0.0
        if elapsed > 0:
            rate = frames_saved / elapsed
        self.status.showMessage(f'Saving... frames={frames_saved} ({rate:.1f} FPS)', 0)

    def on_save_finished(self, success: bool, error: str, frames_saved: int):
        self.btn_capture.setEnabled(True)
        if success:
            msg = f'Saved {frames_saved} frames to:\n{self._last_save_path}'
            if self._last_metadata_path:
                msg += f'\nMetadata: {self._last_metadata_path}'
            QtWidgets.QMessageBox.information(
                self,
                'Save complete',
                msg
            )
            status_msg = f'Saved {frames_saved} frames'
            if self._last_metadata_path:
                status_msg += ' (+ metadata)'
            self.status.showMessage(status_msg, 5000)
        else:
            QtWidgets.QMessageBox.critical(self, 'Save failed', error or 'Unknown error')
            self.status.showMessage('Save failed', 5000)

        if self._resume_preview_after_save:
            self._resume_preview_after_save = False
            if not self.running:
                self.on_start_stop()

    def _cleanup_save_thread(self):
        if self.save_thread:
            self.save_thread.deleteLater()
            self.save_thread = None
            self.save_worker = None

def main():
    cam = CameraDevice(cam_index=0)
    try:
        cam.init()
    except Exception as e:
        print('Failed to initialize camera:', e)
        return

    app = QtWidgets.QApplication(sys.argv)
    win = CameraWindow(cam)
    win.show()
    app.exec_()


if __name__ == '__main__':
    main()
