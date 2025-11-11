#!/usr/bin/env python3
"""GTK3 GUI for the Hamamatsu camera that mirrors the features of qt_gui.py."""

import sys
import os
import threading
from datetime import datetime
import json
import math
import time

import numpy as np
import cv2

import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib, GObject  # noqa: E402
import cairo  # noqa: E402

try:
    from camera import CameraDevice
except Exception:  # pragma: no cover - fallback for package use
    from .camera import CameraDevice


class FrameWorker:
    """Background thread that fetches frames and posts them back to the GTK loop."""

    def __init__(self, camera: CameraDevice, callback):
        self.camera = camera
        self._callback = callback
        self._running = False
        self._preview_interval = 1.0 / 15.0
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
                    self._emit(None, None, None)
                    time.sleep(0.5)
                    continue
                try:
                    res = self.camera.get_frame(timeout_ms=2000)
                except Exception:
                    self._emit(None, None, None)
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
                self._emit(None, None, hist)
                continue
            img_rgb = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)
            self._emit(img_rgb, img16, hist)

    def _emit(self, img_rgb, img16, hist):
        if not self._callback:
            return

        def _dispatch():
            self._callback(img_rgb, img16, hist)
            return False

        GLib.idle_add(_dispatch, priority=GLib.PRIORITY_DEFAULT)


class SaveWorker:
    """Background worker that writes raw frames to disk along with metadata."""

    def __init__(self, camera: CameraDevice, path: str, frame_limit: int, duration_limit: float,
                 progress_cb=None, finished_cb=None):
        self.camera = camera
        self.path = path
        self.frame_limit = int(frame_limit)
        self.duration_limit = float(duration_limit)
        self._abort = False
        self.metadata_path = self._derive_metadata_path(path)
        self._progress_cb = progress_cb
        self._finished_cb = finished_cb
        self._thread = None

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

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._abort = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def request_stop(self):
        self._abort = True

    def _emit_progress(self, frames_saved: int, elapsed: float):
        if not self._progress_cb:
            return False
        self._progress_cb(frames_saved, elapsed)
        return False

    def _emit_finished(self, success: bool, error: str, frames_saved: int):
        if not self._finished_cb:
            return False
        self._finished_cb(success, error, frames_saved)
        return False

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
                    GLib.idle_add(self._emit_progress, saved, time.time() - start_time,
                                  priority=GLib.PRIORITY_DEFAULT)
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

        GLib.idle_add(self._emit_finished, success, error_message, saved, priority=GLib.PRIORITY_DEFAULT)


class HistogramWidget(Gtk.DrawingArea):
    """Custom histogram widget with draggable black/white level handles."""

    __gsignals__ = {
        'levels-changed': (GObject.SIGNAL_RUN_FIRST, None, (int, int)),
    }

    def __init__(self, bins: int = 256):
        super().__init__()
        self._bins = int(bins)
        self._hist = np.zeros(self._bins, dtype=np.int64)
        self._min_level = 0
        self._max_level = 65535
        self._active_handle = None
        self.set_size_request(-1, 140)
        self.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK |
            Gdk.EventMask.BUTTON_RELEASE_MASK |
            Gdk.EventMask.POINTER_MOTION_MASK
        )
        self.connect('draw', self._on_draw)
        self.connect('button-press-event', self._on_button_press)
        self.connect('button-release-event', self._on_button_release)
        self.connect('motion-notify-event', self._on_motion)

    def set_histogram(self, hist):
        if hist is None:
            self._hist = np.zeros(self._bins, dtype=np.int64)
        else:
            arr = np.asarray(hist).flatten()
            if arr.size != self._bins:
                arr = np.resize(arr, self._bins)
            self._hist = arr.astype(np.int64, copy=False)
        self.queue_draw()

    def set_level_range(self, low: int, high: int, emit_signal: bool = False):
        low = max(0, min(int(low), 65535))
        high = max(low + 1, min(int(high), 65535))
        if low == self._min_level and high == self._max_level:
            return
        self._min_level = low
        self._max_level = high
        self.queue_draw()
        if emit_signal:
            self.emit('levels-changed', self._min_level, self._max_level)

    def _hist_rect(self):
        alloc = self.get_allocation()
        margin_x = 12
        margin_top = 12
        margin_bottom = 28
        width = max(1, alloc.width - 2 * margin_x)
        height = max(1, alloc.height - (margin_top + margin_bottom))
        return margin_x, margin_top, width, height

    def _level_to_x(self, level: int, rect):
        x, _, width, _ = rect
        t = float(level) / 65535.0
        return x + t * width

    def _x_to_level(self, xpos: float, rect):
        x, _, width, _ = rect
        if width <= 0:
            return 0
        t = (xpos - x) / width
        t = max(0.0, min(1.0, t))
        return int(round(t * 65535))

    def _on_draw(self, _widget, cr: cairo.Context):
        alloc = self.get_allocation()
        cr.set_source_rgb(0.12, 0.12, 0.12)
        cr.rectangle(0, 0, alloc.width, alloc.height)
        cr.fill()

        rect = self._hist_rect()
        rx, ry, rw, rh = rect
        cr.set_source_rgb(0.18, 0.18, 0.18)
        cr.rectangle(rx, ry, rw, rh)
        cr.fill()

        max_count = int(self._hist.max()) if self._hist.size else 0
        max_count = max(1, max_count)
        bar_width = rw / float(self._bins)
        cr.set_source_rgba(0.47, 0.71, 1.0, 0.8)

        for i in range(self._bins):
            count = self._hist[i]
            height = rh * (count / max_count)
            bar_x = rx + i * bar_width
            cr.rectangle(bar_x, ry + rh - height, bar_width, height)
        cr.fill()

        # handles
        cr.set_source_rgb(1.0, 0.78, 0.0)
        cr.set_line_width(2)
        for level in (self._min_level, self._max_level):
            hx = self._level_to_x(level, rect)
            cr.move_to(hx, ry)
            cr.line_to(hx, ry + rh)
            cr.stroke()
            cr.move_to(hx - 6, ry + rh + 8)
            cr.line_to(hx + 6, ry + rh + 8)
            cr.line_to(hx, ry + rh + 18)
            cr.close_path()
            cr.fill()

        cr.set_source_rgb(0.78, 0.78, 0.78)
        cr.set_font_size(11)
        min_text = str(self._min_level)
        max_text = str(self._max_level)
        cr.move_to(rx, ry - 4)
        cr.show_text(min_text)
        max_extents = cr.text_extents(max_text)
        cr.move_to(rx + rw - max_extents.width, ry - 4)
        cr.show_text(max_text)
        return False

    def _on_button_press(self, _widget, event):
        if event.button != 1:
            return False
        rect = self._hist_rect()
        rx, ry, rw, rh = rect
        if not (rx <= event.x <= rx + rw and ry <= event.y <= ry + rh):
            return False
        x_min = self._level_to_x(self._min_level, rect)
        x_max = self._level_to_x(self._max_level, rect)
        self._active_handle = 'min' if abs(event.x - x_min) <= abs(event.x - x_max) else 'max'
        self._update_handle_from_pos(event.x, rect)
        return True

    def _on_motion(self, _widget, event):
        if not self._active_handle:
            return False
        rect = self._hist_rect()
        self._update_handle_from_pos(event.x, rect)
        return True

    def _on_button_release(self, _widget, event):
        if event.button == 1:
            self._active_handle = None
        return False

    def _update_handle_from_pos(self, xpos: float, rect):
        level = self._x_to_level(xpos, rect)
        if self._active_handle == 'min':
            self.set_level_range(level, self._max_level, emit_signal=True)
        elif self._active_handle == 'max':
            self.set_level_range(self._min_level, level, emit_signal=True)


class CameraWindow(Gtk.Window):
    def __init__(self, camera: CameraDevice):
        super().__init__(title='Hamamatsu Camera - GTK')
        self.camera = camera
        self.set_default_size(900, 700)
        self.connect('destroy', self.on_destroy)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        root.set_border_width(8)
        self.add(root)

        self.preview_area = Gtk.DrawingArea()
        self.preview_area.set_size_request(640, 480)
        self.preview_area.connect('draw', self._on_preview_draw)
        root.pack_start(self.preview_area, True, True, 0)

        self.fps_label = Gtk.Label(label='FPS: --')
        self.fps_label.set_justify(Gtk.Justification.CENTER)
        root.pack_start(self.fps_label, False, False, 0)

        # Control row
        ctrl = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        root.pack_start(ctrl, False, False, 0)

        self.btn_start = Gtk.Button(label='Start')
        self.btn_start.connect('clicked', self.on_start_stop)
        ctrl.pack_start(self.btn_start, False, False, 0)

        self.btn_capture = Gtk.Button(label='Save Burst')
        self.btn_capture.connect('clicked', self.on_save_burst)
        ctrl.pack_start(self.btn_capture, False, False, 0)

        ctrl.pack_start(Gtk.Label(label='Exposure'), False, False, 0)

        self.SMAX = 1000
        self.exp_adjustment = Gtk.Adjustment(value=1, lower=0, upper=self.SMAX, step_increment=1, page_increment=10)
        self.exp_slider = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.exp_adjustment)
        self.exp_slider.set_digits(0)
        self.exp_slider.set_hexpand(True)
        self.exp_slider.set_value_pos(Gtk.PositionType.RIGHT)
        self._exp_slider_handler = self.exp_slider.connect('value-changed', self.on_exp_slider_changed)
        ctrl.pack_start(self.exp_slider, True, True, 0)

        self.exp_entry = Gtk.Entry()
        self.exp_entry.set_width_chars(8)
        self.exp_entry.set_text('1')
        self.exp_entry.connect('activate', self.on_exp_entry_activated)
        self.exp_entry.connect('focus-out-event', self.on_exp_entry_focus_out)
        ctrl.pack_start(self.exp_entry, False, False, 0)

        # ROI + misc controls
        roi_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        root.pack_start(roi_box, False, False, 0)

        def spin(label, lower, upper, step, digits=0):
            roi_box.pack_start(Gtk.Label(label=label), False, False, 0)
            widget = Gtk.SpinButton.new_with_range(lower, upper, step)
            widget.set_digits(digits)
            roi_box.pack_start(widget, False, False, 0)
            return widget

        self.spin_hpos = spin('HPOS', 0, 16384, 1)
        self.spin_vpos = spin('VPOS', 0, 16384, 1)
        self.spin_hsize = spin('HSIZE', 2, 16384, 2)
        self.spin_vsize = spin('VSIZE', 2, 16384, 2)

        self.btn_apply_roi = Gtk.Button(label='Apply ROI')
        self.btn_apply_roi.set_tooltip_text('Apply ROI using CameraDevice.set_subarray()')
        self.btn_apply_roi.connect('clicked', self.on_apply_roi)
        roi_box.pack_start(self.btn_apply_roi, False, False, 0)

        roi_box.pack_start(Gtk.Label(label='Preview FPS'), False, False, 0)
        self.spin_preview_fps = Gtk.SpinButton.new_with_range(1, 60, 1)
        self.spin_preview_fps.set_value(15)
        self.spin_preview_fps.connect('value-changed', self.on_preview_fps_changed)
        roi_box.pack_start(self.spin_preview_fps, False, False, 0)

        roi_box.pack_start(Gtk.Label(label='Frames to save'), False, False, 0)
        self.spin_save_frames = Gtk.SpinButton.new_with_range(0, 1_000_000, 1)
        self.spin_save_frames.set_value(1000)
        roi_box.pack_start(self.spin_save_frames, False, False, 0)

        roi_box.pack_start(Gtk.Label(label='Duration (s)'), False, False, 0)
        self.spin_save_secs = Gtk.SpinButton.new_with_range(0.0, 3600.0, 0.1)
        self.spin_save_secs.set_digits(1)
        self.spin_save_secs.set_value(0.0)
        roi_box.pack_start(self.spin_save_secs, False, False, 0)

        # Histogram group
        hist_frame = Gtk.Frame(label='Histogram / Display Range')
        root.pack_start(hist_frame, False, False, 0)
        hist_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        hist_box.set_border_width(4)
        hist_frame.add(hist_box)

        self.hist_widget = HistogramWidget()
        self.hist_widget.connect('levels-changed', self.on_hist_levels_changed)
        hist_box.pack_start(self.hist_widget, True, True, 0)

        levels_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        hist_box.pack_start(levels_box, False, False, 0)

        levels_box.pack_start(Gtk.Label(label='Black level'), False, False, 0)
        self.spin_black = Gtk.SpinButton.new_with_range(0, 65534, 1)
        self.spin_black.connect('value-changed', self.on_black_level_changed)
        levels_box.pack_start(self.spin_black, False, False, 0)

        levels_box.pack_start(Gtk.Label(label='White level'), False, False, 0)
        self.spin_white = Gtk.SpinButton.new_with_range(1, 65535, 1)
        self.spin_white.connect('value-changed', self.on_white_level_changed)
        levels_box.pack_start(self.spin_white, False, False, 0)
        levels_box.pack_start(Gtk.Label(), True, True, 0)

        # Status bar
        self.status = Gtk.Statusbar()
        self.status_context = self.status.get_context_id('status')
        root.pack_start(self.status, False, False, 0)
        self._status_timeout_id = None

        # Worker + state
        self.worker = FrameWorker(self.camera, self.on_frame_ready)
        self.display_min = 0
        self.display_max = 65535
        self.hist_widget.set_level_range(self.display_min, self.display_max, emit_signal=False)
        self._block_black_spin = False
        self._block_white_spin = False
        self._set_spin_value(self.spin_black, self.display_min, attr='_block_black_spin')
        self._set_spin_value(self.spin_white, self.display_max, attr='_block_white_spin')
        self.worker.set_display_levels(self.display_min, self.display_max)
        self.worker.set_preview_fps(self.spin_preview_fps.get_value())

        self.last_rgb = None
        self.last_16 = None
        self.last_hist = None
        self._latest_pixbuf = None
        self.running = False
        self._frame_times = []
        self.save_worker = None
        self._resume_preview_after_save = False
        self._last_save_path = None
        self._last_metadata_path = None
        self._frame_error_reported = False

        self._updating_exp_slider = False
        self._updating_exp_entry = False

        # Exposure mapping setup
        try:
            emin, emax, estep, edef = self.camera.get_exposure_range()
            self.exp_min = max(emin, 1e-12)
            self.exp_max = max(emax, self.exp_min * 1.0)
            self.exp_step = estep
            self.exp_default = edef
        except Exception:
            self.exp_min = 1e-6
            self.exp_max = 10.0
            self.exp_step = 1e-6
            self.exp_default = 1.0

        self.exp_slider.set_range(0, self.SMAX)

        def exp_to_pos(exp):
            exp = max(self.exp_min, min(self.exp_max, max(exp, 1e-12)))
            span = math.log(self.exp_max) - math.log(self.exp_min)
            if span <= 0:
                return 0
            return int(round((math.log(exp) - math.log(self.exp_min)) / span * self.SMAX))

        def pos_to_exp(pos):
            t = float(pos) / float(self.SMAX)
            val = math.exp(math.log(self.exp_min) + t * (math.log(self.exp_max) - math.log(self.exp_min)))
            if self.exp_step and self.exp_step > 0:
                q = round(val / self.exp_step) * self.exp_step
                val = q
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
                self.spin_hsize.set_value(width)
            if height > 0:
                self.spin_vsize.set_value(height)
        except Exception:
            pass

        self.show_all()
        self.show_status('Ready')

    def _set_spin_value(self, spin, value, attr):
        flag = getattr(self, attr)
        setattr(self, attr, True)
        spin.set_value(value)
        setattr(self, attr, flag)

    def _set_slider_value(self, value):
        self._updating_exp_slider = True
        self.exp_slider.set_value(value)
        self._updating_exp_slider = False

    def _set_exp_entry(self, value):
        self._updating_exp_entry = True
        self.exp_entry.set_text(f"{value:.6g}")
        self._updating_exp_entry = False

    def show_status(self, message: str, timeout_ms: int = 0):
        if self._status_timeout_id:
            GLib.source_remove(self._status_timeout_id)
            self._status_timeout_id = None
        self.status.remove_all(self.status_context)
        self.status.push(self.status_context, message)
        if timeout_ms > 0:
            self._status_timeout_id = GLib.timeout_add(timeout_ms, self._clear_status)

    def _clear_status(self):
        self.status.remove_all(self.status_context)
        self._status_timeout_id = None
        return False

    def on_destroy(self, _widget):
        if self.save_worker:
            self.save_worker.request_stop()
        if self.worker:
            self.worker.stop()
        try:
            self.camera.stop()
        except Exception:
            pass
        Gtk.main_quit()

    def on_start_stop(self, _button):
        if not self.running:
            try:
                self.camera.start()
            except Exception as e:
                self.show_error('Error', f'Failed to start: {e}')
                return
            self.worker.start()
            self.btn_start.set_label('Stop')
            self.running = True
            self.show_status('Running')
        else:
            self.worker.stop()
            try:
                self.camera.stop()
            except Exception:
                pass
            self.btn_start.set_label('Start')
            self.running = False
            self.show_status('Stopped')

    def on_frame_ready(self, img_rgb, img16, hist):
        if img_rgb is None:
            if not self._frame_error_reported:
                self.show_error('Frame Error', 'Frame worker failed to read frames')
                self._frame_error_reported = True
            return
        self._frame_error_reported = False
        self.last_rgb = img_rgb
        self.last_16 = img16
        if hist is not None:
            self.last_hist = hist
            self.hist_widget.set_histogram(hist)

        self._latest_pixbuf = self._numpy_to_pixbuf(img_rgb)
        self.preview_area.queue_draw()

        now = time.time()
        self._frame_times.append(now)
        while self._frame_times and now - self._frame_times[0] > 2.0:
            self._frame_times.pop(0)
        fps = 0.0
        if len(self._frame_times) >= 2:
            dt = self._frame_times[-1] - self._frame_times[0]
            if dt > 0:
                fps = (len(self._frame_times) - 1) / dt
        h, w, _ = img_rgb.shape
        self.fps_label.set_text(f'FPS: {fps:.1f}  ({w}x{h})')

    def _numpy_to_pixbuf(self, img_rgb: np.ndarray):
        arr = np.ascontiguousarray(img_rgb, dtype=np.uint8)
        h, w, _ = arr.shape
        data = GLib.Bytes.new(arr.tobytes())
        pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(data, GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * 3)
        pixbuf._bytes_ref = data  # keep reference alive
        return pixbuf

    def _on_preview_draw(self, widget, cr: cairo.Context):
        alloc = widget.get_allocation()
        cr.set_source_rgb(0, 0, 0)
        cr.rectangle(0, 0, alloc.width, alloc.height)
        cr.fill()
        if not self._latest_pixbuf:
            return False
        pixbuf = self._latest_pixbuf
        img_w = pixbuf.get_width()
        img_h = pixbuf.get_height()
        if img_w <= 0 or img_h <= 0:
            return False
        if alloc.width <= 0 or alloc.height <= 0:
            return False
        scale = min(alloc.width / img_w, alloc.height / img_h)
        scale = max(scale, 1.0 if alloc.width >= img_w and alloc.height >= img_h else scale)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))
        draw_pixbuf = pixbuf if (new_w == img_w and new_h == img_h) else \
            pixbuf.scale_simple(new_w, new_h, GdkPixbuf.InterpType.BILINEAR)
        x = (alloc.width - new_w) / 2
        y = (alloc.height - new_h) / 2
        Gdk.cairo_set_source_pixbuf(cr, draw_pixbuf, x, y)
        cr.paint()
        return False

    def on_hist_levels_changed(self, _widget, low, high):
        self._set_display_levels(low, high, source='hist')

    def on_black_level_changed(self, spin):
        if self._block_black_spin:
            return
        value = int(spin.get_value())
        high = self.display_max
        if value >= high:
            high = min(65535, value + 1)
        self._set_display_levels(value, high, source='black_spin')

    def on_white_level_changed(self, spin):
        if self._block_white_spin:
            return
        value = int(spin.get_value())
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
            self._set_spin_value(self.spin_black, low, '_block_black_spin')
        if source != 'white_spin':
            self._set_spin_value(self.spin_white, high, '_block_white_spin')
        if self.worker:
            self.worker.set_display_levels(low, high)

    def on_preview_fps_changed(self, spin):
        try:
            self.worker.set_preview_fps(float(spin.get_value()))
            self.show_status(f'Preview FPS limited to {int(spin.get_value())}', 3000)
        except Exception:
            pass

    def on_save_burst(self, _button):
        if self.save_worker is not None:
            self.show_info('Saving', 'A save operation is already running.')
            return

        frame_limit = int(self.spin_save_frames.get_value())
        duration_limit = float(self.spin_save_secs.get_value())
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
            self.btn_start.set_label('Start')
            self.show_status('Preview paused for saving...', 3000)

        self.btn_capture.set_sensitive(False)
        self.save_worker = SaveWorker(
            self.camera,
            fname,
            frame_limit,
            duration_limit,
            progress_cb=self.on_save_progress,
            finished_cb=self.on_save_finished
        )
        self._last_metadata_path = self.save_worker.metadata_path
        self.save_worker.start()
        self.show_status('Saving frames...', 0)

    def _choose_save_file(self, default_name: str):
        dialog = Gtk.FileChooserDialog(
            title='Save burst',
            parent=self,
            action=Gtk.FileChooserAction.SAVE,
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_SAVE, Gtk.ResponseType.OK
        )
        dialog.set_current_name(default_name)
        filter_bin = Gtk.FileFilter()
        filter_bin.set_name('Binary files (*.bin)')
        filter_bin.add_pattern('*.bin')
        dialog.add_filter(filter_bin)
        filename = None
        if dialog.run() == Gtk.ResponseType.OK:
            filename = dialog.get_filename()
        dialog.destroy()
        return filename

    def on_save_progress(self, frames_saved: int, elapsed: float):
        rate = (frames_saved / elapsed) if elapsed > 0 else 0.0
        self.show_status(f'Saving... frames={frames_saved} ({rate:.1f} FPS)', 0)

    def on_save_finished(self, success: bool, error: str, frames_saved: int):
        self.btn_capture.set_sensitive(True)
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
                self.on_start_stop(None)

        self.save_worker = None

    def on_exp_slider_changed(self, scale):
        if self._updating_exp_slider:
            return
        pos = scale.get_value()
        try:
            exp = self._pos_to_exp(pos)
        except Exception as e:
            self.show_error('Mapping error', f'Failed to map slider to exposure: {e}')
            return
        self._set_exp_entry(exp)
        try:
            self.camera.set_exposure(float(exp))
            self.current_exposure = float(exp)
            self.show_status(f'Exposure set to {exp:.6g} s', 3000)
        except Exception as e:
            self._set_slider_value(self._exp_to_pos(self.current_exposure))
            self._set_exp_entry(self.current_exposure)
            self.show_error('Exposure failed', f'Failed to set exposure: {e}')

    def on_exp_entry_activated(self, _entry):
        self._apply_exp_entry()

    def on_exp_entry_focus_out(self, _entry, _event):
        self._apply_exp_entry()
        return False

    def _apply_exp_entry(self):
        if self._updating_exp_entry:
            return
        try:
            v = float(self.exp_entry.get_text())
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
        except Exception as e:
            self._set_exp_entry(self.current_exposure)
            self._set_slider_value(self._exp_to_pos(self.current_exposure))
            self.show_error('Exposure failed', f'Failed to set exposure: {e}')

    def on_apply_roi(self, _button):
        hpos = int(self.spin_hpos.get_value())
        vpos = int(self.spin_vpos.get_value())
        hsize = int(self.spin_hsize.get_value())
        vsize = int(self.spin_vsize.get_value())
        try:
            self.camera.set_subarray(hpos, vpos, hsize, vsize, mode=2)
            self.show_status(f'ROI applied: HPOS={hpos} VPOS={vpos} HSIZE={hsize} VSIZE={vsize}', 5000)
        except Exception as e:
            self.show_error('ROI failed', f'Failed to set ROI: {e}')

    def show_error(self, title: str, message: str):
        self._show_message(Gtk.MessageType.ERROR, title, message)

    def show_info(self, title: str, message: str):
        self._show_message(Gtk.MessageType.INFO, title, message)

    def _show_message(self, msg_type, title, message):
        dialog = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=msg_type,
            buttons=Gtk.ButtonsType.OK,
            text=title
        )
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()


def main():
    cam = CameraDevice(cam_index=0)
    try:
        cam.init()
    except Exception as e:
        print('Failed to initialize camera:', e)
        return

    win = CameraWindow(cam)
    Gtk.main()


if __name__ == '__main__':
    main()
