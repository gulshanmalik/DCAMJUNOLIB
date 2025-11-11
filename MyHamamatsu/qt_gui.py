#!/usr/bin/env python3
"""PyQt5 GUI for Hamamatsu camera: start/stop, exposure, preview, save raw .bin frames.

Dependencies: PyQt5, numpy, opencv-python

Run from the MyHamamatsu folder so it can import `camera` module, or set PYTHONPATH accordingly.
"""
import sys
import os
import threading
from datetime import datetime

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
    frame_ready = QtCore.pyqtSignal(object, object)  # (img_rgb, img16)

    def __init__(self, camera: CameraDevice, parent=None):
        super().__init__(parent)
        self.camera = camera
        self._running = False
        self._preview_interval = 1.0 / 15.0  # default 15 FPS preview
        self._last_emit = 0.0
        self._thread = None

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
                    self.frame_ready.emit(None, None)
                    time.sleep(0.5)
                    continue

                # after recovery try once
                try:
                    res = self.camera.get_frame(timeout_ms=2000)
                except Exception:
                    self.frame_ready.emit(None, None)
                    time.sleep(0.5)
                    continue

            if not res:
                continue

            img8, img16, idx, fr = res
            now = time.time()
            if now - self._last_emit < self._preview_interval:
                continue
            self._last_emit = now
            img_rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
            self.frame_ready.emit(img_rgb, img16)


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

    @QtCore.pyqtSlot()
    def run(self):
        saved = 0
        start_time = time.time()
        try:
            self.camera.start()
            with open(self.path, 'wb') as f:
                while not self._abort:
                    if self.frame_limit > 0 and saved >= self.frame_limit:
                        break
                    if self.duration_limit > 0 and (time.time() - start_time) >= self.duration_limit:
                        break
                    res = self.camera.get_frame(timeout_ms=2000)
                    if not res:
                        continue
                    img8, img16, idx, fr = res
                    arr = np.ascontiguousarray(img16, dtype=np.uint16)
                    f.write(arr.astype('<u2').tobytes())
                    saved += 1
                    self.progress.emit(saved, time.time() - start_time)
            self.finished.emit(True, '', saved)
        except Exception as e:
            self.finished.emit(False, str(e), saved)
        finally:
            try:
                self.camera.stop()
            except Exception:
                pass

    def stop(self):
        self._abort = True


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

        # Status bar
        self.status = self.statusBar()

        # frame worker
        self.worker = FrameWorker(self.camera)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.set_preview_fps(self.spin_preview_fps.value())

        self.last_rgb = None
        self.last_16 = None
        self.running = False
        self._frame_times = []
        self.save_thread = None
        self.save_worker = None
        self._resume_preview_after_save = False
        self._last_save_path = None

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

    def on_frame_ready(self, img_rgb, img16):
        if img_rgb is None:
            QtWidgets.QMessageBox.critical(self, 'Frame Error', 'Frame worker failed to read frames')
            return
        self.last_rgb = img_rgb
        self.last_16 = img16

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
            QtWidgets.QMessageBox.information(
                self,
                'Save complete',
                f'Saved {frames_saved} frames to:\n{self._last_save_path}'
            )
            self.status.showMessage(f'Saved {frames_saved} frames', 5000)
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
