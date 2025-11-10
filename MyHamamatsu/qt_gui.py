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

    def start(self):
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._running = False

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
            img_rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
            self.frame_ready.emit(img_rgb, img16)


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

        # Controls
        ctrl = QtWidgets.QHBoxLayout()

        self.btn_start = QtWidgets.QPushButton('Start')
        self.btn_start.clicked.connect(self.on_start_stop)
        ctrl.addWidget(self.btn_start)

        self.btn_capture = QtWidgets.QPushButton('Save .bin')
        self.btn_capture.clicked.connect(self.on_save_bin)
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

        # Status bar
        self.status = self.statusBar()

        # frame worker
        self.worker = FrameWorker(self.camera)
        self.worker.frame_ready.connect(self.on_frame_ready)

        self.last_rgb = None
        self.last_16 = None
        self.running = False

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

    def on_save_bin(self):
        if self.last_16 is None:
            QtWidgets.QMessageBox.information(self, 'No frame', 'No frame available to save')
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save raw frame', f'frame_{datetime.now().strftime("%Y%m%d_%H%M%S")}.bin', 'Binary files (*.bin)')
        if not fname:
            return
        try:
            # save raw uint16 little-endian
            self.last_16.astype('<u2').tofile(fname)
            QtWidgets.QMessageBox.information(self, 'Saved', f'Saved: {fname}')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Save failed', str(e))

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
