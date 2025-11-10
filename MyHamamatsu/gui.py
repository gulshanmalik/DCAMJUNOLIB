#!/usr/bin/env python3
"""Simple GTK GUI for Hamamatsu camera: exposure control, live preview, and save frames.

Dependencies: PyGObject (GTK3), numpy, opencv-python

Run from the MyHamamatsu folder or adjust PYTHONPATH so `camera` module is importable.
"""
import sys
import os
import threading
import time
from datetime import datetime

import gi
try:
    gi.require_version("Gtk", "3.0")
    from gi.repository import Gtk, GdkPixbuf, GLib
except ModuleNotFoundError:
    # Provide a helpful error when running inside a virtualenv without system GTK bindings
    sys.stderr.write(
        "\nThe Python 'gi' module (PyGObject) is not installed or not visible in this environment.\n"
        "On Debian/Ubuntu install the system packages: \n"
        "  sudo apt update\n"
        "  sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0\n\n"
        "If you are using a virtualenv, either run the GUI with the system Python or recreate the venv so it can access system site-packages:\n"
        "  # recreate venv with access to system site packages\n"
        "  python3 -m venv --system-site-packages .venv\n"
        "  source .venv/bin/activate\n"
        "  pip install numpy opencv-python\n\n"
        "Alternatively, installing PyGObject from pip may work but often requires build dependencies:\n"
        "  pip install pygobject pycairo\n\n"
        "After installing the system GTK bindings, run the GUI again.\n\n"
    )
    sys.exit(1)

import cv2
import numpy as np
import traceback

try:
    from camera import CameraDevice
except Exception:
    # If running as a package, try package import
    from .camera import CameraDevice


class CameraGUI(Gtk.Window):
    def __init__(self, camera: CameraDevice):
        super().__init__(title="Hamamatsu Camera")
        self.set_default_size(800, 600)

        self.camera = camera
        self.running = False
        self.worker = None
        self.last_frame = None  # store last RGB frame for saving
        self.save_dir = os.getcwd()
        # determine preview size (use camera size if available, otherwise sensible default)
        try:
            cam_w = int(self.camera.width) if getattr(self.camera, "width", None) else None
            cam_h = int(self.camera.height) if getattr(self.camera, "height", None) else None
        except Exception:
            cam_w = cam_h = None

        # cap preview size so the window won't grow out of the screen
        self.preview_w = cam_w or 640
        self.preview_h = cam_h or 480
        # limit to some reasonable maximum
        self.preview_w = min(self.preview_w, 1200)
        self.preview_h = min(self.preview_h, 900)

        # Layout
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)

        # Preview area
        self.image_widget = Gtk.Image()
        # ensure the image widget has an initial allocation so we can scale to it
        self.image_widget.set_size_request(self.preview_w, self.preview_h)
        frame_box = Gtk.Frame()
        frame_box.set_label("Live Preview")
        frame_box.add(self.image_widget)
        vbox.pack_start(frame_box, True, True, 0)

        # Controls
        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        # Start/Stop button
        self.btn_toggle = Gtk.ToggleButton(label="Start")
        self.btn_toggle.connect("toggled", self.on_toggle)
        controls.pack_start(self.btn_toggle, False, False, 0)

        # Capture button
        btn_capture = Gtk.Button(label="Capture")
        btn_capture.connect("clicked", self.on_capture)
        controls.pack_start(btn_capture, False, False, 0)

        # Recover button
        btn_recover = Gtk.Button(label="Recover")
        btn_recover.connect("clicked", self.on_recover)
        controls.pack_start(btn_recover, False, False, 0)

        # Diagnostics button
        btn_diag = Gtk.Button(label="Diagnostics")
        btn_diag.connect("clicked", self.on_diagnostics)
        controls.pack_start(btn_diag, False, False, 0)

        # Save folder chooser
        btn_folder = Gtk.Button(label="Save Folder")
        btn_folder.connect("clicked", self.on_choose_folder)
        controls.pack_start(btn_folder, False, False, 0)

        # Exposure control (float adjustment)
        adj = Gtk.Adjustment(value=1.0, lower=0.001, upper=10000.0, step_increment=0.1, page_increment=1.0)
        self.scale_exp = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=adj)
        self.scale_exp.set_digits(3)
        self.scale_exp.set_value_pos(Gtk.PositionType.RIGHT)
        self.scale_exp.set_size_request(300, -1)
        self.scale_exp.connect("value-changed", self.on_exposure_changed)
        controls.pack_start(Gtk.Label(label="Exposure:"), False, False, 0)
        controls.pack_start(self.scale_exp, True, True, 0)

        vbox.pack_start(controls, False, False, 6)

        self.connect("destroy", self.on_destroy)

    def on_toggle(self, button):
        if button.get_active():
            button.set_label("Stop")
            self.start_camera()
        else:
            button.set_label("Start")
            self.stop_camera()

    def start_camera(self):
        if not self.camera:
            return
        try:
            # start capture
            self.camera.start()
        except Exception as e:
            self.show_error(f"Failed to start camera: {e}")
            return

        self.running = True
        self.worker = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker.start()

    def stop_camera(self):
        self.running = False
        if self.camera:
            try:
                self.camera.stop()
            except Exception:
                pass

    def worker_loop(self):
        while self.running:
            try:
                    # now get both preview (8-bit) and original 16-bit frame
                    img8, img16, idx, fr = self.camera.get_frame(timeout_ms=2000)
            except Exception as e:
                GLib.idle_add(self.show_error, f"Frame read error: {e}")
                break

            if img8 is None:
                # timeout, continue
                continue

            # convert mono to RGB for display
            img_rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
            self.last_frame = img_rgb
            # keep raw 16-bit frame for saving as .bin
            self.last_frame16 = img16
            # schedule UI update
            GLib.idle_add(self.update_image, img_rgb)

        # ensure UI gets updated when loop ends
        GLib.idle_add(self.btn_toggle.set_active, False)

    def update_image(self, img_rgb: np.ndarray):
        # img_rgb is HxWx3, uint8
        h, w, _ = img_rgb.shape
        rowstride = w * 3
        # create pixbuf from bytes
        pb = GdkPixbuf.Pixbuf.new_from_data(img_rgb.tobytes(), GdkPixbuf.Colorspace.RGB, False, 8, w, h, rowstride)

        # get allocated size for the image widget (fall back to preview size)
        alloc = self.image_widget.get_allocation()
        target_w = alloc.width if alloc.width > 0 else self.preview_w
        target_h = alloc.height if alloc.height > 0 else self.preview_h

        # maintain aspect ratio and scale to fit inside target
        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        if new_w != w or new_h != h:
            try:
                pb = pb.scale_simple(new_w, new_h, GdkPixbuf.InterpType.BILINEAR)
            except Exception:
                # scaling may fail on some platforms; fall back to unscaled pb
                pass

        self.image_widget.set_from_pixbuf(pb)
        return False

    def on_exposure_changed(self, widget):
        val = widget.get_value()
        # Apply exposure change; some cameras / drivers don't accept changes while capture is running.
        try:
            self.camera.set_exposure(float(val))
        except Exception as e:
            # If change fails while running, try stop->set->start sequence once.
            tb = traceback.format_exc()
            if self.running:
                try:
                    self.stop_camera()
                    # small pause to let driver settle
                    GLib.idle_add(lambda: None)
                    self.camera.set_exposure(float(val))
                    # restart capture
                    self.camera.start()
                    self.running = True
                    # ensure toggle button reflects running
                    GLib.idle_add(self.btn_toggle.set_active, True)
                    self.show_info(f"Exposure set to {val} (restarted capture)")
                except Exception as e2:
                    tb2 = traceback.format_exc()
                    self.show_error(f"Failed to set exposure:\n{e2}\n\nTraceback:\n{tb2}")
            else:
                self.show_error(f"Failed to set exposure:\n{e}\n\nTraceback:\n{tb}")

    def on_capture(self, button):
        if self.last_frame is None:
            self.show_error("No frame available to save")
            return

        fname = datetime.now().strftime("capture_%Y%m%d_%H%M%S.png")
        path = os.path.join(self.save_dir, fname)
        # write using OpenCV (expects BGR) -> convert
        bgr = cv2.cvtColor(self.last_frame, cv2.COLOR_RGB2BGR)
        try:
            cv2.imwrite(path, bgr)
            self.show_info(f"Saved: {path}")
        except Exception as e:
            self.show_error(f"Failed to save: {e}")

    def on_recover(self, button):
        # Run a hard reset in background to avoid blocking UI
        def _work():
            try:
                self.camera.hard_reset()
                GLib.idle_add(self.show_info, "Recovery succeeded: camera reopened")
            except Exception as e:
                GLib.idle_add(self.show_error, f"Recovery failed: {e}")

        threading.Thread(target=_work, daemon=True).start()

    def on_diagnostics(self, button):
        # Query camera status and display
        try:
            status = self.camera.dump_status()
            if not status:
                self.show_info("No status available")
                return
            lines = []
            for k, v in status.items():
                lines.append(f"{k}: {v}")
            self.show_info("\n".join(lines))
        except Exception as e:
            self.show_error(f"Diagnostics failed: {e}")

    def on_choose_folder(self, button):
        dlg = Gtk.FileChooserDialog(title="Select Save Folder", parent=self, action=Gtk.FileChooserAction.SELECT_FOLDER)
        dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
        dlg.set_current_folder(self.save_dir)
        res = dlg.run()
        if res == Gtk.ResponseType.OK:
            self.save_dir = dlg.get_filename()
        dlg.destroy()

    def show_error(self, msg: str):
        dlg = Gtk.MessageDialog(transient_for=self, flags=0, message_type=Gtk.MessageType.ERROR,
                                buttons=Gtk.ButtonsType.CLOSE, text=msg)
        dlg.run()
        dlg.destroy()

    def show_info(self, msg: str):
        dlg = Gtk.MessageDialog(transient_for=self, flags=0, message_type=Gtk.MessageType.INFO,
                                buttons=Gtk.ButtonsType.OK, text=msg)
        dlg.run()
        dlg.destroy()

    def on_destroy(self, *args):
        self.stop_camera()
        if self.camera:
            try:
                self.camera.close()
            except Exception:
                pass
        Gtk.main_quit()


def main():
    # initialize camera
    cam = CameraDevice(cam_index=0)
    try:
        cam.init()
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        return

    app = CameraGUI(cam)
    app.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
