import ctypes
from ctypes import c_int16, c_int32, c_double, byref, c_void_p, POINTER, Structure, cast, c_uint16
import numpy as np
import cv2
import threading
import time

from dcam_lib import (
    dcam,                       # raw ctypes DLL
    dcamapi_init_simple,
    dcamapi_uninit_simple,
    open_camera,
    close_camera,
    alloc_camera_buffer,
    release_camera_buffer,
    start_capture,
    stop_capture,
    create_wait_handle,
    close_wait_handle,
    wait_for_event,
    lock_frame,
    get_transfer_info,
    DcamCapStart,
    DcamWaitEvent,
    check_err,
    DCAMBUF_FRAME,
    DCAMPROP_ATTR,
)
from dcam_props import (
    # property IDs
    DCAM_IDPROP_EXPOSURETIME,
    DCAM_IDPROP_TRIGGERSOURCE,
    DCAM_IDPROP_IMAGE_WIDTH,
    DCAM_IDPROP_IMAGE_HEIGHT,
    DCAM_IDPROP_IMAGE_PIXELTYPE,

    # mode values
    DCAMPROP_TRIGGERSOURCE__INTERNAL,
)


# ----------------------------------------------------------------------
# Small helpers for property get/set
# ----------------------------------------------------------------------

def set_prop(hdcam, prop_id: int, value: float):
    """Wrapper around dcamprop_setvalue."""
    err = dcam.dcamprop_setvalue(hdcam, c_int32(prop_id), c_double(value))
    check_err(err, f"dcamprop_setvalue(0x{prop_id:08X})")


def get_prop(hdcam, prop_id: int) -> float:
    """Wrapper around dcamprop_getvalue."""
    val = c_double(0)
    err = dcam.dcamprop_getvalue(hdcam, c_int32(prop_id), byref(val))
    check_err(err, f"dcamprop_getvalue(0x{prop_id:08X})")
    return float(val.value)


def get_prop_attr(hdcam, prop_id: int):
    """Query DCAM property attributes (min/max/step/default).

    Returns a tuple (valuemin, valuemax, valuestep, valuedefault) as floats.
    """
    attr = DCAMPROP_ATTR()
    attr.cbSize = ctypes.sizeof(DCAMPROP_ATTR)
    attr.iProp = c_int32(prop_id).value
    attr.option = 0
    err = dcam.dcamprop_getattr(hdcam, byref(attr))
    check_err(err, f"dcamprop_getattr(0x{prop_id:08X})")
    return float(attr.valuemin), float(attr.valuemax), float(attr.valuestep), float(attr.valuedefault)


# ----------------------------------------------------------------------
# Camera wrapper class
# ----------------------------------------------------------------------


class CameraDevice:
    """Lightweight wrapper around the DCAM access used in this repo.

    Methods:
      init() -> initialize API and open camera
      start() -> start sequence capture
      stop() -> stop capture
      set_exposure(value) -> set exposure (float, units as driver expects)
      get_frame(timeout_ms=5000) -> returns (img8, idx, fr) or (None, None, None) on timeout
      close() -> release resources
    """

    def __init__(self, cam_index: int = 0, framecount: int = 16):
        self.cam_index = cam_index
        self.hdcam = None
        self.hwait = None
        self.framecount = framecount
        self.running = False
        self.width = None
        self.height = None
        self._lock = threading.Lock()

    def get_exposure_range(self):
        """Return (min, max, step, default) for exposure property, in seconds."""
        if not self.hdcam:
            raise RuntimeError("Camera not initialized")
        return get_prop_attr(self.hdcam, DCAM_IDPROP_EXPOSURETIME)

    def init(self):
        n = dcamapi_init_simple()
        if n <= 0:
            raise RuntimeError("No DCAM cameras found")

        self.hdcam = open_camera(self.cam_index)

        # default exposure and trigger left to caller; allocate buffers and wait handle
        alloc_camera_buffer(self.hdcam, self.framecount)
        self.hwait = create_wait_handle(self.hdcam)

        # determine image size
        self.width = int(get_prop(self.hdcam, DCAM_IDPROP_IMAGE_WIDTH))
        self.height = int(get_prop(self.hdcam, DCAM_IDPROP_IMAGE_HEIGHT))

    def start(self):
        if not self.hdcam:
            raise RuntimeError("Camera not initialized")
        start_capture(self.hdcam, DcamCapStart.SEQUENCE)
        self.running = True

    def stop(self):
        if self.hdcam and self.running:
            try:
                stop_capture(self.hdcam)
            except Exception:
                pass
        self.running = False

    def set_exposure(self, value: float):
        if not self.hdcam:
            raise RuntimeError("Camera not initialized")

        # Some drivers do not accept exposure changes while capture is running.
        # Perform a protected stop->set->(restart) sequence under a lock so callers
        # do not need to manage capture state. Restore the running state if possible.
        with self._lock:
            was_running = self.running
            try:
                if was_running:
                    try:
                        stop_capture(self.hdcam)
                    except Exception:
                        # If stop fails, continue and attempt to set property anyway
                        pass

                # small pause to let driver settle
                time.sleep(0.02)

                set_prop(self.hdcam, DCAM_IDPROP_EXPOSURETIME, value)

                # restart capture if it was running
                if was_running:
                    try:
                        start_capture(self.hdcam, DcamCapStart.SEQUENCE)
                    except Exception:
                        # if restart fails, leave running flag False and raise
                        self.running = False
                        raise

                # update internal running flag
                self.running = was_running

            except Exception as e:
                # try best-effort to restore capture if we stopped it
                try:
                    if was_running and not self.running:
                        start_capture(self.hdcam, DcamCapStart.SEQUENCE)
                        self.running = True
                except Exception:
                    pass
                raise

    def get_frame(self, timeout_ms: int = 5000):
        """Wait for a new frame and return an 8-bit numpy image plus frame metadata.

        Returns (img8, idx, fr) or (None, None, None) on timeout.
        """
        if not self.hwait:
            return None, None, None

        try:
            events = wait_for_event(self.hwait, DcamWaitEvent.CAP_FRAMEREADY, timeout_ms=timeout_ms)
            if not (events & DcamWaitEvent.CAP_FRAMEREADY):
                return None, None, None

            info = get_transfer_info(self.hdcam)
            idx = info.nNewestFrameIndex
            fr = lock_frame(self.hdcam, idx)

            # convert DCAMBUF_FRAME -> numpy (MONO16 -> 8-bit)
            row_words = fr.rowbytes // 2
            total_words = row_words * fr.height

            buf_type = c_uint16 * total_words
            buf_ptr = cast(fr.buf, POINTER(buf_type))

            arr = np.ctypeslib.as_array(buf_ptr.contents)
            img16 = arr.reshape(fr.height, row_words)
            img16 = img16[:, :fr.width]
            img8 = (img16 / 256).astype("uint8")

            # return both 8-bit preview and original 16-bit frame
            return img8, img16, idx, fr

        except Exception as e:
            # Provide a richer error message with traceback to help diagnose DCAM API failures
            import traceback
            tb = traceback.format_exc()

            # Try a single automatic recovery for common dcamwait_start failure
            msg = str(e)
            if ("dcamwait_start" in msg) or ("0x80000106" in msg) or ("-2147483386" in msg):
                print("DCAM wait start failed; attempting automatic recovery (restart capture / recreate wait handle)")
                try:
                    # Protect recovery sequence
                    with self._lock:
                        try:
                            stop_capture(self.hdcam)
                        except Exception:
                            pass
                        try:
                            if self.hwait:
                                close_wait_handle(self.hwait)
                        except Exception:
                            pass

                        # recreate wait handle and restart capture
                        self.hwait = create_wait_handle(self.hdcam)
                        start_capture(self.hdcam, DcamCapStart.SEQUENCE)

                    # retry waiting for a frame once
                    events = wait_for_event(self.hwait, DcamWaitEvent.CAP_FRAMEREADY, timeout_ms=timeout_ms)
                    if not (events & DcamWaitEvent.CAP_FRAMEREADY):
                        return None, None, None

                    info = get_transfer_info(self.hdcam)
                    idx = info.nNewestFrameIndex
                    fr = lock_frame(self.hdcam, idx)

                    row_words = fr.rowbytes // 2
                    total_words = row_words * fr.height
                    buf_type = c_uint16 * total_words
                    buf_ptr = cast(fr.buf, POINTER(buf_type))
                    arr = np.ctypeslib.as_array(buf_ptr.contents)
                    img16 = arr.reshape(fr.height, row_words)
                    img16 = img16[:, :fr.width]
                    img8 = (img16 / 256).astype("uint8")

                    return img8, img16, idx, fr

                except Exception as e2:
                    tb2 = traceback.format_exc()
                    raise RuntimeError(
                        f"DCAM frame read error: {e}\n\nTraceback:\n{tb}\n\nAttempted recovery failed: {e2}\n\nRecovery traceback:\n{tb2}"
                    ) from e2

            # if not a known recoverable error, raise with the original traceback
            raise RuntimeError(f"DCAM frame read error: {e}\n\nTraceback:\n{tb}") from e

    def close(self):
        # stop and release resources
        try:
            if self.hwait:
                close_wait_handle(self.hwait)
        except Exception:
            pass

        try:
            if self.hdcam:
                release_camera_buffer(self.hdcam)
                close_camera(self.hdcam)
        except Exception:
            pass

        try:
            dcamapi_uninit_simple()
        except Exception:
            pass

    def dump_status(self) -> dict:
        """Return a dictionary with some useful camera/driver status for diagnostics."""
        status = {}
        try:
            if self.hdcam:
                status['width'] = int(get_prop(self.hdcam, DCAM_IDPROP_IMAGE_WIDTH))
                status['height'] = int(get_prop(self.hdcam, DCAM_IDPROP_IMAGE_HEIGHT))
                status['exposure'] = float(get_prop(self.hdcam, DCAM_IDPROP_EXPOSURETIME))
                status['trigger'] = float(get_prop(self.hdcam, DCAM_IDPROP_TRIGGERSOURCE))
                status['pixeltype'] = int(get_prop(self.hdcam, DCAM_IDPROP_IMAGE_PIXELTYPE))
            else:
                status['error'] = 'camera handle not open'
        except Exception as e:
            status['error'] = str(e)
        return status

    def hard_reset(self):
        """Perform a full camera reset: stop capture, release buffers, close and re-open the camera,
        recreate buffers and wait handle. This is intended as a last-resort recovery path.
        """
        with self._lock:
            # stop capture if running
            try:
                stop_capture(self.hdcam)
            except Exception:
                pass

            # close wait handle
            try:
                if self.hwait:
                    close_wait_handle(self.hwait)
                    self.hwait = None
            except Exception:
                pass

            # release buffers and close camera
            try:
                if self.hdcam:
                    try:
                        release_camera_buffer(self.hdcam)
                    except Exception:
                        pass
                    try:
                        close_camera(self.hdcam)
                    except Exception:
                        pass
                    self.hdcam = None
            except Exception:
                pass

            # small pause to let driver settle
            time.sleep(0.1)

            # try re-opening camera and re-allocating
            try:
                self.hdcam = open_camera(self.cam_index)
                alloc_camera_buffer(self.hdcam, self.framecount)
                self.hwait = create_wait_handle(self.hdcam)
            except Exception as e:
                raise RuntimeError(f"hard_reset failed: {e}") from e



# ----------------------------------------------------------------------
# Main streaming demo
# ----------------------------------------------------------------------

def main():
    print("Initializing DCAM-API...")
    n = dcamapi_init_simple()
    print(f"DCAM initialized, found {n} camera(s).")
    if n <= 0:
        print("No cameras found, exiting.")
        dcamapi_uninit_simple()
        return

    hdcam = None
    hwait = None

    try:
        # --- open camera 0 ---
        print("Opening camera 0...")
        hdcam = open_camera(0)

        # --- basic settings: exposure, trigger source ---
        print("Setting exposure and trigger source...")
        # e.g. 0.01s = 10 ms (change as you like)
        set_prop(hdcam, DCAM_IDPROP_EXPOSURETIME, 1)

        # internal trigger -> free-run / internal timing
        set_prop(hdcam, DCAM_IDPROP_TRIGGERSOURCE, DCAMPROP_TRIGGERSOURCE__INTERNAL)

        # --- query image size / pixel type ---
        width  = int(get_prop(hdcam, DCAM_IDPROP_IMAGE_WIDTH))
        height = int(get_prop(hdcam, DCAM_IDPROP_IMAGE_HEIGHT))
        pixeltype = int(get_prop(hdcam, DCAM_IDPROP_IMAGE_PIXELTYPE))

        print(f"Camera image: {width} x {height}, pixeltype=0x{pixeltype:08X}")

        # --- allocate driver buffers ---
        framecount = 16  # number of frames in the ring buffer
        print(f"Allocating {framecount} frames in driver buffer...")
        alloc_camera_buffer(hdcam, framecount)

        # --- create wait handle ---
        print("Creating wait handle...")
        hwait = create_wait_handle(hdcam)

        # --- start continuous (sequence) capture ---
        print("Starting SEQUENCE capture (live)...")
        start_capture(hdcam, DcamCapStart.SEQUENCE)

        print("Starting live view (press ESC to quit)...")

        frame_index = 0
        while True:
            # wait until a frame is ready
            events = wait_for_event(
                hwait,
                DcamWaitEvent.CAP_FRAMEREADY,
                timeout_ms=5000
            )

            if not (events & DcamWaitEvent.CAP_FRAMEREADY):
                print("No frame ready (timeout or other event), events = 0x%04X" % events)
                continue

            # get newest frame index
            info = get_transfer_info(hdcam)
            idx = info.nNewestFrameIndex

            # lock that frame
            fr = lock_frame(hdcam, idx)

            # -------------------------------
            # DCAMBUF_FRAME -> NumPy (MONO16)
            # -------------------------------
            row_words   = fr.rowbytes // 2            # uint16 per row
            total_words = row_words * fr.height       # total uint16 in the buffer

            buf_type = c_uint16 * total_words
            buf_ptr  = cast(fr.buf, POINTER(buf_type))

            arr = np.ctypeslib.as_array(buf_ptr.contents)
            img16 = arr.reshape(fr.height, row_words)

            # crop to actual width (safety)
            img16 = img16[:, :fr.width]

            # convert 16-bit -> 8-bit for display
            img8 = (img16 / 256).astype("uint8")

            # show in a window
            cv2.imshow("Hamamatsu Live", img8)

            # optional: print frame info
            print(
                f"Frame {frame_index:03d}: idx={idx}, size={fr.width}x{fr.height}, "
                f"rowbytes={fr.rowbytes}, framestamp={fr.framestamp}"
            )
            frame_index += 1

            # ESC to quit
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            # ------------------------------------------------------------------
            # OPTIONAL: convert to NumPy and display (if you have numpy+opencv)
            # ------------------------------------------------------------------
            # import numpy as np
            # import cv2
            #
            # # assume MONO16 -> uint16
            # buf_type = ctypes.c_uint16 * (fr.rowbytes // 2 * fr.height)
            # raw_ptr = ctypes.cast(fr.buf, ctypes.POINTER(buf_type))
            # arr = np.ctypeslib.as_array(raw_ptr.contents)
            # # reshape: each row is rowbytes bytes, so rowbytes/2 pixels (uint16)
            # frame_np = arr.reshape(fr.height, fr.rowbytes // 2)
            # # crop to width if needed
            # frame_np = frame_np[:, :fr.width]
            # # convert to 8-bit for display
            # img_8u = (frame_np / 256).astype("uint8")
            # cv2.imshow("DCAM Live", img_8u)
            # if cv2.waitKey(1) & 0xFF == 27:  # ESC to break
            #     break

        print("Stopping capture...")
        stop_capture(hdcam)
        cv2.destroyAllWindows()

    finally:
        print("Cleaning up...")
        if hwait:
            try:
                close_wait_handle(hwait)
            except Exception as e:
                print("Error closing wait handle:", e)

        if hdcam:
            try:
                release_camera_buffer(hdcam)
            except Exception as e:
                print("Error releasing buffers:", e)
            try:
                close_camera(hdcam)
            except Exception as e:
                print("Error closing camera:", e)

        dcamapi_uninit_simple()
        print("Done.")


if __name__ == "__main__":
    main()
