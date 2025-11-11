#!/usr/bin/env python3
"""Capture-only tool to measure camera FPS for a given ROI and exposure.

Usage examples:
  python3 capture_only.py --duration 10 --hsize 256 --vsize 256 --hpos 1 --vpos 1 --exposure-ms 1.0

This uses the repository's `camera.CameraDevice` and does not require PyQt5.
"""
import argparse
import time
import sys

import numpy as np
import subprocess
import shlex
import os

try:
    from camera import CameraDevice
except Exception:
    # try package import if running from repo root
    try:
        from MyHamamatsu.camera import CameraDevice
    except Exception:
        raise
try:
    import camera as camera_module
except Exception:
    try:
        from MyHamamatsu import camera as camera_module
    except Exception:
        camera_module = None


def run_capture(cam_index: int, duration: float, hpos: int, vpos: int, hsize: int, vsize: int, exposure_ms: float, subarray_mode: int = None):
    cam = CameraDevice(cam_index=cam_index)
    try:
        print('Initializing camera...')
        cam.init()
    except Exception as e:
        print('Camera init failed:', e, file=sys.stderr)
        return 2

    try:
        print('Initial status:', cam.dump_status())
        # Print subarray info if available
        try:
            print('Subarray info:', cam.get_subarray_info())
        except Exception:
            pass

        # apply ROI and exposure using either the high-level CameraDevice.set_subarray
        # or fallback to low-level set_prop writes / helper scripts.
        applied = False
        try:
            print(f'Applying ROI hpos={hpos}, vpos={vpos}, hsize={hsize}, vsize={vsize} ...')

            # Prefer using the high-level CameraDevice.set_subarray() if available
            try:
                if hasattr(cam, 'set_subarray'):
                    print('Using CameraDevice.set_subarray() to apply ROI (preferred)')
                    try:
                        cam.set_subarray(int(hpos), int(vpos), int(hsize), int(vsize), mode=(2 if subarray_mode is None else int(subarray_mode)))
                        applied = True
                        print('set_subarray() succeeded (or did not raise).')
                    except Exception as e:
                        print('CameraDevice.set_subarray() failed:', e, file=sys.stderr)
                        applied = False
                else:
                    applied = False
            except Exception as e:
                print('Error while attempting CameraDevice.set_subarray():', e, file=sys.stderr)
                applied = False

            # If high-level method not available or failed, try low-level writes as before
            if not applied and camera_module is not None and hasattr(camera_module, 'set_prop') and getattr(cam, 'hdcam', None) is not None:
                try:
                    print('Attempting direct set_prop sequence: mode -> HPOS -> HSIZE -> VPOS -> VSIZE')
                    try:
                        cam.stop()
                    except Exception:
                        pass

                    # Try to set readout speed to fastest before changing ROI (if available)
                    try:
                        readout_id = getattr(camera_module, 'DCAM_IDPROP_READOUTSPEED', None)
                        fastest_val = getattr(camera_module, 'DCAMPROP_READOUTSPEED__FASTEST', None)
                        if fastest_val is None:
                            fastest_val = 0x7FFFFFFF
                        if readout_id is not None:
                            try:
                                print('Setting READOUTSPEED to fastest', flush=True)
                                camera_module.set_prop(cam.hdcam, readout_id, float(fastest_val))
                                time.sleep(0.02)
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # Try 2x2 hardware binning if available
                    try:
                        bin_id = getattr(camera_module, 'DCAM_IDPROP_BINNING', None)
                        bin_2 = getattr(camera_module, 'DCAMPROP_BINNING__2', None) or 2
                        if bin_id is not None:
                            try:
                                print('Setting BINNING to 2x2', flush=True)
                                camera_module.set_prop(cam.hdcam, bin_id, float(bin_2))
                                time.sleep(0.02)
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # enable/disable mode according to requested subarray_mode
                    mode_to_set = 2 if subarray_mode is None else int(subarray_mode)
                    camera_module.set_prop(cam.hdcam, camera_module.DCAM_IDPROP_SUBARRAYMODE, float(mode_to_set))
                    time.sleep(0.02)
                    camera_module.set_prop(cam.hdcam, camera_module.DCAM_IDPROP_SUBARRAYHPOS, float(hpos))
                    time.sleep(0.005)
                    camera_module.set_prop(cam.hdcam, camera_module.DCAM_IDPROP_SUBARRAYHSIZE, float(hsize))
                    time.sleep(0.005)
                    camera_module.set_prop(cam.hdcam, camera_module.DCAM_IDPROP_SUBARRAYVPOS, float(vpos))
                    time.sleep(0.005)
                    camera_module.set_prop(cam.hdcam, camera_module.DCAM_IDPROP_SUBARRAYVSIZE, float(vsize))

                    # If direct writes were accepted, enforce new geometry by hard reset
                    try:
                        print('Direct writes done; performing hard_reset to apply geometry')
                        cam.hard_reset()
                        applied = True
                        print('Direct sequence applied via hard_reset')
                    except Exception as e:
                        print('Direct sequence wrote properties but hard_reset failed:', e, file=sys.stderr)
                        # Even if hard_reset failed, we consider writes attempted and will fall back to helper
                        applied = False
                except Exception as e:
                    print('Direct set_prop attempt failed:', e, file=sys.stderr)

            if not applied:
                # To allow the helper process to initialize the DCAM API, we must
                # fully close the camera in this process first (release buffers and
                # handles). After the helper completes, re-open the camera.
                try:
                    cam.close()
                except Exception:
                    pass

                # fallback helper: use roi_try_readout_bin.py (safer than missing subarray_worker)
                helper = os.path.join(os.path.dirname(__file__), 'roi_try_readout_bin.py')
                if os.path.exists(helper):
                    cmd = [sys.executable, helper, '--hpos', str(hpos), '--vpos', str(vpos), '--hsize', str(hsize), '--vsize', str(vsize)]
                else:
                    # if helper not present, fallback to roi_sequence_scanner
                    helper2 = os.path.join(os.path.dirname(__file__), 'roi_sequence_scanner.py')
                    cmd = [sys.executable, helper2, '--hpos', str(hpos), '--vpos', str(vpos), '--hsize', str(hsize), '--vsize', str(vsize)]

                try:
                    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15.0, text=True)
                    if proc.stdout:
                        print(proc.stdout)
                    if proc.returncode != 0:
                        print('Warning: helper process failed:', proc.returncode, file=sys.stderr)
                        if proc.stderr:
                            print(proc.stderr, file=sys.stderr)
                    else:
                        print('helper completed successfully')
                        applied = True
                except subprocess.TimeoutExpired:
                    print('Warning: helper timed out (driver may be hanging)', file=sys.stderr)

                # Re-open camera after helper
                try:
                    print('Re-initializing camera after helper...')
                    cam.init()
                except Exception as e:
                    print('Failed to re-init camera after helper:', e, file=sys.stderr)
                    return 2
        except Exception as e:
            print('Warning: set_subarray helper invocation failed:', e, file=sys.stderr)
        except Exception as e:
            print('Warning: set_subarray helper invocation failed:', e, file=sys.stderr)

        try:
            print(f'Setting exposure = {exposure_ms} ms')
            cam.set_exposure(exposure_ms / 1000.0)
        except Exception as e:
            print('Warning: set_exposure failed:', e, file=sys.stderr)

        print('Starting capture...')
        cam.start()

        print(f'Capturing for {duration:.1f} seconds...')
        t0 = time.time()
        timestamps = []
        shapes = []
        frames = 0
        last_print = t0
        while True:
            if time.time() - t0 >= duration:
                break
            try:
                res = cam.get_frame(timeout_ms=2000)
            except Exception as e:
                # try one recovery
                print('get_frame exception:', e, file=sys.stderr)
                try:
                    cam.hard_reset()
                    print('Attempted hard_reset, restarting capture...')
                    cam.start()
                    continue
                except Exception as e2:
                    print('Recovery failed:', e2, file=sys.stderr)
                    break

            if not res:
                # timeout or no frame
                continue
            img8, img16, idx, fr = res
            ts = time.time()
            timestamps.append(ts)
            if img16 is not None and hasattr(img16, 'shape'):
                shapes.append(img16.shape)
            frames += 1
            # every 1 second print a short status
            if ts - last_print >= 1.0:
                last_print = ts
                # compute short-term fps
                if len(timestamps) >= 2:
                    fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
                else:
                    fps = 0.0
                print(f'Captured {frames} frames, recent FPS ~ {fps:.2f}', flush=True)

        # stop capture
        print('Stopping capture...')
        cam.stop()

        # summarize
        total_time = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0.0
        avg_fps = (len(timestamps) - 1) / total_time if total_time > 0 else 0.0
        print('\nCapture summary:')
        print('  Frames:', len(timestamps))
        print(f'  Average FPS: {avg_fps:.3f}')
        if shapes:
            # most common shape
            unique, counts = np.unique(np.array(shapes, dtype=object), return_counts=True)
            # unique is an array of tuples; find most frequent
            most_idx = int(np.argmax(counts))
            print('  Most common captured shape (H, W):', unique[most_idx])
        try:
            print('Final subarray info:', cam.get_subarray_info())
        except Exception:
            pass

    finally:
        try:
            cam.close()
        except Exception:
            pass
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Capture-only throughput tester')
    p.add_argument('--cam', type=int, default=0, help='Camera index')
    p.add_argument('--duration', type=float, default=10, help='Capture duration in seconds')
    p.add_argument('--subarray-mode', type=int, choices=[2], default=None, help='Optional SUBARRAYMODE to set before ROI writes')
    p.add_argument('--hpos', type=int, default=1, help='ROI HPOS (0-based)')
    p.add_argument('--vpos', type=int, default=1, help='ROI VPOS (0-based)')
    p.add_argument('--hsize', type=int, default=400, help='ROI HSIZE')
    p.add_argument('--vsize', type=int, default=400, help='ROI VSIZE')
    p.add_argument('--exposure-ms', type=float, default=0.02, help='Exposure in milliseconds')

    args = p.parse_args()

    rc = run_capture(args.cam, args.duration, args.hpos, args.vpos, args.hsize, args.vsize, args.exposure_ms, args.subarray_mode)
    sys.exit(rc)
