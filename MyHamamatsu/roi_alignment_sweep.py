#!/usr/bin/env python3
"""Automated ROI alignment sweep

This script will try a grid of small VSIZE/HSIZE/HPOS/VPOS values (quantized to device
step) and attempt several write sequences. It can optionally perform a soft
hard_reset (camera.close(); camera.init()) between attempts to force a clean
state. It logs each attempt and exits early if a subarray is accepted (image
size reduces or SUBARRAYMODE readback indicates enabled).

Usage examples (from MyHamamatsu dir):
  # quick scan small V sizes and a few offsets
  python3 roi_alignment_sweep.py --hsize 256 --vsize-list 2 4 8 --hpos-list 0 2 4 --vpos-list 0 2 4 --tries 1

  # same but try a hard reset between every attempt (slower)
  python3 roi_alignment_sweep.py --hsize 256 --vsize-list 2 4 8 --hpos-list 0 2 4 --vpos-list 0 2 4 --tries 1 --force-reset

The script prints CSV-style lines to stdout with columns:
  trial,seq,hpos,hsize,vpos,vsize,mode,image_w,image_h,frame_shape,notes

Note: this does NOT power-cycle the camera. If nothing is accepted, consider
manual power-cycle or vendor support.
"""

import sys
import time
import argparse

try:
    import camera as camera_module
    CameraDevice = camera_module.CameraDevice
except Exception:
    try:
        from MyHamamatsu import camera as camera_module
        CameraDevice = camera_module.CameraDevice
    except Exception as e:
        print('Failed to import local camera module:', e, file=sys.stderr)
        raise

try:
    from dcam_props import (
        DCAM_IDPROP_SUBARRAYMODE,
        DCAM_IDPROP_SUBARRAYHPOS,
        DCAM_IDPROP_SUBARRAYHSIZE,
        DCAM_IDPROP_SUBARRAYVPOS,
        DCAM_IDPROP_SUBARRAYVSIZE,
        DCAM_IDPROP_IMAGE_WIDTH,
        DCAM_IDPROP_IMAGE_HEIGHT,
        DCAMPROP_MODE__OFF,
    )
except Exception:
    from MyHamamatsu.dcam_props import (
        DCAM_IDPROP_SUBARRAYMODE,
        DCAM_IDPROP_SUBARRAYHPOS,
        DCAM_IDPROP_SUBARRAYHSIZE,
        DCAM_IDPROP_SUBARRAYVPOS,
        DCAM_IDPROP_SUBARRAYVSIZE,
        DCAM_IDPROP_IMAGE_WIDTH,
        DCAM_IDPROP_IMAGE_HEIGHT,
        DCAMPROP_MODE__OFF,
    )

try:
    from dcam_lib import get_cap_status, DcamCapStatus
except Exception:
    from MyHamamatsu.dcam_lib import get_cap_status, DcamCapStatus


def quantize(value, step):
    if step == 0:
        return int(value)
    return int((int(value) // int(step)) * int(step))


def read_back_props(cam):
    vals = {}
    try:
        if hasattr(cam, 'get_prop'):
            vals['mode'] = cam.get_prop(DCAM_IDPROP_SUBARRAYMODE)
            vals['w'] = cam.get_prop(DCAM_IDPROP_IMAGE_WIDTH)
            vals['h'] = cam.get_prop(DCAM_IDPROP_IMAGE_HEIGHT)
            vals['hpos'] = cam.get_prop(DCAM_IDPROP_SUBARRAYHPOS)
            vals['hsize'] = cam.get_prop(DCAM_IDPROP_SUBARRAYHSIZE)
            vals['vpos'] = cam.get_prop(DCAM_IDPROP_SUBARRAYVPOS)
            vals['vsize'] = cam.get_prop(DCAM_IDPROP_SUBARRAYVSIZE)
        else:
            vals['mode'] = getattr(cam, 'subarray_mode', None)
            vals['w'] = getattr(cam, 'width', None)
            vals['h'] = getattr(cam, 'height', None)
            vals['hpos'] = getattr(cam, 'subarray_hpos', None)
            vals['hsize'] = getattr(cam, 'subarray_hsize', None)
            vals['vpos'] = getattr(cam, 'subarray_vpos', None)
            vals['vsize'] = getattr(cam, 'subarray_vsize', None)
    except Exception as e:
        vals['error'] = str(e)
    return vals


def wait_cap_ready(cam, timeout=0.5, poll=0.01):
    """Poll DCAM capture status until READY/STABLE before property writes."""
    hdcam = getattr(cam, 'hdcam', None)
    if not hdcam:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            st = get_cap_status(hdcam)
        except Exception:
            return
        if st in (DcamCapStatus.READY, DcamCapStatus.STABLE):
            return
        time.sleep(poll)


def safe_set_prop(cam, prop, val):
    try:
        wait_cap_ready(cam)
        if hasattr(cam, 'set_prop'):
            cam.set_prop(prop, float(val))
            return True, None
        if hasattr(cam, 'set_property'):
            cam.set_property(prop, float(val))
            return True, None
        if hasattr(camera_module, 'set_prop') and getattr(cam, 'hdcam', None) is not None:
            camera_module.set_prop(cam.hdcam, prop, float(val))
            return True, None
    except Exception as e:
        return False, str(e)
    return False, 'no-set-method'


def attempt_sequences(cam, hpos, hsize, vpos, vsize, sleep=0.02):
    sequences = [
        ('sizes_then_pos_then_mode', [
            (DCAM_IDPROP_SUBARRAYHSIZE, hsize),
            (DCAM_IDPROP_SUBARRAYVSIZE, vsize),
            (DCAM_IDPROP_SUBARRAYHPOS, hpos),
            (DCAM_IDPROP_SUBARRAYVPOS, vpos),
            (DCAM_IDPROP_SUBARRAYMODE, 2),
        ]),
        ('disable_sizes_pos_enable', [
            (DCAM_IDPROP_SUBARRAYMODE, DCAMPROP_MODE__OFF),
            (DCAM_IDPROP_SUBARRAYHSIZE, hsize),
            (DCAM_IDPROP_SUBARRAYVSIZE, vsize),
            (DCAM_IDPROP_SUBARRAYHPOS, hpos),
            (DCAM_IDPROP_SUBARRAYVPOS, vpos),
            (DCAM_IDPROP_SUBARRAYMODE, 2),
        ]),
    ]
    results = []
    for name, acts in sequences:
        writes = []
        for prop, val in acts:
            ok, err = safe_set_prop(cam, prop, val)
            writes.append((prop, val, ok, err))
            time.sleep(sleep)
        rb = read_back_props(cam)
        results.append({'seq': name, 'writes': writes, 'readback': rb})
    return results


def capture_shape(cam, timeout_ms=1500):
    try:
        cam.start()
        res = cam.get_frame(timeout_ms=timeout_ms)
        cam.stop()
        if res:
            img8, img16, idx, fr = res
            return getattr(img16, 'shape', None)
    except Exception:
        return None
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hsize', type=int, default=256)
    p.add_argument('--vsize-list', type=int, nargs='+', default=[2,4,8,16])
    p.add_argument('--hpos-list', type=int, nargs='+', default=[0,2,4])
    p.add_argument('--vpos-list', type=int, nargs='+', default=[0,2,4])
    p.add_argument('--sleep', type=float, default=0.02)
    p.add_argument('--tries', type=int, default=1)
    p.add_argument('--force-reset', action='store_true', help='call cam.hard_reset between attempts')
    args = p.parse_args()

    cam = CameraDevice(cam_index=0)
    try:
        cam.init()
    except Exception as e:
        print('init failed:', e, file=sys.stderr)
        return 2

    # determine step size
    step_h = 1
    step_v = 1
    try:
        if hasattr(cam, 'get_prop_attr'):
            attrh = cam.get_prop_attr(DCAM_IDPROP_SUBARRAYHSIZE)
            step_h = int(attrh.get('step', 1))
            attrv = cam.get_prop_attr(DCAM_IDPROP_SUBARRAYVSIZE)
            step_v = int(attrv.get('step', 1))
    except Exception:
        pass

    print('#trial,seq,hpos,hsize,vpos,vsize,mode,image_w,image_h,frame_shape,notes')

    trial = 0
    for t in range(args.tries):
        for vsize in args.vsize_list:
            for hpos in args.hpos_list:
                for vpos in args.vpos_list:
                    trial += 1
                    hsize_q = quantize(args.hsize, step_h)
                    vsize_q = quantize(vsize, step_v)
                    hpos_q = quantize(hpos, step_h)
                    vpos_q = quantize(vpos, step_v)

                    notes = []
                    # stop capture before writing
                    try:
                        cam.stop()
                    except Exception:
                        pass

                    if args.force_reset and hasattr(cam, 'hard_reset'):
                        try:
                            cam.hard_reset()
                            notes.append('hard_reset')
                        except Exception as e:
                            notes.append('hard_reset_err:'+str(e))

                    res = attempt_sequences(cam, hpos_q, hsize_q, vpos_q, vsize_q, sleep=args.sleep)
                    # examine last readback
                    last = res[-1]['readback'] if res else {}
                    mode = last.get('mode')
                    iw = last.get('w')
                    ih = last.get('h')
                    shape = None
                    try:
                        shape = capture_shape(cam, timeout_ms=1500)
                    except Exception:
                        shape = None

                    print('%d,%s,%d,%d,%d,%d,%s,%s,%s,%s,%s' % (
                        trial,
                        '/'.join([r['seq'] for r in res]),
                        hpos_q, hsize_q, vpos_q, vsize_q,
                        str(mode), str(iw), str(ih), str(shape), '|'.join(notes)
                    ))

                    # detailed logs
                    for r in res:
                        for w in r['writes']:
                            prop, val, ok, err = w
                            if not ok:
                                print('#WARN: write failed prop=%s val=%s err=%s' % (hex(prop), val, err))

                    # success criteria: image size changed or mode==2
                    if (isinstance(iw, (int, float)) and isinstance(ih, (int, float)) and (int(iw) < int(1920) or int(ih) < int(1200))) or (mode == 2):
                        print('#SUCCESS: accepted at trial %d' % trial)
                        try:
                            cam.close()
                        except Exception:
                            pass
                        return 0

    # finished all attempts
    try:
        cam.close()
    except Exception:
        pass
    print('#DONE: no accepted subarray found')
    return 0


if __name__ == '__main__':
    sys.exit(main())
