#!/usr/bin/env python3
"""Try setting readout speed and binning before applying subarray sequences.

This script will attempt to set any available READOUTSPEED and BINNING
properties (if exposed in `dcam_props`) to faster/stronger values before
attempting the same subarray sequences as the scanner. It is non-destructive
and only performs property writes and readbacks.

Usage:
    python3 roi_try_readout_bin.py --hsize 256 --vsize 256 --hpos 2 --vpos 2

Run in `MyHamamatsu` directory.
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
    from dcam_props import *
except Exception:
    from MyHamamatsu.dcam_props import *

try:
    from dcam_lib import get_cap_status, DcamCapStatus
except Exception:
    from MyHamamatsu.dcam_lib import get_cap_status, DcamCapStatus

# candidate property names to try (module symbols may or may not exist)
CAND_READOUT = [
    'DCAM_IDPROP_READOUTSPEED',
    'DCAM_IDPROP_READOUT_SPEED',
    'DCAM_IDPROP_READOUTSPEEDMODE',
]
CAND_BINNING = [
    'DCAM_IDPROP_BINNING',
    'DCAM_IDPROP_BINNING_MODE',
    'DCAM_IDPROP_BINNINGX',
    'DCAM_IDPROP_BINNINGY',
]


def find_prop(module, names):
    for n in names:
        if hasattr(module, n):
            return getattr(module, n)
    return None


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


def safe_get_prop(cam, prop):
    try:
        if hasattr(cam, 'get_prop'):
            return cam.get_prop(prop)
        if hasattr(cam, 'get_property'):
            return cam.get_property(prop)
        if hasattr(camera_module, 'get_prop') and getattr(cam, 'hdcam', None) is not None:
            return camera_module.get_prop(cam.hdcam, prop)
    except Exception as e:
        return 'ERR:' + str(e)
    return None


def wait_cap_ready(cam, timeout=0.5, poll=0.01):
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


def attempt_preconfigure(cam):
    results = {'readout': None, 'binning': None, 'notes': []}
    readout_prop = find_prop(sys.modules.get('dcam_props', globals()), CAND_READOUT)
    binning_prop = find_prop(sys.modules.get('dcam_props', globals()), CAND_BINNING)

    # try readout speed -> pick max available if attr exists
    if readout_prop is not None:
        try:
            if hasattr(cam, 'get_prop_attr'):
                attr = cam.get_prop_attr(readout_prop)
                maxv = attr.get('max', None)
                if maxv is not None:
                    ok, err = safe_set_prop(cam, readout_prop, maxv)
                    results['readout'] = (readout_prop, maxv, ok, err)
                else:
                    results['readout'] = (readout_prop, 'no-max', False, 'no max attr')
            else:
                # best-effort: try value 2 or 3
                ok, err = safe_set_prop(cam, readout_prop, 2)
                results['readout'] = (readout_prop, 2, ok, err)
        except Exception as e:
            results['readout'] = (readout_prop, 'ERR', False, str(e))
    else:
        results['notes'].append('No readout prop found')

    # try binning -> set to 2 if available
    if binning_prop is not None:
        try:
            ok, err = safe_set_prop(cam, binning_prop, 2)
            results['binning'] = (binning_prop, 2, ok, err)
        except Exception as e:
            results['binning'] = (binning_prop, 'ERR', False, str(e))
    else:
        results['notes'].append('No binning prop found')

    return results


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


def try_sequence(cam, seq_name, actions, sleep=0.02):
    rec = {'seq': seq_name, 'actions': actions[:], 'writes': [], 'readback': None, 'error': None}
    for (prop, val) in actions:
        try:
            wait_cap_ready(cam)
            if hasattr(cam, 'set_prop'):
                cam.set_prop(prop, float(val))
            elif hasattr(cam, 'set_property'):
                cam.set_property(prop, float(val))
            else:
                # try low-level camera_module helper if available
                if hasattr(camera_module, 'set_prop') and getattr(cam, 'hdcam', None) is not None:
                    camera_module.set_prop(cam.hdcam, prop, float(val))
                else:
                    raise RuntimeError('No set_prop available')
            rec['writes'].append((prop, val, 'ok'))
        except Exception as e:
            rec['writes'].append((prop, val, 'ERR: ' + str(e)))
        time.sleep(sleep)
    # readback
    rec['readback'] = read_back_props(cam)
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hsize', type=int, default=256)
    p.add_argument('--vsize', type=int, default=256)
    p.add_argument('--hpos', type=int, default=2)
    p.add_argument('--vpos', type=int, default=2)
    p.add_argument('--exposure-ms', type=float, default=1.0)
    p.add_argument('--sleep', type=float, default=0.02, help='small sleep between writes')
    args = p.parse_args()

    cam = CameraDevice(cam_index=0)
    try:
        print('Init camera')
        cam.init()
    except Exception as e:
        print('init failed:', e, file=sys.stderr)
        return 2

    # query step size for HSIZE/VSIZE -> quantize values
    step_h = 1
    step_v = 1
    try:
        if hasattr(cam, 'get_prop_attr'):
            attr = cam.get_prop_attr(DCAM_IDPROP_SUBARRAYHSIZE)
            step_h = int(attr.get('step', 1))
            attrv = cam.get_prop_attr(DCAM_IDPROP_SUBARRAYVSIZE)
            step_v = int(attrv.get('step', 1))
        else:
            # some camera wrappers store attrs in dcam_props or via get_prop
            pass
    except Exception:
        pass

    hsize_q = quantize(args.hsize, step_h)
    vsize_q = quantize(args.vsize, step_v)
    hpos_q = quantize(args.hpos, step_h)
    vpos_q = quantize(args.vpos, step_v)

    print('Attempting subarray: HPOS=%d HSIZE=%d VPOS=%d VSIZE=%d (steps H=%d V=%d)'
          % (hpos_q, hsize_q, vpos_q, vsize_q, step_h, step_v))

    # ensure capture stopped
    try:
        cam.stop()
    except Exception:
        pass

    # First try the high-level CameraDevice.set_subarray() which now performs
    # the buffer release / reallocation sequence automatically. If it succeeds,
    # there's no need to brute-force the low-level sequences below.
    if hasattr(cam, 'set_subarray'):
        try:
            print('Trying CameraDevice.set_subarray() first (preferred path)')
            cam.set_subarray(hpos_q, vpos_q, hsize_q, vsize_q, mode=2)
            print('CameraDevice.set_subarray() completed without exception; verifying frame size...')
            shape = capture_shape(cam, timeout_ms=1500)
            print('Capture shape after set_subarray:', shape)
            try:
                rb = read_back_props(cam)
                print('Readback after set_subarray:', rb)
            except Exception:
                pass
            return 0
        except Exception as e:
            print('CameraDevice.set_subarray() failed:', e)
            print('Continuing with manual property sequences...\n')

    sequences = []
    # Sequence A: example from user (mode first, then pos/size)
    sequences.append(('user_example', [
        (DCAM_IDPROP_SUBARRAYMODE, 2),
        (DCAM_IDPROP_SUBARRAYHPOS, hpos_q),
        (DCAM_IDPROP_SUBARRAYHSIZE, hsize_q),
        (DCAM_IDPROP_SUBARRAYVPOS, vpos_q),
        (DCAM_IDPROP_SUBARRAYVSIZE, vsize_q),
    ]))

    # Sequence B: sizes -> positions -> mode (common pattern)
    sequences.append(('sizes_then_pos_then_mode', [
        (DCAM_IDPROP_SUBARRAYHSIZE, hsize_q),
        (DCAM_IDPROP_SUBARRAYVSIZE, vsize_q),
        (DCAM_IDPROP_SUBARRAYHPOS, hpos_q),
        (DCAM_IDPROP_SUBARRAYVPOS, vpos_q),
        (DCAM_IDPROP_SUBARRAYMODE, 2),
    ]))

    # Sequence C: enable-first (mode=2) then sizes then positions (alternate)
    sequences.append(('mode_then_sizes_then_pos', [
        (DCAM_IDPROP_SUBARRAYMODE, 2),
        (DCAM_IDPROP_SUBARRAYHSIZE, hsize_q),
        (DCAM_IDPROP_SUBARRAYVSIZE, vsize_q),
        (DCAM_IDPROP_SUBARRAYHPOS, hpos_q),
        (DCAM_IDPROP_SUBARRAYVPOS, vpos_q),
    ]))

    # Sequence D: disable -> write sizes/pos -> enable
    sequences.append(('disable_sizes_pos_enable', [
        (DCAM_IDPROP_SUBARRAYMODE, DCAMPROP_MODE__OFF),
        (DCAM_IDPROP_SUBARRAYMODE, 2),
        (DCAM_IDPROP_SUBARRAYHSIZE, hsize_q),
        (DCAM_IDPROP_SUBARRAYVSIZE, vsize_q),
        (DCAM_IDPROP_SUBARRAYHPOS, hpos_q),
        (DCAM_IDPROP_SUBARRAYVPOS, vpos_q),
        
    ]))

    results = []
    for name, actions in sequences:
        print('\n=== Running sequence: %s ===' % name)
        r = try_sequence(cam, name, actions, sleep=args.sleep)
        results.append(r)
        # print summary
        for prop, val, status in r['writes']:
            if status == 'ok':
                print(f'  write prop={hex(prop)} val={val} -> OK')
            else:
                print(f'  write prop={hex(prop)} val={val} -> {status}')
        rb = r['readback']
        if rb is None:
            print('Readback: none')
        else:
            if 'error' in rb:
                print('Readback error:', rb['error'])
            else:
                print('mode=', rb.get('mode'), 'image(W,H)=', rb.get('w'), rb.get('h'))
                print('HPOS/HSIZE/VPOS/VSIZE =', rb.get('hpos'), rb.get('hsize'), rb.get('vpos'), rb.get('vsize'))

    print('\nNow try starting capture and grabbing a frame to confirm image size:')
    try:
        cam.start()
        res = cam.get_frame(timeout_ms=2000)
        if res:
            img8, img16, idx, fr = res
            print('Captured frame shape (H,W)=', getattr(img16, 'shape', None))
        else:
            print('No frame returned (timeout)')
    except Exception as e:
        print('Capture/get_frame failed:', e)

    finally:
        try:
            cam.stop()
        except Exception:
            pass
        try:
            cam.close()
        except Exception:
            pass
    return 0

if __name__ == '__main__':
    sys.exit(main())
