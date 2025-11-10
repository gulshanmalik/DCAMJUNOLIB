#!/usr/bin/env python3
# print_exposure_range.py -- prints exposure min/max/step from DCAM

from dcam_lib import dcam, DCAMPROP_ATTR, check_err, open_camera, dcamapi_init_simple, dcamapi_uninit_simple
from dcam_props import DCAM_IDPROP_EXPOSURETIME
from ctypes import sizeof, byref

def get_exposure_attr(hdcam):
    a = DCAMPROP_ATTR()
    a.cbSize = sizeof(DCAMPROP_ATTR)
    a.iProp  = DCAM_IDPROP_EXPOSURETIME
    a.option = 0
    err = dcam.dcamprop_getattr(hdcam, byref(a))
    check_err(err, "dcamprop_getattr")
    return a.valuemin, a.valuemax, a.valuestep, a.valuedefault

if __name__ == "__main__":
    print("Initializing DCAM API...")
    n = dcamapi_init_simple()
    print("Found cameras:", n)
    if n <= 0:
        dcamapi_uninit_simple()
        raise SystemExit("No camera found")

    cam = open_camera(0)
    try:
        vmin, vmax, vstep, vdef = get_exposure_attr(cam)
        print("Exposure property (units = seconds):")
        print("  min:     ", vmin)
        print("  max:     ", vmax)
        print("  step:    ", vstep)
        print("  default: ", vdef)
    finally:
        dcamapi_uninit_simple()