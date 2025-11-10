import sys
import ctypes
from enum import IntEnum
from ctypes import (
    Structure,
    c_int32,
    c_void_p,
    POINTER,
    byref,
    sizeof,
)


def load_dcam_library():
    """
    Load the DCAM library based on the operating system.
    Adjust the name/path for your installation if needed.
    """
    if sys.platform.startswith("win"):
        return ctypes.WinDLL("dcamapi")       # or "dcamapi.dll"
    elif sys.platform.startswith("linux"):
        return ctypes.CDLL("libdcamapi.so")   # or full path
    else:
        raise OSError("Unsupported operating system")


# Global handle to the loaded library
dcam = load_dcam_library()


class DcamError(IntEnum):
    """
    Small subset of DCAMERR_* from dcamapi4.h.
    Add more as you need them.
    """

    # Success codes
    NONE    = 0x00000000      # DCAMERR_NONE
    SUCCESS = 0x00000001      # DCAMERR_SUCCESS

    # Status errors
    BUSY         = 0x80000101  # DCAMERR_BUSY
    NOTREADY     = 0x80000103  # DCAMERR_NOTREADY
    TIMEOUT      = 0x80000106  # DCAMERR_TIMEOUT
    ABORT        = 0x80000102  # DCAMERR_ABORT

    # Initialization / resource errors
    NORESOURCE   = 0x80000201  # DCAMERR_NORESOURCE
    NOMEMORY     = 0x80000203  # DCAMERR_NOMEMORY
    NODRIVER     = 0x80000205  # DCAMERR_NODRIVER
    NOCAMERA     = 0x80000206  # DCAMERR_NOCAMERA
    NOGRABBER    = 0x80000207  # DCAMERR_NOGRABBER

    # Calling/parameter errors
    INVALIDCAMERA   = 0x80000806  # DCAMERR_INVALIDCAMERA
    INVALIDHANDLE   = 0x80000807  # DCAMERR_INVALIDHANDLE
    INVALIDPARAM    = 0x80000808  # DCAMERR_INVALIDPARAM
    INVALIDVALUE    = 0x80000821  # DCAMERR_INVALIDVALUE
    OUTOFRANGE      = 0x80000822  # DCAMERR_OUTOFRANGE
    NOTSUPPORT      = 0x80000f03  # DCAMERR_NOTSUPPORT


def check_err(code: int, func_name: str = "DCAM"):
    """
    Raise a Python exception if DCAM returns an error code.

    DCAMERR from the header:
      - SUCCESS (1) and NONE (0) are non-error.
      - All error codes are negative (when seen as signed int32).
    """
    # If code is None or not an int, something strange happened
    if not isinstance(code, int):
        raise RuntimeError(f"{func_name} returned non-integer code: {code!r}")

    # In DCAM, success codes are >= 0, errors are < 0 (signed 32-bit)
    if code >= 0:
        return

    # Try to decode it into a DcamError name
    try:
        name = DcamError(code).name
    except ValueError:
        name = f"UNKNOWN(0x{code & 0xFFFFFFFF:08X})"  # show full hex

    raise RuntimeError(f"{func_name} failed with {code} ({name})")

class DCAMAPI_INIT(Structure):
    """
    Python version of DCAMAPI_INIT from dcamapi4.h
    """
    _fields_ = [
        ("size",           c_int32),          # int32 size;
        ("iDeviceCount",   c_int32),          # int32 iDeviceCount;
        ("reserved",       c_int32),          # int32 reserved;
        ("initoptionbytes",c_int32),          # int32 initoptionbytes;
        ("initoption",     POINTER(c_int32)), # const int32* initoption;
        ("guid",           c_void_p),         # const DCAM_GUID* guid; (we pass None)
    ]

# DCAMERR dcamapi_init(DCAMAPI_INIT* param);
dcam.dcamapi_init.argtypes = [POINTER(DCAMAPI_INIT)]
dcam.dcamapi_init.restype  = c_int32

# DCAMERR dcamapi_uninit(void);
dcam.dcamapi_uninit.argtypes = []
dcam.dcamapi_uninit.restype  = c_int32

def dcamapi_init() -> int:
    """
    Initialize DCAM-API and return number of detected cameras.
    """
    apiinit = DCAMAPI_INIT()
    apiinit.size           = sizeof(DCAMAPI_INIT)
    apiinit.iDeviceCount   = 0
    apiinit.reserved       = 0
    apiinit.initoptionbytes= 0
    apiinit.initoption     = None   # no init options
    apiinit.guid           = None   # no specific GUID

    err = dcam.dcamapi_init(byref(apiinit))
    check_err(err, "dcamapi_init")

    return int(apiinit.iDeviceCount)


def dcamapi_uninit():
    """
    Shut down DCAM-API.
    """
    err = dcam.dcamapi_uninit()
    check_err(err, "dcamapi_uninit")

if __name__ == "__main__":
    print("Initializing DCAM-API...")
    try:
        n = dcamapi_init()
        print(f"DCAM initialized, found {n} camera(s).")
    finally:
        print("Uninitializing DCAM-API...")
        dcamapi_uninit()
        print("Done.")
