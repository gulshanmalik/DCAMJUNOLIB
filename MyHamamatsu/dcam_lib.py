# dcam_lib.py
"""
Minimal DCAM-API 4.x ctypes bindings for:

- library init / uninit
- open / close camera
- property get/set
- buffer allocation & frame access
- capture start/stop/status
- wait for frame events

Designed to be used together with dcam_props.py.
Values are taken from dcamapi4.h (Apr 14, 2025).
"""

import sys
import ctypes
from ctypes import (
    Structure,
    c_int32,
    c_uint32,
    c_uint16,
    c_uint8,
    c_double,
    c_void_p,
    POINTER,
    byref,
    sizeof,
)
from enum import IntEnum


# ---------------------------------------------------------------------------
# Load DCAM library
# ---------------------------------------------------------------------------

def load_dcam_library():
    """
    Load the DCAM library based on the operating system.

    Adjust names/paths as needed for your installation.
    """
    if sys.platform.startswith("win"):
        # If needed, use full path here
        return ctypes.WinDLL("dcamapi")
    elif sys.platform.startswith("linux"):
        return ctypes.CDLL("libdcamapi.so")
    else:
        raise OSError("Unsupported operating system for DCAM")


dcam = load_dcam_library()


# ---------------------------------------------------------------------------
# Handles (opaque pointers in C)
# ---------------------------------------------------------------------------

HDCAM     = c_void_p
HDCAMWAIT = c_void_p
HDCAMREC  = c_void_p   # not used unless you do recording


# ---------------------------------------------------------------------------
# Errors (DCAMERR)
# ---------------------------------------------------------------------------

class DcamError(IntEnum):
    # --- status errors ---
    BUSY                = 0x80000101
    NOTREADY            = 0x80000103
    NOTSTABLE           = 0x80000104
    UNSTABLE            = 0x80000105
    NOTBUSY             = 0x80000107

    EXCLUDED            = 0x80000110

    COOLINGTROUBLE      = 0x80000302
    NOTRIGGER           = 0x80000303
    TEMPERATURE_TROUBLE = 0x80000304
    TOOFREQUENTTRIGGER  = 0x80000305

    # --- wait / capture errors ---
    ABORT               = 0x80000102
    TIMEOUT             = 0x80000106
    LOSTFRAME           = 0x80000301
    INVALIDIMAGE        = 0x80000321

    # --- initialization errors ---
    NORESOURCE          = 0x80000201
    NOMEMORY            = 0x80000203
    NOMODULE            = 0x80000204
    NODRIVER            = 0x80000205
    NOCAMERA            = 0x80000206
    NOGRABBER           = 0x80000207
    NOCOMBINATION       = 0x80000208

    INVALIDMODULE       = 0x80000211
    INVALIDCOMMPORT     = 0x80000212

    # --- calling / parameter errors ---
    INVALIDCAMERA       = 0x80000806
    INVALIDHANDLE       = 0x80000807
    INVALIDPARAM        = 0x80000808
    INVALIDVALUE        = 0x80000821
    OUTOFRANGE          = 0x80000822
    NOTWRITABLE         = 0x80000823
    NOTREADABLE         = 0x80000824
    INVALIDPROPERTYID   = 0x80000825
    NEWAPIREQUIRED      = 0x80000826
    NOPROPERTY          = 0x80000828
    INVALIDCHANNEL      = 0x80000829
    INVALIDVIEW         = 0x8000082A
    INVALIDSUBARRAY     = 0x8000082B
    ACCESSDENY          = 0x8000082C
    NOVALUETEXT         = 0x8000082D
    WRONGPROPERTYVALUE  = 0x8000082E
    FRAMEBUNDLESHOULDBEOFF = 0x80000832
    INVALIDFRAMEINDEX   = 0x80000833
    INVALIDSESSIONINDEX = 0x80000834
    NOTSUPPORT          = 0x80000F03

    # --- general / internal ---
    NONE                = 0x00000000
    INSTALLATIONINPROGRESS = 0x80000F00
    UNREACH             = 0x80000F01
    UNLOADED            = 0x80000F04
    NOCONNECTION        = 0x80000F07
    NOTIMPLEMENT        = 0x80000F02

    # --- success ---
    SUCCESS             = 0x00000001


def check_err(code: int, func_name: str = "DCAM"):
    """
    Raise RuntimeError if DCAM returned an error code.

    DCAM rules:
      - SUCCESS (1) and NONE (0) are non-errors.
      - All error codes are negative when seen as signed 32-bit.
    """
    if not isinstance(code, int):
        raise RuntimeError(f"{func_name} returned non-integer code: {code!r}")

    # DCAMERR is signed 32-bit; negative => error
    if code >= 0:
        return

    try:
        name = DcamError(code).name
    except ValueError:
        name = f"UNKNOWN(0x{code & 0xFFFFFFFF:08X})"

    raise RuntimeError(f"{func_name} failed with {code} ({name})")


# ---------------------------------------------------------------------------
# Other enums
# ---------------------------------------------------------------------------

class DcamPixelType(IntEnum):
    MONO8   = 0x00000001
    MONO16  = 0x00000002
    MONO12  = 0x00000003
    MONO12P = 0x00000005

    RGB24   = 0x00000021
    RGB48   = 0x00000022
    BGR24   = 0x00000029
    BGR48   = 0x0000002A

    NONE    = 0x00000000


class DcamCapStatus(IntEnum):
    ERROR  = 0x0000
    BUSY   = 0x0001
    READY  = 0x0002
    STABLE = 0x0003
    UNSTABLE = 0x0004


class DcamWaitEvent(IntEnum):
    CAP_TRANSFERRED = 0x0001
    CAP_FRAMEREADY  = 0x0002
    CAP_CYCLEEND    = 0x0004
    CAP_EXPOSUREEND = 0x0008
    CAP_STOPPED     = 0x0010
    CAP_RELOADFRAME = 0x0020

    REC_STOPPED     = 0x0100
    REC_WARNING     = 0x0200
    REC_MISSED      = 0x0400
    REC_DISKFULL    = 0x1000
    REC_WRITEFAULT  = 0x2000
    REC_SKIPPED     = 0x4000
    REC_WRITEFRAME  = 0x8000


class DcamCapStart(IntEnum):
    SEQUENCE = -1
    SNAP     = 0


class DcamWaitTimeout(IntEnum):
    INFINITE = 0x80000000


class DcamApiInitOption(IntEnum):
    APIVER_LATEST        = 0x00000001
    APIVER_4_0           = 0x00000400
    MULTIVIEW_DISABLE    = 0x00010002
    ENDMARK              = 0x00000000


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------

class DCAM_GUID(Structure):
    _fields_ = [
        ("Data1", c_uint32),
        ("Data2", c_uint16),
        ("Data3", c_uint16),
        ("Data4", c_uint8 * 8),
    ]


class DCAMAPI_INIT(Structure):
    _fields_ = [
        ("size",           c_int32),          # [in]
        ("iDeviceCount",   c_int32),          # [out]
        ("reserved",       c_int32),          # reserved
        ("initoptionbytes",c_int32),          # [in] size of initoption array (bytes)
        ("initoption",     POINTER(c_int32)), # [in] pointer to init options (DCAMAPI_INITOPTION)
        ("guid",           POINTER(DCAM_GUID)), # [in] optional GUID filter
    ]


class DCAMDEV_OPEN(Structure):
    _fields_ = [
        ("size", c_int32),   # [in]
        ("index", c_int32),  # [in]
        ("hdcam", HDCAM),    # [out]
    ]


class DCAMDEV_STRING(Structure):
    _fields_ = [
        ("size",     c_int32),  # [in]
        ("iString",  c_int32),  # [in] DCAM_IDSTR_xxx
        ("text",     ctypes.c_char_p),  # [in,obuf]
        ("textbytes",c_int32),  # [in]
    ]


class DCAMPROP_ATTR(Structure):
    _fields_ = [
        # input
        ("cbSize",            c_int32),
        ("iProp",             c_int32),
        ("option",            c_int32),
        ("iReserved1",        c_int32),
        # output
        ("attribute",         c_int32),
        ("iGroup",            c_int32),
        ("iUnit",             c_int32),
        ("attribute2",        c_int32),
        ("valuemin",          c_double),
        ("valuemax",          c_double),
        ("valuestep",         c_double),
        ("valuedefault",      c_double),
        ("nMaxChannel",       c_int32),
        ("iReserved3",        c_int32),
        ("nMaxView",          c_int32),
        ("iProp_NumberOfElement", c_int32),
        ("iProp_ArrayBase",       c_int32),
        ("iPropStep_Element",     c_int32),
    ]


class DCAMPROP_VALUETEXT(Structure):
    _fields_ = [
        ("cbSize",   c_int32),
        ("iProp",    c_int32),
        ("value",    c_double),
        ("text",     ctypes.c_char_p),  # buffer
        ("textbytes",c_int32),
    ]


class DCAM_TIMESTAMP(Structure):
    _fields_ = [
        ("sec",      c_uint32),
        ("microsec", c_int32),
    ]


class DCAMCAP_TRANSFERINFO(Structure):
    _fields_ = [
        ("size",             c_int32),
        ("iKind",            c_int32),  # DCAMCAP_TRANSFERKIND (we only use FRAME=0)
        ("nNewestFrameIndex",c_int32),
        ("nFrameCount",      c_int32),
    ]


class DCAMBUF_FRAME(Structure):
    _fields_ = [
        ("size",       c_int32),
        ("iKind",      c_int32),   # reserved, set 0
        ("option",     c_int32),   # reserved, set 0 (or DCAMBUF_FRAME_OPTION if you use it)
        ("iFrame",     c_int32),   # frame index
        ("buf",        c_void_p),  # pointer to image data
        ("rowbytes",   c_int32),   # bytes per row
        ("type",       c_int32),   # DCAM_PIXELTYPE
        ("width",      c_int32),
        ("height",     c_int32),
        ("left",       c_int32),
        ("top",        c_int32),
        ("timestamp",  DCAM_TIMESTAMP),
        ("framestamp", c_int32),
        ("camerastamp",c_int32),
    ]


class DCAMBUF_ATTACH(Structure):
    _fields_ = [
        ("size",        c_int32),
        ("iKind",       c_int32),      # DCAMBUF_ATTACHKIND
        ("buffer",      POINTER(c_void_p)),  # pointer to array of buffers
        ("buffercount", c_int32),
    ]


class DCAMWAIT_OPEN(Structure):
    _fields_ = [
        ("size",        c_int32),
        ("supportevent",c_int32),
        ("hwait",       HDCAMWAIT),
        ("hdcam",       HDCAM),
    ]


class DCAMWAIT_START(Structure):
    _fields_ = [
        ("size",          c_int32),
        ("eventhappened", c_int32),
        ("eventmask",     c_int32),
        ("timeout",       c_int32),
    ]


# ---------------------------------------------------------------------------
# Function prototypes
# ---------------------------------------------------------------------------

# --- Initialize / uninitialize ---
dcam.dcamapi_init.argtypes  = [POINTER(DCAMAPI_INIT)]
dcam.dcamapi_init.restype   = c_int32

dcam.dcamapi_uninit.argtypes = []
dcam.dcamapi_uninit.restype  = c_int32

# --- Device open/close/panel ---
dcam.dcamdev_open.argtypes  = [POINTER(DCAMDEV_OPEN)]
dcam.dcamdev_open.restype   = c_int32

dcam.dcamdev_close.argtypes = [HDCAM]
dcam.dcamdev_close.restype  = c_int32

# --- Property control ---
dcam.dcamprop_getvalue.argtypes = [HDCAM, c_int32, POINTER(c_double)]
dcam.dcamprop_getvalue.restype  = c_int32

dcam.dcamprop_setvalue.argtypes = [HDCAM, c_int32, c_double]
dcam.dcamprop_setvalue.restype  = c_int32

dcam.dcamprop_queryvalue.argtypes = [HDCAM, c_int32, POINTER(c_double), c_int32]
dcam.dcamprop_queryvalue.restype  = c_int32

dcam.dcamprop_getattr.argtypes = [HDCAM, POINTER(DCAMPROP_ATTR)]
dcam.dcamprop_getattr.restype  = c_int32

# --- Buffer control ---
dcam.dcambuf_alloc.argtypes   = [HDCAM, c_int32]
dcam.dcambuf_alloc.restype    = c_int32

dcam.dcambuf_release.argtypes = [HDCAM, c_int32]
dcam.dcambuf_release.restype  = c_int32

dcam.dcambuf_lockframe.argtypes = [HDCAM, POINTER(DCAMBUF_FRAME)]
dcam.dcambuf_lockframe.restype  = c_int32

dcam.dcambuf_copyframe.argtypes = [HDCAM, POINTER(DCAMBUF_FRAME)]
dcam.dcambuf_copyframe.restype  = c_int32

# --- Capture control ---
dcam.dcamcap_start.argtypes = [HDCAM, c_int32]
dcam.dcamcap_start.restype  = c_int32

dcam.dcamcap_stop.argtypes  = [HDCAM]
dcam.dcamcap_stop.restype   = c_int32

dcam.dcamcap_status.argtypes = [HDCAM, POINTER(c_int32)]
dcam.dcamcap_status.restype  = c_int32

dcam.dcamcap_transferinfo.argtypes = [HDCAM, POINTER(DCAMCAP_TRANSFERINFO)]
dcam.dcamcap_transferinfo.restype  = c_int32

# --- Wait control ---
dcam.dcamwait_open.argtypes  = [POINTER(DCAMWAIT_OPEN)]
dcam.dcamwait_open.restype   = c_int32

dcam.dcamwait_close.argtypes = [HDCAMWAIT]
dcam.dcamwait_close.restype  = c_int32

dcam.dcamwait_start.argtypes = [HDCAMWAIT, POINTER(DCAMWAIT_START)]
dcam.dcamwait_start.restype  = c_int32

dcam.dcamwait_abort.argtypes = [HDCAMWAIT]
dcam.dcamwait_abort.restype  = c_int32


# ---------------------------------------------------------------------------
# Convenience helpers (optional but nice)
# ---------------------------------------------------------------------------

def dcamapi_init_simple(init_options=None) -> int:
    """
    Initialize DCAM-API and return number of detected cameras.

    init_options: optional list of DCAMAPI_INITOPTION values.
    """
    if init_options:
        opt_array = (c_int32 * len(init_options))(*init_options)
        initoption = opt_array
        initoptionbytes = sizeof(opt_array)
    else:
        initoption = None
        initoptionbytes = 0

    apiinit = DCAMAPI_INIT()
    apiinit.size           = sizeof(DCAMAPI_INIT)
    apiinit.iDeviceCount   = 0
    apiinit.reserved       = 0
    apiinit.initoptionbytes= initoptionbytes
    apiinit.initoption     = ctypes.cast(initoption, POINTER(c_int32)) if initoption is not None else None
    apiinit.guid           = None

    err = dcam.dcamapi_init(byref(apiinit))
    check_err(err, "dcamapi_init")

    return int(apiinit.iDeviceCount)


def dcamapi_uninit_simple():
    err = dcam.dcamapi_uninit()
    check_err(err, "dcamapi_uninit")


def open_camera(index: int = 0) -> HDCAM:
    """
    Open camera by index (0-based) and return HDCAM handle.
    """
    dev_open = DCAMDEV_OPEN()
    dev_open.size  = sizeof(DCAMDEV_OPEN)
    dev_open.index = index
    dev_open.hdcam = None

    err = dcam.dcamdev_open(byref(dev_open))
    check_err(err, "dcamdev_open")

    return dev_open.hdcam


def close_camera(hdcam: HDCAM):
    if hdcam:
        err = dcam.dcamdev_close(hdcam)
        check_err(err, "dcamdev_close")


def alloc_camera_buffer(hdcam: HDCAM, framecount: int):
    err = dcam.dcambuf_alloc(hdcam, c_int32(framecount))
    check_err(err, "dcambuf_alloc")


def release_camera_buffer(hdcam: HDCAM):
    # iKind = 0 => all kinds
    err = dcam.dcambuf_release(hdcam, c_int32(0))
    check_err(err, "dcambuf_release")


def start_capture(hdcam: HDCAM, mode: DcamCapStart = DcamCapStart.SEQUENCE):
    err = dcam.dcamcap_start(hdcam, c_int32(mode))
    check_err(err, "dcamcap_start")


def stop_capture(hdcam: HDCAM):
    err = dcam.dcamcap_stop(hdcam)
    check_err(err, "dcamcap_stop")


def get_cap_status(hdcam: HDCAM) -> DcamCapStatus:
    status = c_int32(0)
    err = dcam.dcamcap_status(hdcam, byref(status))
    check_err(err, "dcamcap_status")
    return DcamCapStatus(status.value)


def get_transfer_info(hdcam: HDCAM) -> DCAMCAP_TRANSFERINFO:
    info = DCAMCAP_TRANSFERINFO()
    info.size  = sizeof(DCAMCAP_TRANSFERINFO)
    info.iKind = 0  # FRAME
    err = dcam.dcamcap_transferinfo(hdcam, byref(info))
    check_err(err, "dcamcap_transferinfo")
    return info


def create_wait_handle(hdcam: HDCAM) -> HDCAMWAIT:
    w = DCAMWAIT_OPEN()
    w.size = sizeof(DCAMWAIT_OPEN)
    w.hdcam = hdcam
    w.hwait = None
    w.supportevent = 0

    err = dcam.dcamwait_open(byref(w))
    check_err(err, "dcamwait_open")

    return w.hwait


def close_wait_handle(hwait: HDCAMWAIT):
    if hwait:
        err = dcam.dcamwait_close(hwait)
        check_err(err, "dcamwait_close")


def wait_for_event(
    hwait: HDCAMWAIT,
    event_mask: int = DcamWaitEvent.CAP_FRAMEREADY,
    timeout_ms: int = 1000,
) -> int:
    """
    Wait until one of the events in event_mask occurs or timeout.

    Returns the eventhappened bitmask.
    """
    ws = DCAMWAIT_START()
    ws.size          = sizeof(DCAMWAIT_START)
    ws.eventmask     = int(event_mask)
    ws.timeout       = int(timeout_ms)
    ws.eventhappened = 0

    err = dcam.dcamwait_start(hwait, byref(ws))
    check_err(err, "dcamwait_start")

    return ws.eventhappened


def lock_frame(hdcam: HDCAM, frame_index: int = 0) -> DCAMBUF_FRAME:
    """
    Lock a frame in the driver buffer and return DCAMBUF_FRAME with pointer, size etc.
    """
    fr = DCAMBUF_FRAME()
    fr.size   = sizeof(DCAMBUF_FRAME)
    fr.iKind  = 0
    fr.option = 0
    fr.iFrame = frame_index
    fr.buf    = None
    fr.rowbytes = 0
    fr.type   = 0
    fr.width  = 0
    fr.height = 0
    fr.left   = 0
    fr.top    = 0
    fr.timestamp = DCAM_TIMESTAMP(0, 0)
    fr.framestamp = 0
    fr.camerastamp = 0

    err = dcam.dcambuf_lockframe(hdcam, byref(fr))
    check_err(err, "dcambuf_lockframe")

    return fr


# If you run this file directly, do a tiny smoke test.
if __name__ == "__main__":
    print("Initializing DCAM-API...")
    n = 0
    try:
        n = dcamapi_init_simple()
        print(f"DCAM initialized, found {n} camera(s).")
        if n > 0:
            print("Opening camera 0...")
            cam = open_camera(0)
            print("Allocating 10 frames in driver buffer...")
            alloc_camera_buffer(cam, 10)
            print("Starting SEQUENCE capture...")
            start_capture(cam, DcamCapStart.SEQUENCE)

            print("Creating wait handle...")
            hwait = create_wait_handle(cam)
            print("Waiting for frame...")
            events = wait_for_event(hwait, DcamWaitEvent.CAP_FRAMEREADY, 5000)
            print(f"Event mask: 0x{events:04X}")

            print("Locking newest frame (index 0)...")
            fr = lock_frame(cam, 0)
            print(f"Frame: {fr.width} x {fr.height}, rowbytes={fr.rowbytes}, type=0x{fr.type:08X}")

            print("Stopping capture...")
            stop_capture(cam)
            print("Releasing buffers...")
            release_camera_buffer(cam)
            print("Closing wait handle...")
            close_wait_handle(hwait)
            print("Closing camera...")
            close_camera(cam)
    finally:
        print("Uninitializing DCAM-API...")
        dcamapi_uninit_simple()
        print("Done.")
