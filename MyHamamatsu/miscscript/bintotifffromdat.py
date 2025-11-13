#!/usr/bin/env python3
"""Convert raw .bin files to TIFF frames using .dat metadata.

Usage examples:
    python3 bintotiff_dat.py --dat capture.dat
    python3 bintotiff_dat.py --input burst.bin --dat capture.dat
    python3 bintotiff_dat.py --input burst.bin --width 256 --height 256
"""

import argparse
from pathlib import Path

import numpy as np
from tifffile import imwrite


# -------------------------------------------------------------------------
# Parse CLI arguments
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Convert raw binary frames with .dat metadata to TIFF images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", help="Path to .bin file (optional if .dat contains path)")
    p.add_argument("--dat", help="Path to .dat metadata file")
    p.add_argument("--width", type=int, help="Frame width override")
    p.add_argument("--height", type=int, help="Frame height override")
    p.add_argument("--output", help="Directory for TIFF frames")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()

    if not args.input and not args.dat:
        p.error("Provide --input or --dat")
    return args


# -------------------------------------------------------------------------
# Parse .dat file (key value per line)
# -------------------------------------------------------------------------
def load_dat_metadata(dat_path: Path):
    meta = {}
    with open(dat_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                key, value = line.split("\t", 1)
            elif " " in line:
                key, value = line.split(" ", 1)
            else:
                continue
            meta[key.strip()] = value.strip()
    return meta


# -------------------------------------------------------------------------
# Extract geometry + dtype
# -------------------------------------------------------------------------
def resolve_frame_geometry(metadata: dict, args):
    # overrides
    width = args.width
    height = args.height

    # Try metadata-based values
    if width is None:
        w = metadata.get("cam/camera_attributes/image_width")
        if w:
            width = int(w)

    if height is None:
        h = metadata.get("cam/camera_attributes/image_height")
        if h:
            height = int(h)

    if width is None or height is None:
        raise ValueError("Frame width/height missing. Provide --width & --height.")

    # dtype from .dat (expected "<u2")
    dtype_str = metadata.get("save/frame/dtype", "<u2").strip()
    try:
        dtype = np.dtype(dtype_str)
    except Exception:
        dtype = np.dtype("<u2")  # fallback

    return width, height, dtype


# -------------------------------------------------------------------------
# Yield frames from .bin
# -------------------------------------------------------------------------
def read_frames(bin_path: Path, width: int, height: int, dtype: np.dtype):
    frame_bytes = width * height * dtype.itemsize
    with open(bin_path, "rb") as f:
        idx = 0
        while True:
            data = f.read(frame_bytes)
            if len(data) < frame_bytes:
                break
            frame = np.frombuffer(data, dtype=dtype).reshape((height, width))
            yield idx, frame
            idx += 1


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    args = parse_args()

    dat_path = Path(args.dat) if args.dat else None
    metadata = load_dat_metadata(dat_path) if dat_path else {}

    # Determine .bin path
    if args.input:
        bin_path = Path(args.input).expanduser()
    else:
        # Try metadata fallback
        bin_path_str = metadata.get("save/path")
        if not bin_path_str:
            raise ValueError("No --input and .dat does not contain save/path.")
        bin_path = Path(bin_path_str)

    bin_path = bin_path.resolve()

    # Output directory
    out_dir = Path(args.output) if args.output else bin_path.with_name(f"{bin_path.stem}_frames")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine geometry
    width, height, dtype = resolve_frame_geometry(metadata, args)

    print(f"Converting BIN → TIFF")
    print(f"Input:   {bin_path}")
    print(f"Metadata: {dat_path}")
    print(f"Size:    {width}×{height}, dtype={dtype}")
    print(f"Output folder: {out_dir}")

    # Export frames
    exported = 0
    for idx, frame in read_frames(bin_path, width, height, dtype):
        if idx < args.start_frame:
            continue
        if args.max_frames and exported >= args.max_frames:
            break

        out_path = out_dir / f"frame_{idx:06d}.tiff"
        imwrite(out_path, frame)
        exported += 1

        if exported % 500 == 0:
            print(f"Exported {exported} frames...")

    print(f"Done! Exported {exported} frame(s) to {out_dir}")


if __name__ == "__main__":
    main()
