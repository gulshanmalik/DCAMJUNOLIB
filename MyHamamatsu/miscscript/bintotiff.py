#!/usr/bin/env python3
"""Convert raw .bin files produced by qt_gui/capture_only into TIFF frames.

Typical usage:
    # Using metadata sidecar produced by qt_gui Save burst (bin inferred from JSON)
    python3 bintotiff.py --meta burst_20250101_120000.json

    # Explicit bin path with default output folder (<bin>_frames)
    python3 bintotiff.py --input burst.bin

    # Manual width/height override (if metadata missing)
    python3 bintotiff.py --input burst.bin --width 256 --height 256 --output frames/

The script reads uint16 frames (little-endian) either using geometry stored in the
metadata JSON (<bin>.json or `--meta`) or from explicit CLI arguments and writes
each frame as an individual TIFF file in the chosen output folder.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tifffile import imwrite


def parse_args():
    p = argparse.ArgumentParser(
        description='Convert raw binary frames (uint16) to TIFF images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input', help='Path to .bin file produced by saver (optional when --meta points to JSON containing binary_file)')
    p.add_argument('--meta', help='Metadata JSON describing the capture (defaults to <input>.json when omitted)')
    p.add_argument('--width', type=int, help='Frame width in pixels (overrides metadata)')
    p.add_argument('--height', type=int, help='Frame height in pixels (overrides metadata)')
    p.add_argument('--output', help='Directory to store TIFF frames (default: <bin>_frames)')
    p.add_argument('--start-frame', type=int, default=0, help='Starting frame index (default: 0)')
    p.add_argument('--max-frames', type=int, default=0, help='Maximum number of frames to export (0 = all)')
    args = p.parse_args()
    if not args.input and not args.meta:
        p.error('Provide --input or --meta (metadata JSON).')
    return args


def load_metadata(meta_arg: str | None, bin_path: Path | None):
    meta_path = None
    if meta_arg:
        meta_path = Path(meta_arg)
    elif bin_path is not None:
        meta_path = bin_path.with_suffix('.json')
    else:
        return None, None

    if not meta_path.exists():
        if meta_arg:
            raise FileNotFoundError(f'Metadata file not found: {meta_path}')
        return None, None

    with open(meta_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data, meta_path


def resolve_data_sources(args):
    bin_path = Path(args.input).expanduser() if args.input else None
    metadata = None
    meta_path = None
    try:
        metadata, meta_path = load_metadata(args.meta, bin_path)
    except Exception as e:
        raise RuntimeError(f'Failed to load metadata: {e}') from e

    if bin_path is None:
        if not metadata:
            raise ValueError('Metadata is required to determine the binary path when --input is omitted.')
        bin_file = metadata.get('binary_file')
        if not bin_file:
            raise ValueError('Metadata missing "binary_file"; specify --input manually.')
        bin_path = Path(bin_file)
        if not bin_path.is_absolute():
            base = meta_path.parent if meta_path else Path.cwd()
            bin_path = (base / bin_file).resolve()
    else:
        bin_path = bin_path.resolve()

    return bin_path, metadata, meta_path


def resolve_frame_geometry(bin_path: Path, metadata: dict | None, args) -> tuple[int, int, np.dtype]:
    width = args.width
    height = args.height

    if metadata and (width is None or height is None):
        shape = metadata.get('frame_shape')
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            height = height or int(shape[0])
            width = width or int(shape[1])

    if width is None or height is None:
        raise ValueError('Frame width/height are required (provide --width/--height or metadata JSON).')

    dtype = np.dtype('<u2')  # default little-endian uint16
    if metadata:
        dtype_str = metadata.get('numpy_dtype') or metadata.get('dtype')
        if dtype_str:
            try:
                dtype_candidate = np.dtype(dtype_str)
                # Ensure little-endian for consistent reshaping
                if dtype_candidate.kind != 'u' or dtype_candidate.itemsize != 2:
                    raise ValueError
                dtype = np.dtype('<u2')
            except Exception:
                raise ValueError(f'Unsupported dtype in metadata: {dtype_str} (expected uint16)') from None
        endianness = metadata.get('endianness')
        if endianness and str(endianness).lower() not in ('little', 'le'):
            raise ValueError(f'Unsupported endianness in metadata: {endianness} (expected little)')

    return width, height, dtype


def read_frames(bin_path: Path, width: int, height: int, dtype: np.dtype):
    frame_bytes = width * height * dtype.itemsize
    with open(bin_path, 'rb') as f:
        idx = 0
        while True:
            data = f.read(frame_bytes)
            if len(data) < frame_bytes:
                break
            arr = np.frombuffer(data, dtype=dtype).reshape((height, width))
            yield idx, arr
            idx += 1


def main():
    args = parse_args()
    bin_path, metadata, meta_path = resolve_data_sources(args)
    out_dir = Path(args.output) if args.output else bin_path.with_name(f'{bin_path.stem}_frames')
    out_dir.mkdir(parents=True, exist_ok=True)

    width, height, dtype = resolve_frame_geometry(bin_path, metadata, args)

    exported = 0
    for idx, frame in read_frames(bin_path, width, height, dtype):
        if idx < args.start_frame:
            continue
        if args.max_frames and exported >= args.max_frames:
            break
        tiff_path = out_dir / f'frame_{idx:06d}.tiff'
        imwrite(tiff_path, frame)
        exported += 1

    meta_msg = f' using metadata {meta_path}' if meta_path else ''
    print(f'Exported {exported} frame(s) to {out_dir}{meta_msg}')


if __name__ == '__main__':
    main()
