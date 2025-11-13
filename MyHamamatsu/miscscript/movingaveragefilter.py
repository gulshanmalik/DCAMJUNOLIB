#!/usr/bin/env python3
"""
rolling_moving_average_diff.py

Compute the absolute difference between two consecutive N-frame moving averages
over a sequence of TIFF images and save the result as a video.

Example:
  python rolling_moving_average_diff.py \
      --input "/path/to/frames/*.tif" \
      --window 10 \
      --output /path/to/out.mp4 \
      --fps 30 \
      --normalize  # per-frame stretch to 0..255 (optional)

Tip:
  If you don't want per-frame stretching, omit --normalize and adjust --gain.
"""

import argparse
import glob
import os
from typing import List, Tuple

from numpy import diff

import cv2
import numpy as np


def natural_key(s: str):
    """Sort helper that treats numbers in filenames numerically."""
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def list_images(pattern: str) -> List[str]:
    paths = glob.glob(pattern)
    if not paths and os.path.isdir(pattern):
        # If a directory was passed, default to common tiff extensions inside it.
        paths = glob.glob(os.path.join(pattern, "*.tif")) + \
                glob.glob(os.path.join(pattern, "*.tiff"))
    paths.sort(key=natural_key)
    return paths


def ensure_3ch_u8(img: np.ndarray) -> np.ndarray:
    """Convert grayscale/16-bit/etc. to 3-channel uint8 BGR for VideoWriter."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        pass  # already 3-channel
    else:
        raise ValueError("Unsupported image shape for video writing.")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Glob pattern or directory of TIFF frames (e.g., '/data/*.tif' or '/data').")
    ap.add_argument("--output", required=True,
                    help="Output video path, e.g., out.mp4")
    ap.add_argument("--window", type=int, default=10,
                    help="Number of frames N in the moving average (default: 10).")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="Output video FPS (default: 30).")
    ap.add_argument("--normalize", action="store_true",
                    help="Per-frame stretch of diff to 0..255.")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="If NOT normalizing, multiply diff by this gain before clipping to 0..255.")
    args = ap.parse_args()

    paths = list_images(args.input)
    if not paths:
        raise SystemExit(f"No images found for '{args.input}'")

    # Read first image to get size
    first = cv2.imread(paths[0], cv2.IMREAD_UNCHANGED)
    if first is None:
        raise SystemExit(f"Failed to read: {paths[0]}")
    H, W = first.shape[:2]

    # Video writer (use mp4v; widely supported)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.output, fourcc, args.fps, (W, H), True)
    if not vw.isOpened():
        raise SystemExit("Failed to open VideoWriter. Check codec/container support and output path.")

    # Rolling window buffers
    N = max(1, int(args.window))
    window = []
    running_sum = None  # float32 accumulator over last N frames
    prev_avg = None

    def to_float(frame: np.ndarray) -> np.ndarray:
        # Preserve dynamic range (supports 8/16-bit, grayscale or 3-ch)
        return frame.astype(np.float32)

    for idx, p in enumerate(paths, 1):
        frame = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if frame is None:
            print(f"Warning: could not read {p}, skipping.")
            continue

        if frame.shape[:2] != (H, W):
            raise SystemExit(f"All frames must be the same size. {p} is {frame.shape[:2]}, expected {(H, W)}")

        f32 = to_float(frame)

        # Initialize running sum once we know the channel count
        if running_sum is None:
            running_sum = np.zeros_like(f32, dtype=np.float32)

        # Add new frame
        window.append(f32)
        running_sum += f32

        # If window longer than N, drop oldest
        if len(window) > N:
            oldest = window.pop(0)
            running_sum -= oldest

        # Compute current average once we have N frames
        if len(window) == N:
            cur_avg = running_sum / float(N)

            if prev_avg is not None:
                diff = np.abs(cur_avg - prev_avg)

                if args.normalize:
                    max_val = float(diff.max())
                    if max_val > 0:
                        diff = (diff / max_val) * 255.0
                    else:
                        diff = np.zeros_like(diff, dtype=np.float32)
                else:
                    diff = diff * float(args.gain)
                    diff = np.clip(diff, 0, 255)

                # Convert to 3-channel uint8 for robust video writing
                # Invert so static = white, motion = black
                inv = 255.0 - diff
                out_frame = ensure_3ch_u8(inv.astype(np.uint8))

                vw.write(out_frame)

            # Update previous average for the next step (consecutive windows)
            prev_avg = cur_avg

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(paths)} frames...")

    vw.release()
    print(f"Done. Wrote: {args.output}")


if __name__ == "__main__":
    main()
