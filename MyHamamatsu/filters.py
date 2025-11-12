"""Image filter helpers used by GTK GUI variants."""

from __future__ import annotations

import collections
from typing import Deque, Optional

import cv2
import numpy as np

_DEFAULT_KERNEL = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]], dtype=np.float32)


def gaussian_blur(img: np.ndarray, ksize: int = 5, sigma: float = 0.0) -> np.ndarray:
    """Return a blurred copy using Gaussian smoothing."""
    if img is None or img.size == 0:
        return img
    ksize = max(3, int(ksize) // 2 * 2 + 1)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def sharpen(img: np.ndarray, kernel: Optional[np.ndarray] = None) -> np.ndarray:
    """Return a sharpened copy using a high-pass kernel."""
    if img is None or img.size == 0:
        return img
    if kernel is None:
        kernel = _DEFAULT_KERNEL
    return cv2.filter2D(img, -1, kernel)


class MovingAverageFilter:
    """Maintains a trailing average over N frames."""

    def __init__(self, window: int = 5):
        self.window = max(1, int(window))
        self._buffer: Deque[np.ndarray] = collections.deque(maxlen=self.window)
        self._accum: Optional[np.ndarray] = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return frame
        arr = frame.astype(np.float32)
        if self._accum is None:
            self._accum = arr.copy()
            self._buffer.append(arr)
            return frame
        if len(self._buffer) == self._buffer.maxlen:
            oldest = self._buffer.popleft()
            self._accum -= oldest
        self._buffer.append(arr)
        if self._accum.shape != arr.shape:
            self._buffer.clear()
            self._accum = arr.copy()
            self._buffer.append(arr)
            return frame
        self._accum += arr
        avg = (self._accum / len(self._buffer)).astype(np.uint8)
        return avg

    def reset(self):
        self._buffer.clear()
        self._accum = None
