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

class RollingMovingAverageDiff:
    """Compute |avg_{t-N+2…t+1} - avg_{t-N+1…t}| with a sliding window."""

    def __init__(self, length: int = 10, normalize: bool = True):
        self.length = max(1, int(length))
        self.base_filter = MovingAverageFilter(window=self.length)
        self.prev_avg: Optional[np.ndarray] = None
        self.normalize = normalize

    def set_params(self, *, length: Optional[int] = None, normalize: Optional[bool] = None):
        if length is not None and length != self.length:
            self.length = max(1, int(length))
            self.base_filter = MovingAverageFilter(window=self.length)
            self.prev_avg = None
        if normalize is not None:
            self.normalize = bool(normalize)

    def reset(self):
        self.base_filter.reset()
        self.prev_avg = None

    def apply(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if frame is None:
            return None
        avg_frame = self.base_filter.apply(frame)
        if avg_frame is None:
            return None  # still filling the first N frames
        avg_frame = avg_frame.astype(np.float32)
        if self.prev_avg is None:
            self.prev_avg = avg_frame
            return None  # need one more average to take a diff
        diff = np.abs(avg_frame - self.prev_avg)
        self.prev_avg = avg_frame
        if self.normalize:
            max_val = diff.max()
            if max_val > 0:
                diff = (diff / max_val) * 255.0
        return diff.astype(np.uint8)

