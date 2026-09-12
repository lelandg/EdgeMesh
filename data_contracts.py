"""Shared image/depth/mask contracts: BGR uint8, (height, width), bool foreground."""
import cv2
import numpy as np


def as_bgr(image):
    """Return an independent contiguous BGR uint8 image; composite alpha on white."""
    value = np.asarray(image)
    if value.dtype != np.uint8 or value.size == 0:
        raise ValueError('An image must be a nonempty uint8 array.')
    if value.ndim == 2:
        value = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
    elif value.ndim == 3 and value.shape[2] == 4:
        alpha = value[:, :, 3:4].astype(np.float32) / 255
        value = np.rint(value[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    elif value.ndim != 3 or value.shape[2] != 3:
        raise ValueError('Expected grayscale, BGR or BGRA image data.')
    return np.ascontiguousarray(value).copy()


def output_shape(source_shape, requested=None):
    if requested is None or tuple(requested) == (0, 0):
        requested = source_shape[:2]
    if len(requested) != 2 or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n <= 0 for n in requested):
        raise ValueError('Image dimensions must be positive integers in (height, width) order.')
    return tuple(int(n) for n in requested)


def proportional_shape(source_shape, width):
    height, source_width = output_shape(source_shape)
    output_shape((1, 1), (1, width))
    return max(1, round(height * width / source_width)), width


def normalized_depth(depth, shape=None):
    value = np.asarray(depth, dtype=np.float32)
    if value.ndim != 2 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError('Depth must be a nonempty, finite, two-dimensional array.')
    if shape is not None:
        height, width = output_shape(value.shape, shape)
        if value.shape != (height, width):
            value = cv2.resize(value, (width, height), interpolation=cv2.INTER_CUBIC)
    value = np.maximum(value, 0)
    if value.max() == value.min():
        return np.zeros(value.shape, dtype=np.float32)
    return cv2.normalize(value, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)


def foreground_mask(mask, shape):
    value = np.asarray(mask)
    if value.ndim != 2 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError('A foreground mask must be a nonempty finite two-dimensional array.')
    if not np.isin(value, (0, 1, 255)).all():
        raise ValueError('A foreground mask must contain only boolean, 0/1 or 0/255 values.')
    height, width = output_shape(shape)
    value = (value != 0).astype(np.uint8)
    if value.shape != (height, width):
        value = cv2.resize(value, (width, height), interpolation=cv2.INTER_NEAREST)
    return value.astype(bool)
