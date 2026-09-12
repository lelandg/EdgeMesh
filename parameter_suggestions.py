"""Deterministic local suggestions; callers explicitly accept any changes."""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class ParameterSuggestion:
    title: str
    settings: Mapping
    reason: str

    def __post_init__(self):
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))


def suggest_parameters(image_bgr, current_settings):
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8 or image_bgr.size == 0:
        raise ValueError("Suggestions require a nonempty uint8 BGR image")
    # Downsample proportionally for bounded local analysis.
    height, width = image_bgr.shape[:2]
    scale = min(1.0, 512 / max(height, width))
    small = cv2.resize(image_bgr, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    contrast = float(np.std(gray))
    noise = float(np.median(np.abs(gray.astype(float) - cv2.GaussianBlur(gray, (3, 3), 0).astype(float))))
    border = np.concatenate((small[0], small[-1], small[:, 0], small[:, -1])).astype(float)
    corner_spread = float(np.percentile(np.abs(border - np.median(border, axis=0)), 90))
    edge_sensitivity = int(np.clip(round(180 - contrast), 50, 190))
    candidates = [ParameterSuggestion("Tune edge sensitivity", {"sensitivity": edge_sensitivity},
        f"Image contrast is {contrast:.1f}/255; this threshold balances weak detail and noise.")]
    smoothing = "median" if noise > 5 else "bilateral"
    candidates.append(ParameterSuggestion("Preserve boundaries while smoothing", {"smoothing_method": smoothing},
        f"Local residual noise is {noise:.1f}/255; {smoothing} smoothing is a useful starting point."))
    if corner_spread < 35:
        tolerance = int(np.clip(round(corner_spread + 8), 5, 45))
        candidates.append(ParameterSuggestion("Tune background tolerance", {"background_tolerance": tolerance},
            f"Border color variation is {corner_spread:.1f}/255. This adjusts tolerance without enabling removal."))
    depth = 0.7 if contrast > 65 else 1.0
    candidates.append(ParameterSuggestion("Start with a restrained depth scale", {"depth_amount": depth},
        "A modest relative depth scale makes initial relief easier to inspect; it is not a metric depth estimate."))
    return [suggestion for suggestion in candidates
            if any(current_settings.get(key) != value for key, value in suggestion.settings.items())]
