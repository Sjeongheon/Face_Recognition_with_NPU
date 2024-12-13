"""Shared image preprocessing; color order is an explicit checkpoint contract.

Face detection and landmark alignment are NOT included here. Inputs must
already be aligned/cropped face images according to the checkpoint's training
pipeline. The reviewed notebook uses PIL bilinear resize and [-1, 1] scaling.
"""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class PreprocessConfig:
    size: int = 112
    color_order: str = "RGB"

    def __post_init__(self):
        if self.size not in (112, 224):
            raise ValueError("Only 112 and 224 inputs are supported by Backbone")
        if self.color_order not in ("RGB", "BGR"):
            raise ValueError("color_order must be RGB or BGR")


def preprocess_image(image: np.ndarray, config: PreprocessConfig) -> torch.Tensor:
    """Convert an HWC uint8 image in config.color_order to normalized CHW."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected HWC uint8 image with three channels")
    # PIL does not reinterpret channels: BGR input retains BGR numeric order.
    pil = Image.fromarray(image).resize((config.size, config.size), Image.Resampling.BILINEAR)
    data = np.asarray(pil, dtype=np.float32).copy()
    return torch.from_numpy(data.transpose(2, 0, 1)).div_(127.5).sub_(1.0)
