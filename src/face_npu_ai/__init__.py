"""Host-side ArcFace IR-SE model and NPU porting experiments."""

from .checkpoint import load_backbone
from .models.arcface import Backbone

__all__ = ["Backbone", "load_backbone"]
