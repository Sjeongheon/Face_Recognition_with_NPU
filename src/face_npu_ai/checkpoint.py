"""Strict, CPU-first loading of the original IR-SE checkpoint."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping

import torch
from torch import nn

from .models.arcface import Backbone


def unwrap_state_dict(payload: object) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping):
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping) or not all(isinstance(k, str) for k in payload):
        raise TypeError("Checkpoint must contain a state_dict mapping of parameter names")
    keys = list(payload)
    if keys and all(k.startswith("module.") for k in keys):
        payload = {k.removeprefix("module."): v for k, v in payload.items()}
    return payload


def load_backbone(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    input_size: int = 112,
    num_layers: int = 50,
    mode: str = "ir_se",
    affine: bool = True,
) -> nn.Module:
    """Load weights strictly *before* any graph conversion.

    Requires a trusted state_dict checkpoint; arbitrary pickled Python objects
    are deliberately not deserialized. No weights are shipped with the repo.
    """
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = unwrap_state_dict(payload)
    model = Backbone(
        input_size=input_size, num_layers=num_layers, mode=mode, affine=affine
    )
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()
