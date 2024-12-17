"""Baseline model contracts without shipping or downloading model weights."""
from collections import OrderedDict

import numpy as np
import pytest
import torch
from torch import nn

from face_npu_ai.checkpoint import unwrap_state_dict
from face_npu_ai.models.arcface import Backbone, bottleneck_IR_SE
from face_npu_ai.preprocessing import PreprocessConfig, preprocess_image


def test_original_shortcut_and_head_contract():
    backbone = Backbone(input_size=112, num_layers=50, mode="ir_se")
    assert isinstance(backbone.body[0], bottleneck_IR_SE)
    assert isinstance(backbone.body[1].shortcut_layer, nn.MaxPool2d)
    assert isinstance(backbone.output_layer[3], nn.Linear)
    assert isinstance(backbone.output_layer[4], nn.BatchNorm1d)
    assert backbone.output_layer[3].in_features == 512 * 7 * 7


def test_checkpoint_prefix_is_removed_only_when_consistent():
    weights = OrderedDict([("module.foo", torch.tensor([1.])), ("module.bar", torch.tensor([2.]))])
    assert list(unwrap_state_dict({"state_dict": weights})) == ["foo", "bar"]
    mixed = {"module.foo": torch.tensor([1.]), "bar": torch.tensor([2.])}
    assert set(unwrap_state_dict(mixed)) == set(mixed)
    with pytest.raises(TypeError):
        unwrap_state_dict(["not a state dictionary"])


def test_preprocess_preserves_selected_numeric_channel_order():
    image = np.full((5, 5, 3), (10, 20, 30), dtype=np.uint8)
    x = preprocess_image(image, PreprocessConfig(112, "BGR"))
    assert x.shape == (3, 112, 112)
    np.testing.assert_allclose(x[:, 0, 0].numpy(), np.array([10, 20, 30]) / 127.5 - 1, atol=1e-6)
    with pytest.raises(ValueError):
        PreprocessConfig(112, "unspecified")
