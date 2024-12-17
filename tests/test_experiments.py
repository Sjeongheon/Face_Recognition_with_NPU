import numpy as np
import pytest
import torch
from torch import nn

from face_npu_ai.models.arcface import bottleneck_IR_SE
from face_npu_ai.experiments import fold_linear_bn_to_conv, make_variant
from face_npu_ai.evaluation.agreement import compare_embeddings


class SmallBackbone(nn.Module):
    """Same structural contracts as the real checkpoint but light for CPU tests."""
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(bottleneck_IR_SE(16, 16, 2))
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(16), nn.Dropout(0.0), nn.Flatten(),
            nn.Linear(16 * 2 * 2, 8), nn.BatchNorm1d(8),
        )

    def forward(self, x):
        return self.output_layer(self.body(x))


def test_exact_variant_preserves_eval_output_and_baseline():
    torch.manual_seed(10)
    baseline = SmallBackbone().eval()
    with torch.no_grad():
        baseline.output_layer[4].running_mean.copy_(torch.randn(8))
        baseline.output_layer[4].running_var.copy_(torch.rand(8) + 0.1)
        baseline.output_layer[4].weight.copy_(torch.randn(8))
        baseline.output_layer[4].bias.copy_(torch.randn(8))
    x = torch.randn(2, 16, 4, 4)
    expected = baseline(x).detach()
    exact, report = make_variant(baseline, "exact")
    got = exact(x).flatten(1).detach()
    assert expected.shape == got.shape == (2, 8)
    torch.testing.assert_close(got, expected, atol=3e-5, rtol=3e-5)
    assert isinstance(baseline.body[0].shortcut_layer, nn.MaxPool2d)
    assert isinstance(exact.body[0].shortcut_layer.op, nn.Conv2d)
    assert {item["transformation"] for item in report} == {"pool_to_depthwise", "linear_bn_to_conv"}


def test_fold_supports_batchnorm_without_affine():
    torch.manual_seed(1)
    linear = nn.Linear(3 * 2 * 2, 5).eval()
    bn = nn.BatchNorm1d(5, affine=False).eval()
    with torch.no_grad():
        bn.running_mean.copy_(torch.randn(5))
        bn.running_var.copy_(torch.rand(5) + .1)
    x = torch.randn(4, 3, 2, 2)
    expected = bn(linear(x.flatten(1)))
    folded = fold_linear_bn_to_conv(linear, bn, in_channels=3, spatial=2)
    torch.testing.assert_close(folded(x).flatten(1), expected, atol=1e-6, rtol=1e-5)


def test_prelu_approx_is_reported_as_non_equivalent():
    baseline = SmallBackbone().eval()
    with torch.no_grad():
        baseline.body[0].res_layer[2].weight.copy_(torch.linspace(0.1, 0.9, 16))
    variant, report = make_variant(baseline, "prelu_approx")
    rows = [row for row in report if row["transformation"] == "prelu_to_leakyrelu"]
    assert len(rows) == 1
    assert rows[0]["parameters"] == 16
    assert rows[0]["slope_std"] > 0
    assert rows[0]["equivalent"] is False
    assert isinstance(variant.body[0].res_layer[2], nn.LeakyReLU)
    assert isinstance(baseline.body[0].res_layer[2], nn.PReLU)


def test_bn_workaround_preserves_fp32_and_no_se_is_ablation():
    torch.manual_seed(11)
    baseline = SmallBackbone().eval()
    exact, _ = make_variant(baseline, "exact")
    workaround, entries = make_variant(baseline, "bn_workaround")
    x = torch.randn(2, 16, 4, 4)
    torch.testing.assert_close(workaround(x), exact(x), atol=1e-5, rtol=1e-5)
    assert any(row["transformation"] == "identity_dwconv_before_bn" for row in entries)
    removed, entries = make_variant(baseline, "no_se")
    assert any(row["transformation"] == "remove_se" and not row["equivalent"] for row in entries)
    assert not torch.allclose(removed(x), exact(x))


def test_agreement_requires_matching_nonzero_embeddings():
    result = compare_embeddings(np.array([1, 2]), np.array([1, 2]))
    assert result["mean_cosine"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        compare_embeddings([1, 2], [1])
    with pytest.raises(ValueError):
        compare_embeddings([0, 0], [1, 0])
