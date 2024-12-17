import pytest
import torch
from face_npu_ai.models.loss import ArcMarginProduct


def test_arc_margin_preserves_gradient_and_ignores_negative_one_labels():
    torch.manual_seed(4)
    head = ArcMarginProduct(4, 3)
    x = torch.randn(3, 4, requires_grad=True)
    labels = torch.tensor([0, 2, -1])
    out = head(x, labels)
    loss = torch.nn.functional.cross_entropy(out[:2], labels[:2])
    loss.backward()
    assert out.shape == (3, 3)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad[:2].abs().sum() > 0
    assert head.weight.grad is not None
    with pytest.raises(ValueError):
        head(x, torch.tensor([0, 3, -1]))
