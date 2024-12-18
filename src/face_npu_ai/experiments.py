"""Isolated FP32 graph experiments and explicitly non-equivalent ablations.

All transformations happen on a deepcopy of the already-loaded, eval-mode baseline.
A mathematically equivalent FP32 transformation is *not* a guarantee of identical
floating-point rounding or equivalent INT8/NPU execution.
"""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .models.arcface import SEModule, bottleneck_IR, bottleneck_IR_SE

VARIANTS = ("baseline", "exact", "prelu_approx", "bn_workaround", "no_se")


class DepthwiseSubsample(nn.Module):
    """For the source backbone's 1x1, unpadded MaxPool shortcut only."""

    def __init__(self, channels: int, stride: int, *, device=None, dtype=None):
        super().__init__()
        if stride < 1:
            raise ValueError("stride must be positive")
        if stride == 1:
            self.op = nn.Identity()
        else:
            self.op = nn.Conv2d(
                channels, channels, 1, stride=stride, groups=channels,
                bias=False, device=device, dtype=dtype,
            )
            with torch.no_grad():
                self.op.weight.fill_(1.0)
            self.op.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class PReLUAsOps(nn.Module):
    """Inference-only PReLU decomposition preserving learned channel slopes."""

    def __init__(self, prelu: nn.PReLU):
        super().__init__()
        slope = prelu.weight.detach().clone().reshape(1, -1, 1, 1)
        self.register_buffer("slope", slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positive = F.relu(x)
        negative = x - positive
        return positive + self.slope * negative


def _same_pair(value, expected):
    pair = (value, value) if isinstance(value, int) else tuple(value)
    return pair == (expected, expected)


def _replace_shortcuts(model: nn.Module, report: list[dict]) -> None:
    for name, block in model.named_modules():
        if not isinstance(block, (bottleneck_IR, bottleneck_IR_SE)):
            continue
        pool = block.shortcut_layer
        if not isinstance(pool, nn.MaxPool2d):
            continue
        if not (
            _same_pair(pool.kernel_size, 1)
            and _same_pair(pool.padding, 0)
            and _same_pair(pool.dilation, 1)
            and not pool.ceil_mode
            and not pool.return_indices
        ):
            raise ValueError(f"Unsupported shortcut pooling parameters at {name}")
        stride = pool.stride if pool.stride is not None else pool.kernel_size
        if isinstance(stride, tuple):
            if stride[0] != stride[1]:
                raise ValueError(f"Non-square stride at {name}")
            stride = stride[0]
        pre_bn = block.res_layer[0]
        if not isinstance(pre_bn, nn.BatchNorm2d):
            raise TypeError(f"Unexpected residual block structure at {name}")
        ref = pre_bn.running_mean if pre_bn.running_mean is not None else pre_bn.weight
        block.shortcut_layer = DepthwiseSubsample(
            pre_bn.num_features, stride, device=ref.device, dtype=ref.dtype,
        )
        report.append({"transformation": "pool_to_depthwise", "module": name, "stride": stride})


def fold_linear_bn_to_conv(
    linear: nn.Linear, bn: nn.BatchNorm1d, *, in_channels: int, spatial: int
) -> nn.Conv2d:
    """Fold eval Linear + BatchNorm1d into a spatial Conv2d, retaining BN."""
    if linear.training or bn.training:
        raise ValueError("Set Linear and BatchNorm1d to eval() before folding")
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError("BatchNorm1d requires running statistics for eval-time folding")
    if linear.in_features != in_channels * spatial * spatial or linear.out_features != bn.num_features:
        raise ValueError("Linear/BN shapes do not match the expected spatial head")
    if spatial <= 0:
        raise ValueError("spatial must be positive")
    conv = nn.Conv2d(
        in_channels, linear.out_features, spatial, bias=True,
        device=linear.weight.device, dtype=linear.weight.dtype,
    )
    weight = linear.weight.detach().reshape(linear.out_features, in_channels, spatial, spatial)
    bias = linear.bias.detach() if linear.bias is not None else weight.new_zeros(linear.out_features)
    gamma = bn.weight.detach() if bn.affine else torch.ones_like(bn.running_mean)
    beta = bn.bias.detach() if bn.affine else torch.zeros_like(bn.running_mean)
    scale = gamma / torch.sqrt(bn.running_var.detach() + bn.eps)
    with torch.no_grad():
        conv.weight.copy_(weight * scale[:, None, None, None])
        conv.bias.copy_((bias - bn.running_mean.detach()) * scale + beta)
    return conv.eval()


def _convert_output_head(model: nn.Module, report: list[dict]) -> None:
    head = getattr(model, "output_layer", None)
    expected = (nn.BatchNorm2d, nn.Dropout, nn.Flatten, nn.Linear, nn.BatchNorm1d)
    if not isinstance(head, nn.Sequential) or len(head) != 5 or any(
        not isinstance(actual, cls) for actual, cls in zip(head, expected)
    ):
        raise ValueError("Expected original [BN2d, Dropout, Flatten, Linear, BN1d] output head")
    if model.training:
        raise ValueError("Model must be in eval mode before folding")
    linear, bn = head[3], head[4]
    channels = head[0].num_features
    spatial_sq, rem = divmod(linear.in_features, channels)
    spatial = int(spatial_sq ** 0.5)
    if rem or spatial * spatial != spatial_sq:
        raise ValueError("Cannot infer square spatial output of ArcFace head")
    conv = fold_linear_bn_to_conv(linear, bn, in_channels=channels, spatial=spatial)
    head[2] = nn.Identity()
    head[3] = conv
    head[4] = nn.Identity()
    report.append({"transformation": "linear_bn_to_conv", "module": "output_layer", "kernel": spatial})


def _replace_prelu_exact(model: nn.Module, report: list[dict]) -> None:
    for parent_name, parent in list(model.named_modules()):
        for name, child in list(parent.named_children()):
            if not isinstance(child, nn.PReLU):
                continue
            setattr(parent, name, PReLUAsOps(child))
            report.append({
                "transformation": "prelu_to_ops",
                "module": f"{parent_name}.{name}".lstrip("."),
                "parameters": child.weight.numel(),
                "equivalent": True,
            })


def _replace_prelu_approx(model: nn.Module, report: list[dict]) -> None:
    for parent_name, parent in list(model.named_modules()):
        for name, child in list(parent.named_children()):
            if not isinstance(child, nn.PReLU):
                continue
            w = child.weight.detach().float().cpu()
            slope = float(w.mean())
            setattr(parent, name, nn.LeakyReLU(negative_slope=slope, inplace=False))
            report.append({
                "transformation": "prelu_to_leakyrelu", "module": f"{parent_name}.{name}".lstrip("."),
                "mean_slope": slope, "slope_std": float(w.std(unbiased=False)),
                "parameters": w.numel(), "equivalent": w.numel() == 1,
            })


def _insert_bn_workaround(model: nn.Module, report: list[dict]) -> None:
    def walk(parent: nn.Module, path: str):
        if isinstance(parent, nn.Sequential):
            previous = None
            for name, child in list(parent.named_children()):
                full_name = f"{path}.{name}".lstrip(".")
                if isinstance(child, nn.BatchNorm2d) and not isinstance(previous, nn.Conv2d):
                    ref = child.running_mean if child.running_mean is not None else child.weight
                    if ref is None:
                        raise ValueError(f"BatchNorm2d lacks dtype/device reference at {full_name}")
                    conv = nn.Conv2d(
                        child.num_features, child.num_features, 1,
                        groups=child.num_features, bias=False,
                        device=ref.device, dtype=ref.dtype,
                    )
                    with torch.no_grad():
                        conv.weight.fill_(1.0)
                    conv.weight.requires_grad_(False)
                    setattr(parent, name, nn.Sequential(conv.eval(), child))
                    report.append({"transformation": "identity_dwconv_before_bn", "module": full_name})
                else:
                    walk(child, full_name)
                previous = child
        else:
            for name, child in parent.named_children():
                walk(child, f"{path}.{name}".lstrip("."))
    walk(model, "")


def _remove_se(model: nn.Module, report: list[dict]) -> None:
    for parent_name, parent in list(model.named_modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, SEModule):
                setattr(parent, name, nn.Identity())
                report.append({"transformation": "remove_se", "module": f"{parent_name}.{name}".lstrip("."),
                               "equivalent": False})



def prelu_parameter_report(model: nn.Module) -> list[dict]:
    """Report baseline per-channel slope ranges before applying approximation."""
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, nn.PReLU):
            weight = module.weight.detach().float().cpu()
            rows.append({
                "module": name, "parameters": weight.numel(),
                "mean": float(weight.mean()),
                "std": float(weight.std(unbiased=False)),
                "min": float(weight.min()), "max": float(weight.max()),
                "negative_fraction": float((weight < 0).float().mean()),
            })
    return rows


def make_variant(baseline: nn.Module, variant: str) -> tuple[nn.Module, list[dict]]:
    """Build one variant from the strict baseline, never stacking ablations.

    exact: common graph conversions + equivalent per-channel PReLU decomposition.
    prelu_approx: common graph conversions + mean-slope LeakyReLU approximation.
    bn_workaround: exact semantics + BN graph-pattern workaround.
    no_se: exact semantics + destructive SE ablation.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {VARIANTS}")
    if baseline.training:
        raise ValueError("Baseline must be eval() before constructing variants")
    if variant == "baseline":
        return baseline, []
    model = deepcopy(baseline)
    report: list[dict] = []
    _replace_shortcuts(model, report)
    _convert_output_head(model, report)
    if variant == "prelu_approx":
        _replace_prelu_approx(model, report)
    else:
        _replace_prelu_exact(model, report)
    if variant == "bn_workaround":
        _insert_bn_workaround(model, report)
    elif variant == "no_se":
        _remove_se(model, report)
    return model.eval(), report
