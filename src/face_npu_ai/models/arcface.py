"""IR / IR-SE backbone from the reviewed ArcFace experiment.

Module names and tensor shapes deliberately match the reviewed notebook's
checkpoint. Do not apply experimental transformations before strict baseline
checkpoint loading.
The forward output is an *unnormalized* 512-dimensional embedding.
"""

from collections import namedtuple

import torch
from torch import nn


class Bottleneck(namedtuple("Block", ["in_channel", "depth", "stride"])):
    pass


def get_block(in_channel: int, depth: int, num_units: int, stride: int = 2):
    return [Bottleneck(in_channel, depth, stride)] + [
        Bottleneck(depth, depth, 1) for _ in range(num_units - 1)
    ]


def get_blocks(num_layers: int):
    sizes = {
        50: (3, 4, 14, 3),
        100: (3, 13, 30, 3),
        152: (3, 8, 36, 3),
    }
    if num_layers not in sizes:
        raise ValueError(f"Unsupported depth: {num_layers}")
    return [
        get_block(in_ch, out_ch, count)
        for (in_ch, out_ch), count in zip(
            ((64, 64), (64, 128), (128, 256), (256, 512)), sizes[num_layers]
        )
    ]


class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.sigmoid(self.fc2(self.relu(self.fc1(self.avg_pool(x)))))
        return x * gate


class bottleneck_IR(nn.Module):
    def __init__(self, in_channel: int, depth: int, stride: int):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(kernel_size=1, stride=stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, 1, stride=stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, 3, stride=1, padding=1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(depth),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_layer(x) + self.shortcut_layer(x)


class bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel: int, depth: int, stride: int):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(kernel_size=1, stride=stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, 1, stride=stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, 3, stride=1, padding=1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth, 16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_layer(x) + self.shortcut_layer(x)


class Backbone(nn.Module):
    def __init__(
        self,
        input_size: int = 112,
        num_layers: int = 50,
        mode: str = "ir_se",
        drop_ratio: float = 0.6,
        affine: bool = True,
    ):
        super().__init__()
        if input_size not in (112, 224):
            raise ValueError("input_size must be 112 or 224")
        if mode not in ("ir", "ir_se"):
            raise ValueError("mode must be ir or ir_se")
        blocks = get_blocks(num_layers)
        unit_module = bottleneck_IR if mode == "ir" else bottleneck_IR_SE
        self.input_size = input_size
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        spatial = 7 if input_size == 112 else 14
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(drop_ratio),
            nn.Flatten(),
            nn.Linear(512 * spatial * spatial, 512),
            nn.BatchNorm1d(512, affine=affine),
        )
        self.body = nn.Sequential(
            *(unit_module(b.in_channel, b.depth, b.stride)
              for block in blocks for b in block)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.body(self.input_layer(x)))
