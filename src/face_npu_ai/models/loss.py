"""Training-only ArcFace classification head.

Not used by the embedding backbone at inference time. Unlike the historical
MNIST notebook, features and class weights are L2-normalized and the angular
margin is applied without detaching target logits from autograd.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, scale: float = 64.0, margin: float = 0.5):
        super().__init__()
        if in_features <= 0 or out_features <= 0 or scale <= 0 or not 0 <= margin < math.pi / 2:
            raise ValueError("Invalid dimensions, scale, or margin")
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or labels.ndim != 1 or embeddings.shape[0] != labels.shape[0]:
            raise ValueError("Expected [N,D] embeddings and [N] labels")
        if (labels < -1).any() or (labels >= self.weight.shape[0]).any():
            raise ValueError("Labels must be class indices or -1 (ignored)")
        cosine = F.linear(F.normalize(embeddings, dim=1), F.normalize(self.weight, dim=1))
        sine = torch.sqrt((1.0 - cosine.square()).clamp_min(0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        output = cosine.clone()
        valid = (labels != -1).nonzero(as_tuple=True)[0]
        output[valid, labels[valid].long()] = phi[valid, labels[valid].long()]
        return output * self.scale
