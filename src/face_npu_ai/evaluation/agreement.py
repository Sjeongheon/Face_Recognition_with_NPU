"""Same-input embedding comparison (PyTorch / ONNX / NPU)."""

import numpy as np


def _normalize(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / np.maximum(norms, eps)


def compare_embeddings(reference, candidate) -> dict[str, float]:
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError(f"Non-empty equal shapes required: {a.shape} vs {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Embedding contains NaN or infinity")
    a = a.reshape(1, -1) if a.ndim == 1 else a.reshape(a.shape[0], -1)
    b = b.reshape(1, -1) if b.ndim == 1 else b.reshape(b.shape[0], -1)
    if (np.linalg.norm(a, axis=1) <= 1e-12).any() or (np.linalg.norm(b, axis=1) <= 1e-12).any():
        raise ValueError("Cannot compare a zero-norm embedding with cosine similarity")
    cosine = (_normalize(a) * _normalize(b)).sum(axis=1)
    difference = np.abs(a - b)
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "mean_cosine": float(cosine.mean()),
        "min_cosine": float(cosine.min()),
    }
