"""LFW bin pair loading and within-fold verification, not identity clustering.

The .bin format uses pickle: load only a dataset from a trusted source.
This evaluation deliberately does not claim the full paper's augmentation,
face-alignment, or published benchmark protocol.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..preprocessing import PreprocessConfig, preprocess_image


def load_lfw_bin(path: str | Path, *, color_order: str = "RGB") -> tuple[list[np.ndarray], np.ndarray]:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Install the 'eval' extra for LFW JPEG decoding") from exc
    if color_order not in ("RGB", "BGR"):
        raise ValueError("color_order must be RGB or BGR")
    with Path(path).open("rb") as file:
        images_encoded, is_same = pickle.load(file, encoding="bytes")  # trusted file only
    labels = np.asarray(is_same, dtype=bool)
    if len(images_encoded) != 2 * len(labels) or len(labels) == 0:
        raise ValueError("LFW images and pair labels have inconsistent lengths")
    images = []
    for index, raw in enumerate(images_encoded):
        decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"Could not decode LFW image {index}")
        if color_order == "RGB":
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        images.append(decoded)
    return images, labels


class _FaceImages(Dataset):
    def __init__(self, images: list[np.ndarray], config: PreprocessConfig):
        self.images = images
        self.config = config

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        return preprocess_image(self.images[index], self.config)


@torch.inference_mode()
def extract_embeddings(
    model: torch.nn.Module,
    images: list[np.ndarray],
    *,
    config: PreprocessConfig,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    num_workers: int = 0,
) -> np.ndarray:
    if not images or batch_size < 1:
        raise ValueError("Expected non-empty images and positive batch_size")
    model.eval()
    loader = DataLoader(_FaceImages(images, config), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    chunks = []
    for batch in loader:
        embeddings = model(batch.to(device)).flatten(start_dim=1).float()
        chunks.append(F.normalize(embeddings, p=2, dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def evaluate_verification(
    embeddings: np.ndarray, is_same: np.ndarray, *, n_splits: int = 10,
    threshold_steps: int = 2001,
) -> dict:
    try:
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import KFold
    except ImportError as exc:
        raise ImportError("Install the 'eval' extra for LFW verification") from exc
    vectors = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(is_same, dtype=bool)
    if (vectors.ndim != 2 or vectors.shape[0] != 2 * len(labels) or
            len(labels) < n_splits or n_splits < 2 or threshold_steps < 2):
        raise ValueError("Embeddings must contain two vectors per LFW pair and >= n_splits pairs")
    if not np.isfinite(vectors).all() or (np.linalg.norm(vectors, axis=1) <= 1e-12).any():
        raise ValueError("Embeddings must be finite, non-zero vectors")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    scores = np.sum(vectors[0::2] * vectors[1::2], axis=1)
    thresholds = np.linspace(-1.0, 1.0, threshold_steps)
    accuracy = []
    selected_thresholds = []
    for train, test in KFold(n_splits=n_splits, shuffle=False).split(scores):
        # Only training-fold pairs determine the decision threshold.
        train_pred = scores[train, None] >= thresholds[None, :]
        train_accuracy = (train_pred == labels[train, None]).mean(axis=0)
        selected = float(thresholds[int(np.argmax(train_accuracy))])
        accuracy.append(float(np.mean((scores[test] >= selected) == labels[test])))
        selected_thresholds.append(selected)
    result = {
        "accuracy_mean": float(np.mean(accuracy)),
        "accuracy_std": float(np.std(accuracy)),
        "fold_accuracy": accuracy,
        "fold_threshold": selected_thresholds,
        "threshold_mean": float(np.mean(selected_thresholds)),
        "pairs": int(len(labels)),
    }
    result["auc_all_pairs"] = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None
    return result
