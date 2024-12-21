"""Compare final FP32/INT8 detections from three ENLIGHT raw heads."""

import argparse
import csv
from pathlib import Path

import numpy as np


ANCHORS = (((4, 5), (8, 10), (13, 16)),
           ((23, 29), (43, 55), (73, 105)),
           ((146, 217), (231, 300), (335, 433)))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -80, 80)))


def decode(heads):
    candidates = []
    for index, (head, anchors) in enumerate(zip(heads, ANCHORS)):
        stride = 8 * 2 ** index
        side = 16 // 2 ** index
        if head.shape != (1, 48, side, side):
            raise ValueError(f"P{index + 3}: unexpected shape {head.shape}")
        # ONNX raw output has channels [3 anchors x 16 values].
        raw = head.reshape(1, 3, 16, side, side).transpose(0, 1, 3, 4, 2)[0]
        gy, gx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
        grid = np.stack((gx, gy), axis=-1)[None]
        anchor = np.asarray(anchors, dtype=np.float32)[:, None, None, :]
        xy = (sigmoid(raw[..., :2]) * 2 - 0.5 + grid) * stride
        wh = (sigmoid(raw[..., 2:4]) * 2) ** 2 * anchor
        obj = sigmoid(raw[..., 4:5])
        landmarks = (raw[..., 5:15].reshape(3, side, side, 5, 2)
                     * anchor[..., None, :] + grid[..., None, :] * stride)
        score = obj * sigmoid(raw[..., 15:16])
        box = np.concatenate((xy - wh / 2, xy + wh / 2), axis=-1)
        candidates.append(np.concatenate((box, score, landmarks.reshape(3, side, side, 10)), axis=-1)
                          .reshape(-1, 15))
    return np.concatenate(candidates)


def iou(a, b):
    lo = np.maximum(a[:2], b[:2])
    hi = np.minimum(a[2:4], b[2:4])
    intersection = np.prod(np.maximum(0, hi - lo))
    area_a = np.prod(np.maximum(0, a[2:4] - a[:2]))
    area_b = np.prod(np.maximum(0, b[2:4] - b[:2]))
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def nms(rows, conf_threshold=0.4, iou_threshold=0.5):
    rows = rows[rows[:, 4] > conf_threshold]
    rows = rows[np.argsort(-rows[:, 4])]
    keep = []
    for row in rows:
        if all(iou(row, other) <= iou_threshold for other in keep):
            keep.append(row)
    return np.asarray(keep, dtype=np.float32).reshape(-1, 15)


def load_heads(root, image_name):
    folder = root / image_name
    return [np.load(folder / f"{image_name}_p{level}.npy") for level in (3, 4, 5)]


def compare(fp, quantized, min_match_iou=0.5):
    pairs = sorted(((iou(a, b), ai, bi) for ai, a in enumerate(fp)
                    for bi, b in enumerate(quantized)), reverse=True)
    used_fp, used_int8, matches = set(), set(), []
    for overlap, ai, bi in pairs:
        if overlap < min_match_iou:
            break
        if ai not in used_fp and bi not in used_int8:
            used_fp.add(ai)
            used_int8.add(bi)
            landmark_px = float(np.linalg.norm(
                fp[ai, 5:15].reshape(5, 2) - quantized[bi, 5:15].reshape(5, 2), axis=1).mean())
            matches.append((overlap, float(quantized[bi, 4] - fp[ai, 4]), landmark_px))
    return matches, len(fp) - len(matches), len(quantized) - len(matches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-root", type=Path, required=True)
    parser.add_argument("--int8-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    images = sorted(p.name for p in args.fp32_root.iterdir() if p.is_dir())
    if images != sorted(p.name for p in args.int8_root.iterdir() if p.is_dir()):
        parser.error("FP32 and INT8 image sets differ")
    fields = ("image", "fp32_count", "int8_count", "matched", "fp32_only",
              "int8_only", "mean_iou", "mean_conf_delta", "mean_landmark_px")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for image in images:
            fp = nms(decode(load_heads(args.fp32_root, image)))
            quantized = nms(decode(load_heads(args.int8_root, image)))
            matches, fp_only, int8_only = compare(fp, quantized)
            row = dict(image=image, fp32_count=len(fp), int8_count=len(quantized),
                       matched=len(matches), fp32_only=fp_only, int8_only=int8_only,
                       mean_iou=np.mean([m[0] for m in matches]) if matches else "",
                       mean_conf_delta=np.mean([m[1] for m in matches]) if matches else "",
                       mean_landmark_px=np.mean([m[2] for m in matches]) if matches else "")
            writer.writerow(row)
            print(row)


if __name__ == "__main__":
    main()
