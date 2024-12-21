"""Exploratory IoU=0.5 evaluation on a selected WIDER FACE subset."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from yolov5_face_compare import decode, iou, load_heads, nms


def project_gt(item):
    width, height = item["width"], item["height"]
    ratio = 128 / max(width, height)
    rw, rh = int(width * ratio), int(height * ratio)
    sx, sy = rw / width, rh / height
    left = int(round((128 - rw) / 2 - 0.1))
    top = int(round((128 - rh) / 2 - 0.1))
    boxes = []
    for face in item["faces"]:
        x, y, w, h = face["box"]
        if face["invalid"] or w <= 0 or h <= 0:
            continue
        boxes.append((np.array((x * sx + left, y * sy + top,
                                (x + w) * sx + left, (y + h) * sy + top)),
                      min(w * sx, h * sy)))
    return boxes


def evaluate(detections, ground_truth, threshold=0.5):
    pairs = sorted(((iou(det, gt[0]), di, gi) for di, det in enumerate(detections)
                    for gi, gt in enumerate(ground_truth)), reverse=True)
    used_det, used_gt = set(), set()
    for overlap, di, gi in pairs:
        if overlap < threshold:
            break
        if di not in used_det and gi not in used_gt:
            used_det.add(di)
            used_gt.add(gi)
    large = {gi for gi, (_, size) in enumerate(ground_truth) if size >= 8}
    return {"tp": len(used_gt), "fp": len(detections) - len(used_det),
            "fn": len(ground_truth) - len(used_gt), "gt": len(ground_truth),
            "gt_ge8": len(large), "tp_ge8": len(used_gt & large)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-gt", type=Path, required=True)
    parser.add_argument("--fp32-root", type=Path, required=True)
    parser.add_argument("--int8-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    items = json.loads(args.selected_gt.read_text())
    totals = defaultdict(lambda: defaultdict(int))
    with args.csv.open("w", newline="") as stream:
        fields = ("image", "category", "model", "gt", "gt_ge8", "tp", "tp_ge8", "fp", "fn")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in items:
            gt = project_gt(item)
            for model, root in (("fp32", args.fp32_root), ("int8", args.int8_root)):
                detections = nms(decode(load_heads(root, item["image"])))
                row = {"image": item["image"], "category": item["category"], "model": model,
                       **evaluate(detections, gt)}
                writer.writerow(row)
                for field in ("gt", "gt_ge8", "tp", "tp_ge8", "fp", "fn"):
                    totals[(model, item["category"])][field] += row[field]
    for (model, category), values in sorted(totals.items()):
        print(model, category, dict(values))


if __name__ == "__main__":
    main()
