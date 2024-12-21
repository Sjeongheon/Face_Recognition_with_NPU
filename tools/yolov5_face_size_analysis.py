"""Summarize FP32/INT8 agreement by ground-truth face size at 128x128."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from yolov5_face_compare import decode, iou, load_heads, nms
from yolov5_face_wider_metrics import project_gt


def match_to_gt(detections, ground_truth):
    pairs = sorted(((iou(det, gt[0]), di, gi)
                    for di, det in enumerate(detections)
                    for gi, gt in enumerate(ground_truth)), reverse=True)
    matched, used_det = {}, set()
    for overlap, di, gi in pairs:
        if overlap < 0.5:
            break
        if di not in used_det and gi not in matched:
            matched[gi] = detections[di]
            used_det.add(di)
    return matched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-gt", type=Path, required=True)
    parser.add_argument("--fp32-root", type=Path, required=True)
    parser.add_argument("--int8-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    thresholds = (0, 8, 16, 24, 32)
    buckets = {threshold: {key: [] for key in ("fp", "int8", "both", "fp_only", "int8_only", "box_iou", "landmark_px", "landmark_relative")}
               for threshold in thresholds}
    dominant = {key: [] for key in ("fp", "int8", "both", "fp_only", "int8_only", "box_iou", "landmark_px", "landmark_relative")}

    def add(bucket, gt_index, ground_truth, fp_matches, int8_matches):
        fp = fp_matches.get(gt_index)
        quantized = int8_matches.get(gt_index)
        bucket["fp"].append(int(fp is not None))
        bucket["int8"].append(int(quantized is not None))
        bucket["both"].append(int(fp is not None and quantized is not None))
        bucket["fp_only"].append(int(fp is not None and quantized is None))
        bucket["int8_only"].append(int(fp is None and quantized is not None))
        if fp is not None and quantized is not None:
            delta = np.linalg.norm(fp[5:15].reshape(5, 2) - quantized[5:15].reshape(5, 2), axis=1).mean()
            bucket["box_iou"].append(iou(fp, quantized))
            bucket["landmark_px"].append(float(delta))
            bucket["landmark_relative"].append(float(delta / max(ground_truth[gt_index][1], 1e-6)))

    for item in json.loads(args.selected_gt.read_text()):
        name = item["image"]
        gt = project_gt(item)
        fp = nms(decode(load_heads(args.fp32_root, name)))
        quantized = nms(decode(load_heads(args.int8_root, name)))
        fp_matches = match_to_gt(fp, gt)
        int8_matches = match_to_gt(quantized, gt)
        for gi, (_, size) in enumerate(gt):
            for threshold in thresholds:
                if size >= threshold:
                    add(buckets[threshold], gi, gt, fp_matches, int8_matches)
        if gt:
            gi = max(range(len(gt)), key=lambda i: np.prod(gt[i][0][2:4] - gt[i][0][:2]))
            add(dominant, gi, gt, fp_matches, int8_matches)

    rows = []
    for label, bucket in [(f"ge_{threshold}px", buckets[threshold]) for threshold in thresholds] + [("largest_face_per_image", dominant)]:
        row = {"group": label, "gt_count": len(bucket["fp"]),
               **{key: sum(bucket[key]) for key in ("fp", "int8", "both", "fp_only", "int8_only")},
               **{key: float(np.mean(bucket[key])) if bucket[key] else "" for key in ("box_iou", "landmark_px", "landmark_relative")}}
        rows.append(row)
        print(row)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
