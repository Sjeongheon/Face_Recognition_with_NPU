"""Measure how YOLO landmark changes affect aligned ArcFace embeddings."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from yolov5_face_compare import decode, iou, load_heads, nms
from yolov5_face_wider_metrics import project_gt


ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014],
                        [56.0252, 71.7366], [41.5493, 92.3655],
                        [70.7299, 92.2041]], dtype=np.float64)


def match_to_gt(detections, ground_truth):
    pairs = sorted(((iou(det, gt[0]), di, gi) for di, det in enumerate(detections)
                    for gi, gt in enumerate(ground_truth)), reverse=True)
    matched, used_det = {}, set()
    for overlap, di, gi in pairs:
        if overlap < 0.5:
            break
        if di not in used_det and gi not in matched:
            matched[gi] = detections[di]
            used_det.add(di)
    return matched


def restore_landmarks(prediction, width, height):
    ratio = 128 / max(width, height)
    resized_width, resized_height = int(width * ratio), int(height * ratio)
    pad_x = int(round((128 - resized_width) / 2 - 0.1))
    pad_y = int(round((128 - resized_height) / 2 - 0.1))
    points = prediction[5:15].reshape(5, 2).astype(np.float64).copy()
    points[:, 0] = np.clip((points[:, 0] - pad_x) * width / resized_width, 0, width)
    points[:, 1] = np.clip((points[:, 1] - pad_y) * height / resized_height, 0, height)
    # Yolov5Face.detect() rounds and returns int32 landmark coordinates.
    return np.rint(points)


def align_face(image, points):
    # Least-squares similarity transform used by skimage's SimilarityTransform.
    design = np.zeros((10, 4), dtype=np.float64)
    design[0::2, 0], design[0::2, 1] = points[:, 0], -points[:, 1]
    design[0::2, 2] = 1
    design[1::2, 0], design[1::2, 1] = points[:, 1], points[:, 0]
    design[1::2, 3] = 1
    a, b, tx, ty = np.linalg.lstsq(design, ARCFACE_DST.reshape(-1), rcond=None)[0]
    matrix = np.array([[a, -b, tx], [b, a, ty]], dtype=np.float64)
    return cv2.warpAffine(image, matrix, (112, 112), borderValue=0)


def embedding(session, input_name, crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = (rgb.astype(np.float32).transpose(2, 0, 1)[None] / 127.5) - 1
    output = session.run(None, {input_name: tensor})[0].reshape(-1).astype(np.float64)
    norm = np.linalg.norm(output)
    if norm == 0:
        raise ValueError("ArcFace returned zero embedding")
    return output / norm


def summary(rows, key):
    values = np.array([row[key] for row in rows], dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p05": float(np.quantile(values, 0.05)), "min": float(values.min())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-gt", type=Path, required=True)
    parser.add_argument("--original-images", type=Path, required=True)
    parser.add_argument("--fp32-root", type=Path, required=True)
    parser.add_argument("--old-int8-root", type=Path, required=True)
    parser.add_argument("--new-int8-root", type=Path, required=True)
    parser.add_argument("--arcface-onnx", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--min-face-px", type=float, default=24)
    args = parser.parse_args()
    session = ort.InferenceSession(str(args.arcface_onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    rows = []
    eligible, available = 0, 0
    for item in json.loads(args.selected_gt.read_text()):
        ground_truth = project_gt(item)
        selected_indices = [i for i, (_, size) in enumerate(ground_truth) if size >= args.min_face_px]
        if not selected_indices:
            continue
        eligible += len(selected_indices)
        name = item["image"]
        fp = match_to_gt(nms(decode(load_heads(args.fp32_root, name))), ground_truth)
        old = match_to_gt(nms(decode(load_heads(args.old_int8_root, name))), ground_truth)
        new = match_to_gt(nms(decode(load_heads(args.new_int8_root, name))), ground_truth)
        complete = [i for i in selected_indices if i in fp and i in old and i in new]
        if not complete:
            continue
        image = cv2.imread(str(args.original_images / item["source"]))
        if image is None:
            raise ValueError(f"cannot read {item['source']}")
        for i in complete:
            available += 1
            crops = [align_face(image, restore_landmarks(pred, item["width"], item["height"]))
                     for pred in (fp[i], old[i], new[i])]
            features = [embedding(session, input_name, crop) for crop in crops]
            rows.append({"image": name, "source": item["source"], "face_index": i,
                         "face_min_side_at_128": ground_truth[i][1],
                         "cosine_old": float(features[0] @ features[1]),
                         "cosine_new": float(features[0] @ features[2]),
                         "crop_mae_old": float(np.abs(crops[0].astype(np.float32) - crops[1]).mean()),
                         "crop_mae_new": float(np.abs(crops[0].astype(np.float32) - crops[2]).mean())})
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image", "source", "face_index",
                          "face_min_side_at_128", "cosine_old", "cosine_new",
                          "crop_mae_old", "crop_mae_new"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"eligible_gt={eligible} all_three_detected={available}")
    if rows:
        for key in ("cosine_old", "cosine_new", "crop_mae_old", "crop_mae_new"):
            print(key, summary(rows, key))
        print("cosine_new_better_count", sum(row["cosine_new"] > row["cosine_old"] for row in rows))
        print("crop_new_lower_mae_count", sum(row["crop_mae_new"] < row["crop_mae_old"] for row in rows))


if __name__ == "__main__":
    main()
