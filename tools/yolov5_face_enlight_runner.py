"""One-image ENLIGHT simulator runner for the UI smoke path.

Reads an encoded image from stdin and writes one JSON object to stdout.
Run with the Python environment inside the ENLIGHT container.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from yolov5_face_arcface_alignment_eval import align_face, embedding, restore_landmarks
from yolov5_face_compare import decode, load_heads, nms
from yolov5_face_prepare_images import prepare


def enlight_embedding(crop, args, root, index):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1
    input_path = root / f"face_{index}.npy"
    np.save(input_path, tensor)
    result_root = root / "arcface_results"
    command = [sys.executable, str(args.sdk_root / "EnlightSDK/enlight_sim.py"),
               str(args.arcface_model), "--inputs", str(input_path),
               "--result-root", str(result_root), "--enable-save-npy-result-for-unknown"]
    run = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if run.returncode:
        raise RuntimeError(f"ArcFace ENLIGHT simulator failed: {run.stderr[-2000:]} {run.stdout[-2000:]}")
    output_path = result_root / args.arcface_model.stem / input_path.stem / f"{input_path.stem}_result.npy"
    vector = np.load(output_path).reshape(-1).astype(np.float64)
    if vector.shape != (512,) or not np.isfinite(vector).all():
        raise ValueError(f"ArcFace ENLIGHT output has invalid shape or values: {output_path}")
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("ArcFace ENLIGHT returned zero embedding")
    return vector / norm


def restore_box(box, width, height):
    ratio = 128 / max(width, height)
    rw, rh = int(width * ratio), int(height * ratio)
    pad_x = int(round((128 - rw) / 2 - 0.1))
    pad_y = int(round((128 - rh) / 2 - 0.1))
    result = np.asarray(box, dtype=np.float64).copy()
    result[[0, 2]] = np.clip((result[[0, 2]] - pad_x) * width / rw, 0, width)
    result[[1, 3]] = np.clip((result[[1, 3]] - pad_y) * height / rh, 0, height)
    return result.tolist()


def infer(image, args):
    height, width = image.shape[:2]
    prepared, _, _ = prepare(image)
    with tempfile.TemporaryDirectory(prefix="enlight_face_") as directory:
        root = Path(directory)
        input_path = root / "input.png"
        if not cv2.imwrite(str(input_path), prepared):
            raise OSError(f"cannot write {input_path}")
        result_root = root / "results"
        command = [sys.executable, str(args.capture_script), "--sdk-root", str(args.sdk_root),
                   "--model", str(args.model), "--inputs", str(input_path),
                   "--result-root", str(result_root)]
        run = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if run.returncode:
            raise RuntimeError(f"ENLIGHT simulator failed: {run.stderr[-2000:]} {run.stdout[-2000:]}")
        heads = load_heads(result_root / args.model.stem, "input")
        detections = nms(decode(heads), args.conf_threshold, args.nms_iou)

    detections = [row for row in detections if min(row[2] - row[0], row[3] - row[1]) >= args.min_face_px]
    detections.sort(key=lambda row: float((row[2] - row[0]) * (row[3] - row[1])), reverse=True)
    detections = detections[:args.max_faces]
    arcface_model_name = (args.arcface_model.name if args.arcface_backend == "enlight"
                          else args.arcface_onnx.name)
    if not detections:
        return {"faces": [], "model": args.model.name, "arcface_model": arcface_model_name}

    arcface = None
    if args.arcface_backend == "onnx":
        arcface = ort.InferenceSession(str(args.arcface_onnx), providers=["CPUExecutionProvider"])
        input_name = arcface.get_inputs()[0].name
    faces = []
    with tempfile.TemporaryDirectory(prefix="enlight_arcface_") as directory:
        for index, row in enumerate(detections):
            points = restore_landmarks(row, width, height)
            crop = align_face(image, points)
            vector = (enlight_embedding(crop, args, Path(directory), index)
                      if args.arcface_backend == "enlight" else embedding(arcface, input_name, crop))
            faces.append({"box": restore_box(row[:4], width, height),
                          "confidence": float(row[4]), "landmarks": points.tolist(),
                          "embedding": vector.tolist()})
    return {"faces": faces, "model": args.model.name, "arcface_model": arcface_model_name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk-root", type=Path, default=Path("/NPU/tc-nn-toolkit"))
    parser.add_argument("--capture-script", type=Path,
                        default=Path("/NPU/yolov5_face_eval_20261115/scripts/capture.py"))
    parser.add_argument("--model", type=Path, default=Path(
        "/NPU/yolov5_face_eval_20261115/wider_face_val/calibration_face100/"
        "yolov5m-face_raw_facecalib_quantized.enlight"))
    parser.add_argument("--arcface-onnx", type=Path, default=Path("/NPU/arcface_exact.onnx"))
    parser.add_argument("--arcface-model", type=Path, default=Path("/NPU/arcface_exact_quantized.enlight"))
    parser.add_argument("--arcface-backend", choices=("onnx", "enlight"), default="onnx")
    parser.add_argument("--conf-threshold", type=float, default=0.4)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--min-face-px", type=float, default=24)
    parser.add_argument("--max-faces", type=int, default=3)
    args = parser.parse_args()
    data = np.frombuffer(sys.stdin.buffer.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        parser.error("stdin must contain an encoded image")
    print(json.dumps(infer(image, args)), flush=True)


if __name__ == "__main__":
    main()
