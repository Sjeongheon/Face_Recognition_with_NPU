"""Prepare every WIDER FACE validation image for fixed 128x128 ONNX evaluation."""

import argparse
import json
from pathlib import Path

import cv2

from yolov5_face_prepare_images import prepare
from yolov5_face_select_wider import read_annotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    images = args.output / "images"
    images.mkdir(parents=True, exist_ok=True)
    metadata = []
    for index, (relative, faces) in enumerate(read_annotations(args.annotations)):
        if args.limit and index >= args.limit:
            break
        original = cv2.imread(str(args.images_root / relative))
        if original is None:
            raise ValueError(f"cannot read {relative}")
        height, width = original.shape[:2]
        prepared, _, _ = prepare(original)
        image_id = f"val_{index:04d}"
        path = images / f"{image_id}.png"
        if not cv2.imwrite(str(path), prepared):
            raise OSError(f"cannot write {path}")
        metadata.append({"image": image_id, "category": "full_validation",
                         "source": relative, "width": width, "height": height,
                         "faces": faces})
        if (index + 1) % 500 == 0:
            print(f"prepared {index + 1}", flush=True)
    (args.output / "selected_gt.json").write_text(json.dumps(metadata))
    print(f"prepared {len(metadata)} images")


if __name__ == "__main__":
    main()
