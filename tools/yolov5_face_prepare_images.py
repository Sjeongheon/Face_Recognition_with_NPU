"""Create aspect-preserving 128x128 inputs for the fixed-shape ONNX model."""

import argparse
import csv
from pathlib import Path

import cv2


def prepare(image):
    h, w = image.shape[:2]
    ratio = 128 / max(h, w)
    if ratio != 1:
        interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        image = cv2.resize(image, (int(w * ratio), int(h * ratio)), interpolation=interpolation)
    rh, rw = image.shape[:2]
    # The PyTorch detector uses auto=True and may produce 64x128 for tall images.
    # The exported ONNX has a fixed 128x128 input, so pad to that full shape.
    dw, dh = (128 - rw) / 2, (128 - rh) / 2
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    output = cv2.copyMakeBorder(image, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    if output.shape[:2] != (128, 128):
        raise ValueError(f"unexpected letterbox shape: {output.shape}")
    return output, ratio, (left, top)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image", "source", "width", "height", "scale", "pad_x", "pad_y"))
        for path in sorted(args.source.iterdir()):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"cannot read {path}")
            h, w = image.shape[:2]
            output, scale, (pad_x, pad_y) = prepare(image)
            saved = args.output / f"{path.stem}.png"
            if not cv2.imwrite(str(saved), output):
                raise OSError(f"cannot write {saved}")
            writer.writerow((path.stem, str(path), w, h, scale, pad_x, pad_y))
            print(f"{path.name}: {w}x{h} -> {saved.name}")


if __name__ == "__main__":
    main()
