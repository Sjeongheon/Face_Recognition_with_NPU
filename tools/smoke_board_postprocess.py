"""Replay ENLIGHT simulator raw heads through the board C decoder."""

import argparse
import ctypes
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from yolov5_face_compare import decode, iou, load_heads

ROOT = Path(__file__).resolve().parents[1]


class Detection(ctypes.Structure):
    _fields_ = [("values", ctypes.c_float * 15)]


def stable_nms(rows):
    rows = rows[rows[:, 4] > 0.4]
    # Keep original candidate order when quantized scores are identical.
    rows = rows[np.lexsort((np.arange(len(rows)), -rows[:, 4]))]
    kept = []
    for row in rows:
        if all(iou(row, other) <= 0.5 for other in kept):
            kept.append(row)
    return np.asarray(kept, dtype=np.float32).reshape(-1, 15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=40)
    args = parser.parse_args()
    folders = sorted(path for path in args.capture_root.iterdir() if path.is_dir())
    if not folders or args.count < 1:
        parser.error("capture root must contain image folders and count must be positive")
    indexes = np.linspace(0, len(folders) - 1, min(args.count, len(folders)), dtype=int)
    samples = [folders[index] for index in indexes]

    with tempfile.TemporaryDirectory(prefix="face_npu_board_smoke_") as directory:
        library_path = Path(directory) / "libyolov5_face_postprocess.dylib"
        subprocess.run(["cc", "-std=c99", "-O2", "-shared", "-fPIC", "-o", str(library_path),
                        str(ROOT / "board/yolov5_face_postprocess.c"), "-lm"], check=True)
        library = ctypes.CDLL(str(library_path))
        pointer = ctypes.POINTER(ctypes.c_float)
        library.yolov5_face_decode_nms.argtypes = [ctypes.POINTER(pointer), ctypes.c_float,
                                                    ctypes.c_float, ctypes.POINTER(Detection), ctypes.c_size_t]
        library.yolov5_face_decode_nms.restype = ctypes.c_int

        total_detections = 0
        for sample in samples:
            heads = [np.ascontiguousarray(head, dtype=np.float32)
                     for head in load_heads(args.capture_root, sample.name)]
            head_pointers = (pointer * 3)(*(head.ctypes.data_as(pointer) for head in heads))
            output = (Detection * 1008)()
            count = library.yolov5_face_decode_nms(head_pointers, 0.4, 0.5, output, 1008)
            if count < 0:
                raise AssertionError(f"C decoder failed for {sample.name}")
            actual = np.array([list(output[i].values) for i in range(count)], dtype=np.float32).reshape(-1, 15)
            expected = stable_nms(decode(heads))
            if actual.shape != expected.shape:
                raise AssertionError(f"detection count differs for {sample.name}: {count} vs {len(expected)}")
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-3,
                                       err_msg=f"C/Python mismatch for {sample.name}")
            total_detections += count
    print(f"C/Python decode + NMS parity: {len(samples)} images, {total_detections} detections")


if __name__ == "__main__":
    main()
