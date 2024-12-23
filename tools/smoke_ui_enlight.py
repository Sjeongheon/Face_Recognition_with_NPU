"""Exercise the UI WebSocket with ENLIGHT detector and ArcFace inference.

Example:
    .venv/bin/python tools/smoke_ui_enlight.py \
        --enroll /path/to/person_a_enroll.jpg \
        --known /path/to/person_a_query.jpg \
        --unknown /path/to/person_b_query.jpg
"""

import argparse
import base64
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "UI"))

from enlight_smoke import DockerEnlightClient  # noqa: E402


def websocket_recognize(client, image_path):
    encoded = base64.b64encode(image_path.read_bytes()).decode()
    with client.websocket_connect("/ws") as websocket:
        websocket.send_text("recognizedata:image/jpeg;base64," + encoded)
        return websocket.receive_json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enroll", type=Path, required=True)
    parser.add_argument("--known", type=Path, required=True)
    parser.add_argument("--unknown", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--arcface-backend", choices=("onnx", "enlight"), default="onnx")
    args = parser.parse_args()
    os.environ["FACE_NPU_ARCFACE_BACKEND"] = args.arcface_backend

    image = cv2.imread(str(args.enroll))
    if image is None:
        parser.error(f"cannot read enrollment image: {args.enroll}")
    enrollment = DockerEnlightClient().infer(image)
    if not enrollment["faces"]:
        raise AssertionError("no face found in enrollment image")
    expected_model = ("arcface_exact.onnx" if args.arcface_backend == "onnx"
                      else "arcface_exact_quantized.enlight")
    assert enrollment["arcface_model"] == expected_model

    with tempfile.TemporaryDirectory(prefix="face_npu_ui_smoke_") as directory:
        gallery_path = Path(directory) / "gallery.json"
        gallery_path.write_text(json.dumps({
            "threshold": args.threshold,
            "people": [{"name": "smoke_person", "embedding": enrollment["faces"][0]["embedding"]}],
        }))
        os.environ["FACE_NPU_ENLIGHT_SMOKE_GALLERY"] = str(gallery_path)
        from main import app  # import after setting the opt-in environment variable

        with TestClient(app) as client:
            page = client.get("/")
            assert page.status_code == 200
            assert "WebSocket" in page.text or "websocket" in page.text
            icon = client.get("/static/fuel_icon3.png")
            assert icon.status_code == 200 and icon.content
            known = websocket_recognize(client, args.known)
            unknown = websocket_recognize(client, args.unknown)

    print("GET /:", page.status_code)
    print("GET /static/fuel_icon3.png:", icon.status_code)
    print("ArcFace backend:", args.arcface_backend)
    print("known:", known["type"], known.get("name"))
    print("unknown:", unknown["type"], unknown.get("name"))
    assert known["type"] == "recognition_result" and known["name"] == "smoke_person"
    assert unknown["type"] == "recognition_processing"


if __name__ == "__main__":
    main()
