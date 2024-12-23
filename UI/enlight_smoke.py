"""Opt-in ENLIGHT simulator recognition path for UI smoke testing."""

import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np


class DockerEnlightClient:
    def __init__(self, container=None, python=None, script=None, arcface_backend=None):
        self.container = container or os.getenv("FACE_NPU_ENLIGHT_CONTAINER", "enlight-dev")
        self.python = python or os.getenv(
            "FACE_NPU_ENLIGHT_PYTHON",
            "/NPU/ENLIGHT_NPU_V20_SW_Toolkit_v0.9.9/venv/bin/python",
        )
        self.script = script or os.getenv(
            "FACE_NPU_ENLIGHT_RUNNER",
            "/NPU/yolov5_face_eval_20261115/scripts/yolov5_face_enlight_runner.py",
        )
        self.arcface_backend = arcface_backend or os.getenv("FACE_NPU_ARCFACE_BACKEND", "onnx")
        if self.arcface_backend not in ("onnx", "enlight"):
            raise ValueError("FACE_NPU_ARCFACE_BACKEND must be 'onnx' or 'enlight'")

    def infer(self, frame):
        encoded, data = cv2.imencode(".png", frame)
        if not encoded:
            raise ValueError("cannot encode frame")
        command = ["docker", "exec", "-i", self.container, self.python, self.script,
                   "--arcface-backend", self.arcface_backend]
        result = subprocess.run(command, input=data.tobytes(), capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError(f"ENLIGHT runner failed: {result.stderr.decode(errors='replace')[-2000:]}")
        output = json.loads(result.stdout)
        if not isinstance(output.get("faces"), list):
            raise ValueError("ENLIGHT runner returned invalid faces")
        return output


class EnlightSmokeService:
    def __init__(self, gallery_path, client=None):
        self.gallery_path = Path(gallery_path)
        self.client = client or DockerEnlightClient()
        gallery = json.loads(self.gallery_path.read_text())
        self.threshold = float(gallery["threshold"])
        if not -1.0 <= self.threshold <= 1.0:
            raise ValueError("gallery threshold must be between -1 and 1")
        self.people = []
        for person in gallery["people"]:
            embedding = np.asarray(person["embedding"], dtype=np.float64)
            if embedding.shape != (512,) or not np.isfinite(embedding).all():
                raise ValueError("gallery embedding must be finite and 512-dimensional")
            norm = np.linalg.norm(embedding)
            if norm == 0:
                raise ValueError("gallery embedding must have nonzero norm")
            self.people.append((person["name"], embedding / norm))
        if not self.people:
            raise ValueError("gallery must contain at least one person")

    def recognize(self, frame):
        result = self.client.infer(frame)
        best_name, best_score = None, -1.0
        for face in result["faces"]:
            embedding = np.asarray(face["embedding"], dtype=np.float64)
            if embedding.shape != (512,) or not np.isfinite(embedding).all():
                raise ValueError("inference embedding must be finite and 512-dimensional")
            norm = np.linalg.norm(embedding)
            if norm == 0:
                raise ValueError("inference embedding must have nonzero norm")
            embedding /= norm
            for name, enrolled in self.people:
                score = float(embedding @ enrolled)
                if score > best_score:
                    best_name, best_score = name, score
        return best_name if best_score >= self.threshold else None
