import pickle
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from face_npu_ai.cli import build_parser, main
from face_npu_ai.evaluation.dataset import prepare_lfw_bin
from face_npu_ai.evaluation.lfw import load_lfw_bin, evaluate_verification


def test_zip_preparation_copies_only_lfw_and_never_overwrites(tmp_path):
    zpath = tmp_path / "faces_webface_112x112.zip"
    payload = pickle.dumps(([b"image_a", b"image_b"], [True]))
    with ZipFile(zpath, "w") as zf:
        zf.writestr("faces_webface_112x112/lfw.bin", payload)
        zf.writestr("../../unrelated.txt", "not extracted")
    result = tmp_path / "out" / "lfw.bin"
    assert prepare_lfw_bin(zpath, result) == result
    assert result.read_bytes() == payload
    assert not (tmp_path / "unrelated.txt").exists()
    with pytest.raises(FileExistsError):
        prepare_lfw_bin(zpath, result)
    assert main(["prepare-lfw", "--zip", str(zpath), "--output", str(result), "--overwrite"]) == 0


def test_lfw_loader_preserves_labels_and_pair_order(tmp_path):
    def encode(rgb):
        ok, encoded = cv2.imencode(".jpg", np.full((8, 8, 3), rgb, dtype=np.uint8))
        assert ok
        return encoded.tobytes()
    path = tmp_path / "lfw.bin"
    path.write_bytes(pickle.dumps(([encode(15), encode(25), encode(30), encode(40)], [True, False])))
    images, labels = load_lfw_bin(path)
    assert labels.tolist() == [True, False]
    assert len(images) == 4
    assert images[0].shape == (8, 8, 3)


def test_verification_threshold_is_fit_per_training_fold():
    labels = np.array([True, False] * 10)
    vectors = []
    for same in labels:
        vectors.extend(([1.0, 0.0], [1.0, 0.0] if same else [0.0, 1.0]))
    result = evaluate_verification(np.asarray(vectors), labels, n_splits=10)
    assert result["pairs"] == 20
    assert result["accuracy_mean"] == pytest.approx(1.0)
    assert result["auc_all_pairs"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        evaluate_verification(np.zeros((40, 2)), labels)


def test_lfw_requires_explicit_color_order():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["lfw", "--lfw-bin", "lfw.bin", "--checkpoint", "model.pth"])
