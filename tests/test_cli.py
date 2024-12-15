import pytest
from face_npu_ai.cli import build_parser


def test_variant_and_export_commands_accept_documented_options():
    parser = build_parser()
    args = parser.parse_args(["compare-variants", "--checkpoint", "weights.pth", "--variant", "exact"])
    assert args.variant == "exact"
    exp = parser.parse_args(["export", "--checkpoint", "weights.pth", "--variant", "prelu_approx",
                             "--output", "arcface.onnx"])
    assert exp.output.name == "arcface.onnx"
    with pytest.raises(SystemExit):
        parser.parse_args(["compare-variants", "--checkpoint", "weights.pth", "--variant", "invalid"])
