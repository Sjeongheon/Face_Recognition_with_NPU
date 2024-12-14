"""Static ONNX export and identical-input PyTorch/ONNXRuntime comparison."""

from __future__ import annotations

from pathlib import Path

import inspect
import numpy as np
import torch

from .evaluation.agreement import compare_embeddings


def export_onnx(
    model: torch.nn.Module,
    output: str | Path,
    *,
    input_size: int = 112,
    opset: int = 11,
    batch_size: int = 1,
) -> Path:
    try:
        import onnx
    except ImportError as exc:
        raise ImportError("Install 'face-npu-ai[onnx]' to export and check an ONNX model") from exc
    if batch_size < 1 or input_size not in (112, 224):
        raise ValueError("Invalid ONNX input size or batch size")
    # Exporting a CPU copy avoids unexpectedly relocating the caller's model.
    from copy import deepcopy
    cpu_model = deepcopy(model).cpu().eval()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros(batch_size, 3, input_size, input_size, dtype=torch.float32)
    kwargs = dict(export_params=True, do_constant_folding=True,
                  input_names=["input"], output_names=["output"], opset_version=opset)
    # Newer PyTorch defaults to dynamo, which may change legacy opset export behavior.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(cpu_model, sample, str(path), **kwargs)
    onnx.checker.check_model(onnx.load(str(path)))
    return path


@torch.inference_mode()
def compare_onnx(
    model: torch.nn.Module, onnx_path: str | Path, input_tensor: torch.Tensor,
) -> dict[str, float]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("Install 'face-npu-ai[onnx]' for ONNXRuntime comparison") from exc
    from copy import deepcopy
    model_cpu = deepcopy(model).cpu().eval()
    sample = input_tensor.detach().cpu().float()
    if sample.ndim != 4 or sample.shape[1] != 3:
        raise ValueError("Expected NCHW RGB/BGR face input [N,3,H,W]")
    pytorch_out = model_cpu(sample).flatten(start_dim=1).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_out = session.run(None, {input_name: np.ascontiguousarray(sample.numpy())})[0]
    if pytorch_out.shape != onnx_out.reshape(sample.shape[0], -1).shape:
        raise ValueError("PyTorch/ONNX output shape mismatch")
    return compare_embeddings(pytorch_out, onnx_out.reshape(sample.shape[0], -1))
