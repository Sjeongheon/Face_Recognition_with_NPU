"""Host-only command-line experiments. Never claims successful NPU deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .checkpoint import load_backbone
from .evaluation.agreement import compare_embeddings
from .preprocessing import PreprocessConfig, preprocess_image
from .experiments import VARIANTS, make_variant, prelu_parameter_report


def _base_args(parser):
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-size", type=int, choices=(112, 224), default=112)
    parser.add_argument("--num-layers", type=int, choices=(50, 100, 152), default=50)
    parser.add_argument("--mode", choices=("ir", "ir_se"), default="ir_se")
    parser.add_argument("--device", default="cpu", help="PyTorch device for LFW evaluation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-lfw", help="Copy lfw.bin from trusted WebFace ZIP")
    prepare.add_argument("--zip", type=Path, required=True, dest="archive")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--overwrite", action="store_true")

    lfw = commands.add_parser("lfw", help="Run labeled LFW verification of separate model variants")
    _base_args(lfw)
    lfw.add_argument("--lfw-bin", type=Path, required=True, help="Trusted LFW .bin file (pickle format)")
    lfw.add_argument("--color-order", choices=("RGB", "BGR"), required=True,
                     help="Must match the checkpoint's training preprocessing")
    lfw.add_argument("--variants", choices=VARIANTS, nargs="+", default=["baseline", "exact", "prelu_approx"])
    lfw.add_argument("--batch-size", type=int, default=64)
    lfw.add_argument("--workers", type=int, default=0)
    lfw.add_argument("--output-json", type=Path)

    agreement = commands.add_parser("compare-variants", help="Same synthetic tensor on baseline and one variant")
    _base_args(agreement)
    agreement.add_argument("--variant", choices=tuple(v for v in VARIANTS if v != "baseline"), required=True)
    agreement.add_argument("--batch-size", type=int, default=2)

    export = commands.add_parser("export", help="Export one variant as static-batch ONNX")
    _base_args(export)
    export.add_argument("--variant", choices=VARIANTS, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--opset", type=int, default=11,
                        help="Confirm opset support for your installed converter")

    compare = commands.add_parser("compare-onnx", help="Compare same-input PyTorch and ONNX outputs")
    _base_args(compare)
    compare.add_argument("--variant", choices=VARIANTS, required=True)
    compare.add_argument("--onnx", type=Path, required=True)
    compare.add_argument("--image", type=Path, help="Already-aligned face image; otherwise use deterministic random input")
    compare.add_argument("--color-order", choices=("RGB", "BGR"))

    return parser


def _load(args):
    return load_backbone(
        args.checkpoint, device=args.device, input_size=args.input_size,
        num_layers=args.num_layers, mode=args.mode,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-lfw":
        from .evaluation.dataset import prepare_lfw_bin
        print(prepare_lfw_bin(args.archive, args.output, overwrite=args.overwrite))
        return 0

    baseline = _load(args)
    if args.command == "lfw":
        from .evaluation.lfw import extract_embeddings, evaluate_verification, load_lfw_bin
        images, same = load_lfw_bin(args.lfw_bin, color_order=args.color_order)
        config = PreprocessConfig(size=args.input_size, color_order=args.color_order)
        variants = list(dict.fromkeys(["baseline", *args.variants]))
        result = {"dataset_pairs": len(same), "color_order": args.color_order,
                  "baseline_prelu": prelu_parameter_report(baseline), "models": {}}
        reference = None
        for variant in variants:
            model, report = make_variant(baseline, variant)
            embeddings = extract_embeddings(model, images, config=config, device=args.device,
                                            batch_size=args.batch_size, num_workers=args.workers)
            row = {"verification": evaluate_verification(embeddings, same), "transformations": report}
            if variant == "baseline":
                reference = embeddings
            else:
                row["agreement_with_baseline"] = compare_embeddings(reference, embeddings)
            result["models"][variant] = row
            if variant != "baseline":
                del model
        output = json.dumps(result, indent=2, ensure_ascii=False)
        print(output)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(output + "\n", encoding="utf-8")
        return 0

    model, report = make_variant(baseline, args.variant)
    if args.command == "compare-variants":
        if args.batch_size < 1:
            raise ValueError("batch-size must be positive")
        sample = torch.randn(args.batch_size, 3, args.input_size, args.input_size,
                             generator=torch.Generator().manual_seed(0)).to(args.device)
        with torch.inference_mode():
            reference = baseline(sample).flatten(start_dim=1).cpu().numpy()
            candidate = model(sample).flatten(start_dim=1).cpu().numpy()
        print(json.dumps({"comparison": compare_embeddings(reference, candidate),
                          "variant": args.variant, "transformations": report}, indent=2))
        return 0
    if args.command == "export":
        from .export import export_onnx
        path = export_onnx(model, args.output, input_size=args.input_size, opset=args.opset)
        print(json.dumps({"onnx": str(path), "variant": args.variant,
                          "transformations": report}, indent=2))
        return 0

    from .export import compare_onnx
    if args.image:
        if not args.color_order:
            raise ValueError("--color-order must be specified when --image is provided")
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("Install the 'eval' extra to decode --image") from exc
        bgr = cv2.imread(str(args.image))
        if bgr is None:
            raise ValueError(f"Could not read image: {args.image}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if args.color_order == "RGB" else bgr
        sample = preprocess_image(image, PreprocessConfig(args.input_size, args.color_order)).unsqueeze(0)
    else:
        sample = torch.randn(1, 3, args.input_size, args.input_size,
                             generator=torch.Generator().manual_seed(0))
    print(json.dumps({"comparison": compare_onnx(model, args.onnx, sample), "variant": args.variant}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
