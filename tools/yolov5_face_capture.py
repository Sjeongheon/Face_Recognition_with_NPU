"""Save every raw YOLOv5-Face head from the ENLIGHT simulator.

Run inside the ENLIGHT container with its Python environment. This wrapper leaves
the vendor SDK untouched and replaces only its default first-output saver.
"""

import argparse
import runpy
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()

    sdk_script = args.sdk_root / "EnlightSDK" / "enlight_sim.py"
    if not sdk_script.is_file():
        parser.error(f"simulator script not found: {sdk_script}")
    sys.path.insert(0, str(sdk_script.parent))

    from simulator.helpers.basic_simulator import BasicSimulator

    def save_all_heads(self, outputs, _target):
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
            raise ValueError(f"expected exactly three raw heads, got {type(outputs)} / {len(outputs)}")
        base = Path(self.data_generator.get_input_path())
        expected = ((48, 16, 16), (48, 8, 8), (48, 4, 4))
        for level, (output, shape) in enumerate(zip(outputs, expected), start=3):
            array = output.detach().cpu().numpy()
            if array.shape != (1, *shape):
                raise ValueError(f"P{level} shape {array.shape}, expected {(1, *shape)}")
            path = base.parent / f"{base.name}_p{level}.npy"
            np.save(path, array)
            print(f"P{level}: {path} {array.shape}")

    BasicSimulator.handle_output = save_all_heads
    sys.argv = [str(sdk_script), str(args.model), "--inputs", str(args.inputs),
                "--result-root", str(args.result_root)]
    runpy.run_path(str(sdk_script), run_name="__main__")


if __name__ == "__main__":
    main()
