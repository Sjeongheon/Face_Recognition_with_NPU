"""Select a deterministic random sample of preprocessed WIDER FACE images."""

import argparse
import json
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261115)
    args = parser.parse_args()
    items = json.loads((args.prepared_root / "selected_gt.json").read_text())
    if not 0 < args.count <= len(items):
        parser.error("count must be between 1 and the number of prepared images")
    rng = random.Random(args.seed)
    selected = sorted(rng.sample(items, args.count), key=lambda item: item["image"])
    images = args.output / "images"
    images.mkdir(parents=True, exist_ok=True)
    for item in selected:
        name = f"{item['image']}.png"
        shutil.copy2(args.prepared_root / "images" / name, images / name)
    (args.output / "selected_gt.json").write_text(json.dumps(selected))
    print(f"selected {len(selected)} / {len(items)} images with seed {args.seed}")


if __name__ == "__main__":
    main()
