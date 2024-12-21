"""Build a face-focused, disjoint ENLIGHT calibration set with provenance."""

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path


def largest_face_size(item):
    ratio = 128 / max(item["width"], item["height"])
    sizes = [min(face["box"][2:4]) * ratio for face in item["faces"]
             if not face["invalid"] and min(face["box"][2:4]) > 0]
    return max(sizes, default=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--evaluation-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261115)
    args = parser.parse_args()
    all_items = json.loads((args.prepared_root / "selected_gt.json").read_text())
    evaluation_ids = {item["image"] for item in json.loads(args.evaluation_list.read_text())}
    pools = {"large_ge24": [], "medium_16_24": [], "small_8_16": []}
    for item in all_items:
        if item["image"] in evaluation_ids:
            continue
        size = largest_face_size(item)
        if size >= 24:
            pools["large_ge24"].append(item)
        elif size >= 16:
            pools["medium_16_24"].append(item)
        elif size >= 8:
            pools["small_8_16"].append(item)
    rng = random.Random(args.seed)
    selected = []
    for group, count in (("large_ge24", 50), ("medium_16_24", 30), ("small_8_16", 20)):
        if len(pools[group]) < count:
            raise ValueError(f"{group}: only {len(pools[group])} candidates for {count}")
        chosen = rng.sample(pools[group], count)
        selected.extend((group, item) for item in chosen)
        print(f"{group}: {count} of {len(pools[group])} candidates")
    image_dir = args.output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, (group, item) in enumerate(selected):
        source = args.prepared_root / "images" / f"{item['image']}.png"
        destination = image_dir / f"calib_{index:04d}.png"
        shutil.copy2(source, destination)
        manifest.append({"calibration_file": destination.name, "source_id": item["image"],
                         "source": item["source"], "group": group,
                         "largest_face_size_at_128": largest_face_size(item),
                         "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    (args.output / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"selected {len(manifest)} calibration images; overlap with evaluation: 0")


if __name__ == "__main__":
    main()
