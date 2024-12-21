"""Choose a reproducible WIDER FACE validation subset outside the repository."""

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image


def read_annotations(path):
    lines = path.read_text().splitlines()
    cursor = 0
    while cursor < len(lines):
        name = lines[cursor].strip()
        count = int(lines[cursor + 1])
        cursor += 2
        faces = []
        for _ in range(count):
            values = list(map(int, lines[cursor].split()))
            cursor += 1
            faces.append({"box": values[:4], "blur": values[4],
                          "invalid": values[7], "occlusion": values[8],
                          "pose": values[9]})
        yield name, faces


def category(faces, width, height):
    valid = [f for f in faces if not f["invalid"] and f["box"][2] > 0 and f["box"][3] > 0]
    projected = [min(f["box"][2:4]) * 128 / max(width, height) for f in valid]
    if len(valid) == 1 and projected[0] >= 16:
        return "single_large"
    if 2 <= len(valid) <= 5 and sum(size >= 8 for size in projected) >= 2:
        return "multi"
    if len(valid) >= 10:
        return "crowded"
    if any((f["occlusion"] == 2 or f["pose"] == 1) and size >= 8
           for f, size in zip(valid, projected)):
        return "occluded_or_pose"
    if valid and max(projected) < 8:
        return "small"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=8)
    args = parser.parse_args()
    groups = {key: [] for key in ("single_large", "multi", "crowded", "occluded_or_pose", "small")}
    for relative, faces in read_annotations(args.annotations):
        source = args.images_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        with Image.open(source) as image:
            width, height = image.size
        key = category(faces, width, height)
        if key:
            groups[key].append((relative, faces, width, height))

    args.output.mkdir(parents=True, exist_ok=True)
    originals = args.output / "originals"
    originals.mkdir(exist_ok=True)
    selected = []
    used = set()
    rng = random.Random(20261115)
    for key, candidates in groups.items():
        rng.shuffle(candidates)
        chosen = 0
        for relative, faces, width, height in candidates:
            if relative in used:
                continue
            used.add(relative)
            image_id = f"{key}_{chosen:02d}"
            shutil.copy2(args.images_root / relative, originals / f"{image_id}.jpg")
            selected.append({"image": image_id, "category": key,
                             "source": relative, "width": width, "height": height,
                             "faces": faces})
            chosen += 1
            if chosen == args.per_category:
                break
        print(f"{key}: {chosen} / {len(candidates)} candidates")
    (args.output / "selected_gt.json").write_text(json.dumps(selected, indent=2))
    print(f"selected {len(selected)} images; metadata: {args.output / 'selected_gt.json'}")


if __name__ == "__main__":
    main()
