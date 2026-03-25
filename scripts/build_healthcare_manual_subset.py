#!/usr/bin/env python3
"""
Build a small healthcare-side manual-labeling subset from an existing converted dataset.

Typical use:
- take the AI-Hub raw side-view seed dataset
- keep only rows missing priority keypoints
- sample a limited number of frames across clips
- copy images/labels into a compact folder ready for CVAT/manual review
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from convert_pose_to_healthcare_side import clip_key, frame_index, select_evenly_spaced, write_target_yaml


DEFAULT_PRIORITY_POINTS = [
    "sacrum",
    "tail_base",
    "tail_end",
    "far_rear_hip",
    "far_rear_stifle",
    "far_rear_hock",
    "far_rear_paw",
    "far_front_shoulder",
    "far_front_elbow",
    "far_front_carpus",
    "far_front_paw",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact manual-labeling subset for healthcare-side data")
    parser.add_argument("--input-root", required=True, help="Converted dataset root with labels/, train.txt/val.txt, manual_label_queue.csv")
    parser.add_argument("--output-root", required=True, help="Output root for the manual-labeling subset")
    parser.add_argument(
        "--priority-points",
        default=",".join(DEFAULT_PRIORITY_POINTS),
        help="Comma-separated target keypoints to prioritize",
    )
    parser.add_argument(
        "--sample-per-clip",
        type=int,
        default=1,
        help="How many evenly spaced frames to keep per clip after filtering",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1500,
        help="Optional cap on total selected frames after sampling (0 means no cap)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Which splits to include",
    )
    return parser.parse_args()


def parse_priority_points(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_queue_rows(queue_path: Path, splits: set[str], priority_points: set[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(queue_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") not in splits:
                continue
            missing = set(filter(None, (row.get("missing_points") or "").split(",")))
            hit = sorted(missing & priority_points)
            if not hit:
                continue
            item = dict(row)
            item["priority_missing_points"] = ",".join(hit)
            rows.append(item)
    return rows


def grouped_sample(rows: Sequence[Dict[str, str]], sample_per_clip: int) -> List[Dict[str, str]]:
    if sample_per_clip <= 0:
        return list(rows)

    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        rel_label = Path(row["label"])
        clip = f"{rel_label.parent.as_posix()}/{clip_key(rel_label)}"
        groups[(row["split"], clip)].append(row)

    selected: List[Dict[str, str]] = []
    for _, group_rows in groups.items():
        group_rows = sorted(group_rows, key=lambda row: (frame_index(Path(row["label"])), row["label"]))
        sampled_idx = set(
            select_evenly_spaced(list(range(len(group_rows))), sample_per_clip)
            if group_rows
            else []
        )
        for idx, row in enumerate(group_rows):
            if idx in sampled_idx:
                selected.append(row)
    return sorted(selected, key=lambda row: (row["split"], row["label"]))


def limit_rows(rows: Sequence[Dict[str, str]], max_rows: int) -> List[Dict[str, str]]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return list(rows)
    keep_indices = set(select_evenly_spaced(list(range(len(rows))), max_rows))
    return [row for idx, row in enumerate(rows) if idx in keep_indices]


def build_image_lookup(input_root: Path, splits: Iterable[str]) -> Dict[Tuple[str, str], str]:
    lookup: Dict[Tuple[str, str], str] = {}
    for split in splits:
        index_path = input_root / f"{split}.txt"
        if not index_path.exists():
            continue
        for line in index_path.read_text(encoding="utf-8").splitlines():
            abs_path = line.strip()
            if not abs_path:
                continue
            abs_path_obj = Path(abs_path)
            parts = abs_path_obj.parts
            for n in range(1, min(8, len(parts)) + 1):
                suffix = Path(*parts[-n:]).as_posix()
                lookup.setdefault((split, suffix), abs_path)
    return lookup


def resolve_image_path(row: Dict[str, str], lookup: Dict[Tuple[str, str], str]) -> Optional[Path]:
    split = row["split"]
    rel_image = row["image"].replace("\\", "/")
    candidates = [rel_image]
    image_path_obj = Path(rel_image)
    parts = image_path_obj.parts
    for n in range(1, min(8, len(parts)) + 1):
        candidates.append(Path(*parts[-n:]).as_posix())

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        abs_path = lookup.get((split, candidate))
        if abs_path:
            return Path(abs_path)
    return None


def write_subset_queue(output_root: Path, rows: Sequence[Dict[str, str]]) -> None:
    queue_path = output_root / "manual_label_queue.csv"
    fieldnames = ["split", "label", "image", "view", "missing_points", "priority_missing_points"]
    with open(queue_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_split_index(output_root: Path, split: str, image_paths: Sequence[str]) -> None:
    path = output_root / f"{split}.txt"
    path.write_text("\n".join(image_paths) + ("\n" if image_paths else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    splits = set(args.splits)
    priority_points = set(parse_priority_points(args.priority_points))
    queue_rows = read_queue_rows(input_root / "manual_label_queue.csv", splits, priority_points)
    queue_rows = grouped_sample(queue_rows, args.sample_per_clip)
    queue_rows = limit_rows(queue_rows, args.max_rows)

    image_lookup = build_image_lookup(input_root, splits)
    write_target_yaml(output_root)

    train_images: List[str] = []
    val_images: List[str] = []
    copied = 0
    missing_images = 0

    for row in queue_rows:
        split = row["split"]
        rel_label = Path(row["label"])
        src_label = input_root / "labels" / split / rel_label
        if not src_label.exists():
            continue

        abs_image = resolve_image_path(row, image_lookup)
        if abs_image is None or not abs_image.exists():
            missing_images += 1
            continue

        dst_label = output_root / "labels" / split / rel_label
        dst_image = output_root / "images" / split / Path(row["image"])
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_label, dst_label)
        shutil.copy2(abs_image, dst_image)
        rel_image_out = dst_image.relative_to(output_root).as_posix()
        if split == "train":
            train_images.append(rel_image_out)
        elif split == "val":
            val_images.append(rel_image_out)
        copied += 1

    copied_rows = [row for row in queue_rows if (output_root / "labels" / row["split"] / row["label"]).exists()]
    write_subset_queue(output_root, copied_rows)
    write_split_index(output_root, "train", train_images)
    write_split_index(output_root, "val", val_images)

    print(
        "Done. "
        f"selected_rows={len(queue_rows)} "
        f"copied_rows={copied} "
        f"missing_images={missing_images} "
        f"train_images={len(train_images)} "
        f"val_images={len(val_images)}"
    )
    print(f"Wrote subset queue to: {output_root / 'manual_label_queue.csv'}")


if __name__ == "__main__":
    main()
