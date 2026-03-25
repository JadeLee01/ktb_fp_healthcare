#!/usr/bin/env python3
"""
Convert the raw AI-Hub dog gait dataset directly into the healthcare side-view 23-keypoint schema.

This script reads the original JSON annotations under:
  - Training/02.라벨링데이터
  - Validation/02.라벨링데이터

and matches them to the corresponding images under:
  - Training/01.원천데이터
  - Validation/01.원천데이터

Only Left/Right view folders are used.
The output is YOLO pose format in the same healthcare-side-final-23 schema used by
convert_pose_to_healthcare_side.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from convert_pose_to_healthcare_side import (
    IMAGE_EXTS,
    TARGET_INDEX,
    TARGET_KPT_NAMES,
    clip_key,
    format_yolo_pose_line,
    frame_index,
    select_evenly_spaced,
    write_target_yaml,
)


SPLIT_INFO = {
    "train": ("Training", "01.원천데이터", "02.라벨링데이터"),
    "val": ("Validation", "01.원천데이터", "02.라벨링데이터"),
}

RAW_TO_TARGET_EXACT = {
    "T13 Spinous precess": "t13_spinous_process",
    "Iliac crest": "iliac_crest",
    "Acromion/Greater tubercle": "near_front_shoulder",
    "Lateral humeral epicondyle": "near_front_elbow",
    "Ulnar styloid process": "near_front_carpus",
    "Distal lateral aspect of fifth metacarpal bone": "near_front_paw",
    "Femoral greater trochanter": "near_rear_hip",
    "Femorotibial joint": "near_rear_stifle",
    "Lateral malleolus of the distal tibia": "near_rear_hock",
    "Distal lateral aspect of the fifth metatarsus": "near_rear_paw",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw AI-Hub Left/Right gait JSON annotations into healthcare-side-final-23"
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Raw AI-Hub dataset root, for example: 60.반려견_보행영상_기반_건강관리_데이터",
    )
    parser.add_argument("--output-root", required=True, help="Output root for the converted dataset")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(SPLIT_INFO.keys()),
        default=["train", "val"],
        help="Dataset splits to convert",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into the output dataset. Default is labels only with absolute train.txt/val.txt paths.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of JSON files per split for smoke tests",
    )
    parser.add_argument(
        "--sample-per-clip",
        type=int,
        default=0,
        help="Optional max number of evenly spaced frames to keep per clip",
    )
    parser.add_argument(
        "--use-dorsal-scapular-spine-as-withers-proxy",
        action="store_true",
        help="Map raw 'Dorsal scapular spine' to target 'withers' as a proxy to reduce missing anchors.",
    )
    return parser.parse_args()


def find_image_for_json(image_split_root: Path, rel_json: Path) -> Optional[Path]:
    image_parent = image_split_root / rel_json.parent
    stem = rel_json.stem
    for ext in IMAGE_EXTS:
        candidate = image_parent / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def build_bbox(keypoints: Sequence[Tuple[float, float, int]]) -> Optional[Tuple[float, float, float, float]]:
    visible = [(x, y) for x, y, v in keypoints if v > 0]
    if not visible:
        return None

    xs = [x for x, _ in visible]
    ys = [y for _, y in visible]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w_span = max_x - min_x
    h_span = max_y - min_y
    pad_x = max(w_span * 0.15, 0.02)
    pad_y = max(h_span * 0.15, 0.02)
    b_min_x = max(0.0, min_x - pad_x)
    b_max_x = min(1.0, max_x + pad_x)
    b_min_y = max(0.0, min_y - pad_y)
    b_max_y = min(1.0, max_y + pad_y)
    cx = (b_min_x + b_max_x) / 2
    cy = (b_min_y + b_max_y) / 2
    bw = max(0.0, b_max_x - b_min_x)
    bh = max(0.0, b_max_y - b_min_y)
    return cx, cy, bw, bh


def convert_raw_annotations(
    anns: Sequence[Dict[str, str]],
    use_dorsal_proxy: bool,
) -> Tuple[List[Tuple[float, float, int]], List[str]]:
    target = [(0.0, 0.0, 0) for _ in TARGET_KPT_NAMES]

    def put(target_name: str, x: float, y: float) -> None:
        target[TARGET_INDEX[target_name]] = (x, y, 2)

    for ann in anns:
        label = (ann.get("label") or "").strip()
        if not label:
            continue

        try:
            x = float(ann["x"])
            y = float(ann["y"])
        except (KeyError, TypeError, ValueError):
            continue

        target_name = RAW_TO_TARGET_EXACT.get(label)
        if target_name is not None:
            put(target_name, x, y)
            continue

        if use_dorsal_proxy and label == "Dorsal scapular spine":
            put("withers", x, y)

    missing = [name for name, (_, _, v) in zip(TARGET_KPT_NAMES, target) if v == 0]
    return target, missing


def write_split_index(output_root: Path, split: str, image_paths: Sequence[str]) -> None:
    index_path = output_root / f"{split}.txt"
    index_path.write_text("\n".join(image_paths) + ("\n" if image_paths else ""), encoding="utf-8")


def write_manual_queue(output_root: Path, split: str, rows: Sequence[Dict[str, str]]) -> None:
    queue_path = output_root / f"manual_label_queue_{split}.csv"
    fieldnames = ["split", "label", "image", "view", "missing_points"]
    with open(queue_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def merge_queues(output_root: Path, splits: Sequence[str]) -> None:
    rows: List[Dict[str, str]] = []
    for split in splits:
        queue_path = output_root / f"manual_label_queue_{split}.csv"
        if not queue_path.exists():
            continue
        with open(queue_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows.extend(reader)

    queue_path = output_root / "manual_label_queue.csv"
    fieldnames = ["split", "label", "image", "view", "missing_points"]
    with open(queue_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_json_candidates(label_split_root: Path) -> List[Tuple[Path, str]]:
    candidates: List[Tuple[Path, str]] = []
    for json_path in sorted(label_split_root.rglob("*.json")):
        view = json_path.parent.name
        if view == "Left":
            candidates.append((json_path, "left"))
        elif view == "Right":
            candidates.append((json_path, "right"))
    return candidates


def convert_split(
    input_root: Path,
    output_root: Path,
    split: str,
    copy_images: bool,
    limit: int,
    sample_per_clip: int,
    use_dorsal_proxy: bool,
) -> Dict[str, int]:
    split_dir_name, image_dir_name, label_dir_name = SPLIT_INFO[split]
    base_dir = input_root / "3.개방데이터" / "1.데이터" / split_dir_name
    image_split_root = base_dir / image_dir_name
    label_split_root = base_dir / label_dir_name

    out_label_dir = output_root / "labels" / split
    out_image_dir = output_root / "images" / split
    out_label_dir.mkdir(parents=True, exist_ok=True)
    if copy_images:
        out_image_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "converted": 0,
        "skipped_no_image": 0,
        "skipped_invalid": 0,
    }

    candidates = collect_json_candidates(label_split_root)
    if limit > 0:
        candidates = candidates[:limit]

    if sample_per_clip > 0:
        grouped: Dict[str, List[Tuple[Path, str]]] = {}
        for json_path, view in candidates:
            rel_json = json_path.relative_to(label_split_root)
            grouped.setdefault(clip_key(rel_json), []).append((json_path, view))

        sampled: List[Tuple[Path, str]] = []
        for entries in grouped.values():
            entries = sorted(entries, key=lambda item: (frame_index(item[0].relative_to(label_split_root)), item[0].name))
            sampled_paths = set(select_evenly_spaced([entry[0] for entry in entries], sample_per_clip))
            sampled.extend([entry for entry in entries if entry[0] in sampled_paths])
        candidates = sorted(sampled, key=lambda item: item[0])

    manual_rows: List[Dict[str, str]] = []
    split_images: List[str] = []

    for json_path, view in candidates:
        rel_json = json_path.relative_to(label_split_root)
        image_path = find_image_for_json(image_split_root, rel_json)
        if image_path is None:
            stats["skipped_no_image"] += 1
            continue

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            stats["skipped_invalid"] += 1
            continue

        anns = data.get("annotation_info")
        if not isinstance(anns, list):
            stats["skipped_invalid"] += 1
            continue

        target_keypoints, missing = convert_raw_annotations(anns, use_dorsal_proxy=use_dorsal_proxy)
        bbox = build_bbox(target_keypoints)
        if bbox is None:
            stats["skipped_invalid"] += 1
            continue

        out_label_path = (out_label_dir / rel_json).with_suffix(".txt")
        out_label_path.parent.mkdir(parents=True, exist_ok=True)
        prefix = ["0", f"{bbox[0]:.6f}", f"{bbox[1]:.6f}", f"{bbox[2]:.6f}", f"{bbox[3]:.6f}"]
        out_label_path.write_text(format_yolo_pose_line(prefix, target_keypoints) + "\n", encoding="utf-8")

        if copy_images:
            out_image_path = out_image_dir / rel_json.parent / image_path.name
            out_image_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_image_path.exists():
                shutil.copy2(image_path, out_image_path)
            split_images.append(out_image_path.relative_to(output_root).as_posix())
        else:
            split_images.append(str(image_path))

        if missing:
            manual_rows.append(
                {
                    "split": split,
                    "label": str(rel_json.with_suffix(".txt")),
                    "image": str((rel_json.parent / image_path.name).as_posix()),
                    "view": view,
                    "missing_points": ",".join(sorted(missing)),
                }
            )

        stats["converted"] += 1

    write_split_index(output_root, split, split_images)
    write_manual_queue(output_root, split, manual_rows)
    return stats


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    write_target_yaml(output_root)

    total = {"converted": 0, "skipped_no_image": 0, "skipped_invalid": 0}
    for split in args.splits:
        stats = convert_split(
            input_root=input_root,
            output_root=output_root,
            split=split,
            copy_images=args.copy_images,
            limit=args.limit,
            sample_per_clip=args.sample_per_clip,
            use_dorsal_proxy=args.use_dorsal_scapular_spine_as_withers_proxy,
        )
        for key in total:
            total[key] += stats[key]
        print(
            f"[{split}] converted={stats['converted']} "
            f"skipped_no_image={stats['skipped_no_image']} "
            f"skipped_invalid={stats['skipped_invalid']}"
        )

    merge_queues(output_root, args.splits)
    print(f"Wrote schema yaml to: {output_root / 'healthcare-side-final-23.yaml'}")
    print(f"Wrote CVAT/Ultralytics yaml to: {output_root / 'data.yaml'}")
    print(f"Wrote manual queue to: {output_root / 'manual_label_queue.csv'}")
    print(
        "Done. "
        f"converted={total['converted']} "
        f"skipped_no_image={total['skipped_no_image']} "
        f"skipped_invalid={total['skipped_invalid']}"
    )


if __name__ == "__main__":
    main()
