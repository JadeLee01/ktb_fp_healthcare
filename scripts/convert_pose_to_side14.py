#!/usr/bin/env python3
"""
Convert a pose dataset into a side-view 14-keypoint gait schema.

Supported sources:
- ultralytics24: the 24-keypoint dog pose schema in dog-pose.yaml
- integrated34: the merged AI-Hub schema in dataset_yolo/integrated_dog_pose.yaml

The output dataset is YOLO pose format with 14 keypoints:
1. withers
2. t13_spinous_process
3. sacrum
4. tail_base
5. near_rear_hip
6. near_rear_stifle
7. near_rear_hock
8. near_rear_paw
9. far_rear_hip
10. far_rear_stifle
11. far_rear_hock
12. far_rear_paw
13. near_front_paw
14. far_front_paw

The script automatically copies keypoints that already exist in the source schema and
leaves unknown points as 0 0 0. It also writes a CSV queue listing which files still
need manual labeling.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP")

TARGET_KPT_NAMES = [
    "withers",
    "t13_spinous_process",
    "sacrum",
    "tail_base",
    "near_rear_hip",
    "near_rear_stifle",
    "near_rear_hock",
    "near_rear_paw",
    "far_rear_hip",
    "far_rear_stifle",
    "far_rear_hock",
    "far_rear_paw",
    "near_front_paw",
    "far_front_paw",
]
TARGET_INDEX = {name: idx for idx, name in enumerate(TARGET_KPT_NAMES)}

ULTRALYTICS24_NAMES = [
    "front_left_paw",
    "front_left_knee",
    "front_left_elbow",
    "rear_left_paw",
    "rear_left_knee",
    "rear_left_elbow",
    "front_right_paw",
    "front_right_knee",
    "front_right_elbow",
    "rear_right_paw",
    "rear_right_knee",
    "rear_right_elbow",
    "tail_start",
    "tail_end",
    "left_ear_base",
    "right_ear_base",
    "nose",
    "chin",
    "left_ear_tip",
    "right_ear_tip",
    "left_eye",
    "right_eye",
    "withers",
    "throat",
]

INTEGRATED34_NAMES = [
    "Acromion/Greater tubercle",
    "Chin",
    "Distal lateral aspect of fifth metacarpal bone",
    "Distal lateral aspect of the fifth metatarsus",
    "Dorsal scapular spine",
    "Ear",
    "Femoral greater trochanter",
    "Femorotibial joint",
    "Iliac crest",
    "L_Acromion/Greater tubercle",
    "L_Distal lateral aspect of fifth metacarpal bone",
    "L_Distal lateral aspect of the fifth metatarsus",
    "L_Femoral greater trochanter",
    "L_Femorotibial joint",
    "L_Lateral humeral epicondyle",
    "L_Lateral malleolus of the distal tibia",
    "L_Ulnar styloid process",
    "Lateral humeral epicondyle",
    "Lateral malleolus of the distal tibia",
    "Nose",
    "R_Acromion/Greater tubercle",
    "R_Distal lateral aspect of fifth metacarpal bone",
    "R_Distal lateral aspect of the fifth metatarsus",
    "R_Femoral greater trochanter",
    "R_Femorotibial joint",
    "R_Lateral humeral epicondyle",
    "R_Lateral malleolus of the distal tibia",
    "R_Ulnar styloid process",
    "Sacrum",
    "T13 Spinous precess",
    "Tail end",
    "Tail start",
    "Ulnar styloid process",
    "Withers",
]

SOURCE_SCHEMAS = {
    "ultralytics24": ULTRALYTICS24_NAMES,
    "integrated34": INTEGRATED34_NAMES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert YOLO pose labels into the side14 gait schema")
    parser.add_argument("--input-root", required=True, help="Dataset root containing images/{split} and labels/{split}")
    parser.add_argument("--output-root", required=True, help="Output root for the side14 dataset")
    parser.add_argument(
        "--source-schema",
        choices=sorted(SOURCE_SCHEMAS.keys()),
        required=True,
        help="Source keypoint schema",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to convert, for example: train val",
    )
    parser.add_argument(
        "--view-source",
        choices=["auto", "parent", "filename", "manifest", "fixed-left", "fixed-right"],
        default="auto",
        help="How to determine whether an image is left-view or right-view",
    )
    parser.add_argument(
        "--manifest",
        help="Optional CSV with columns path,view[,include]. Used when view-source is manifest or auto.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into the output dataset. Default is labels only.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of labels per split for smoke tests",
    )
    parser.add_argument(
        "--sample-per-clip",
        type=int,
        default=0,
        help="Optional max number of evenly spaced frames to keep per clip",
    )
    return parser.parse_args()


def load_manifest(path: Optional[str]) -> Dict[str, Tuple[str, bool]]:
    if not path:
        return {}

    manifest: Dict[str, Tuple[str, bool]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_key = row.get("path") or row.get("image") or row.get("label") or row.get("stem")
            raw_view = (row.get("view") or "").strip().lower()
            raw_include = (row.get("include") or "1").strip().lower()
            if not raw_key or raw_view not in {"left", "right"}:
                continue
            include = raw_include not in {"0", "false", "no", "n"}
            key = normalize_manifest_key(raw_key)
            manifest[key] = (raw_view, include)
    return manifest


def normalize_manifest_key(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    if "." in Path(normalized).name:
        normalized = str(Path(normalized).with_suffix(""))
    return normalized.lower()


def infer_view(rel_label: Path, mode: str, manifest: Dict[str, Tuple[str, bool]]) -> Optional[str]:
    manifest_candidates = [
        normalize_manifest_key(str(rel_label)),
        normalize_manifest_key(str(rel_label.with_suffix(""))),
        normalize_manifest_key(rel_label.name),
        normalize_manifest_key(rel_label.stem),
    ]

    if mode in {"auto", "manifest"}:
        for candidate in manifest_candidates:
            if candidate in manifest:
                view, include = manifest[candidate]
                return view if include else None
        if mode == "manifest":
            return None

    if mode in {"auto", "filename"}:
        view = infer_view_from_texts([rel_label.stem, rel_label.name])
        if view:
            return view
        if mode == "filename":
            return None

    if mode in {"auto", "parent"}:
        view = infer_view_from_texts(list(rel_label.parts[:-1]))
        if view:
            return view
        if mode == "parent":
            return None

    if mode == "fixed-left":
        return "left"
    if mode == "fixed-right":
        return "right"
    return None


def infer_view_from_texts(values: Iterable[str]) -> Optional[str]:
    found = set()
    for value in values:
        text = value.lower()
        if has_token(text, "left"):
            found.add("left")
        if has_token(text, "right"):
            found.add("right")
    if len(found) == 1:
        return next(iter(found))
    return None


def has_token(text: str, token: str) -> bool:
    if text == token:
        return True
    if text.startswith(token + "_") or text.endswith("_" + token):
        return True
    if text.startswith(token + "-") or text.endswith("-" + token):
        return True
    pattern = rf"(^|[^a-z0-9]){re.escape(token)}([^a-z0-9]|$)"
    return re.search(pattern, text) is not None


def find_image_for_label(image_split_dir: Path, rel_label: Path) -> Optional[Path]:
    parent = image_split_dir / rel_label.parent
    stem = rel_label.stem
    for ext in IMAGE_EXTS:
        candidate = parent / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def clip_key(rel_label: Path) -> str:
    stem_parts = rel_label.stem.split("_")
    if stem_parts and stem_parts[-1].isdigit():
        return "_".join(stem_parts[:-1])
    return str(rel_label.with_suffix(""))


def frame_index(rel_label: Path) -> int:
    stem_parts = rel_label.stem.split("_")
    if stem_parts and stem_parts[-1].isdigit():
        return int(stem_parts[-1])
    return -1


def select_evenly_spaced(items: Sequence[Path], count: int) -> List[Path]:
    if count <= 0 or len(items) <= count:
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]

    selected_indices = set()
    for step in range(count):
        raw_idx = round(step * (len(items) - 1) / (count - 1))
        selected_indices.add(raw_idx)
    return [items[idx] for idx in sorted(selected_indices)]


def parse_yolo_pose_line(line: str, kpt_count: int) -> Optional[Tuple[List[str], List[Tuple[float, float, int]]]]:
    parts = line.strip().split()
    expected = 5 + (kpt_count * 3)
    if len(parts) < expected:
        return None

    prefix = parts[:5]
    kpt_tokens = parts[5:expected]
    keypoints: List[Tuple[float, float, int]] = []
    for idx in range(kpt_count):
        x = float(kpt_tokens[idx * 3])
        y = float(kpt_tokens[idx * 3 + 1])
        v = int(float(kpt_tokens[idx * 3 + 2]))
        keypoints.append((x, y, v))
    return prefix, keypoints


def format_yolo_pose_line(prefix: Sequence[str], keypoints: Sequence[Tuple[float, float, int]]) -> str:
    body: List[str] = list(prefix)
    for x, y, v in keypoints:
        body.extend([f"{x:.6f}", f"{y:.6f}", str(v)])
    return " ".join(body)


def convert_keypoints(
    source_keypoints: Sequence[Tuple[float, float, int]],
    source_schema: str,
    view: str,
) -> Tuple[List[Tuple[float, float, int]], List[str]]:
    target = [(0.0, 0.0, 0) for _ in TARGET_KPT_NAMES]
    source_index = {name: idx for idx, name in enumerate(SOURCE_SCHEMAS[source_schema])}

    def put(target_name: str, source_name: Optional[str]) -> bool:
        if source_name is None:
            return False
        source_idx = source_index.get(source_name)
        if source_idx is None or source_idx >= len(source_keypoints):
            return False
        x, y, v = source_keypoints[source_idx]
        if v <= 0 or (x == 0.0 and y == 0.0):
            return False
        target[TARGET_INDEX[target_name]] = (x, y, v)
        return True

    if source_schema == "ultralytics24":
        near_prefix = "left" if view == "left" else "right"
        far_prefix = "right" if view == "left" else "left"

        put("withers", "withers")
        put("tail_base", "tail_start")
        put("near_rear_stifle", f"rear_{near_prefix}_knee")
        put("near_rear_hock", f"rear_{near_prefix}_elbow")
        put("near_rear_paw", f"rear_{near_prefix}_paw")
        put("far_rear_stifle", f"rear_{far_prefix}_knee")
        put("far_rear_hock", f"rear_{far_prefix}_elbow")
        put("far_rear_paw", f"rear_{far_prefix}_paw")
        put("near_front_paw", f"front_{near_prefix}_paw")
        put("far_front_paw", f"front_{far_prefix}_paw")

    elif source_schema == "integrated34":
        near_prefix = "L_" if view == "left" else "R_"
        far_prefix = "R_" if view == "left" else "L_"

        put("withers", "Withers")
        put("t13_spinous_process", "T13 Spinous precess")
        put("sacrum", "Sacrum")
        put("tail_base", "Tail start")
        put("near_rear_hip", f"{near_prefix}Femoral greater trochanter")
        put("near_rear_stifle", f"{near_prefix}Femorotibial joint")
        put("near_rear_hock", f"{near_prefix}Lateral malleolus of the distal tibia")
        put("near_rear_paw", f"{near_prefix}Distal lateral aspect of the fifth metatarsus")
        put("far_rear_hip", f"{far_prefix}Femoral greater trochanter")
        put("far_rear_stifle", f"{far_prefix}Femorotibial joint")
        put("far_rear_hock", f"{far_prefix}Lateral malleolus of the distal tibia")
        put("far_rear_paw", f"{far_prefix}Distal lateral aspect of the fifth metatarsus")
        put("near_front_paw", f"{near_prefix}Distal lateral aspect of fifth metacarpal bone")
        put("far_front_paw", f"{far_prefix}Distal lateral aspect of fifth metacarpal bone")
    else:
        raise ValueError(f"Unsupported source schema: {source_schema}")

    missing = [name for name, (_, _, v) in zip(TARGET_KPT_NAMES, target) if v == 0]
    return target, missing


def write_target_yaml(output_root: Path) -> None:
    yaml_path = output_root / "side14-pose.yaml"
    lines = [
        f"path: {output_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "",
        "kpt_shape: [14, 3]",
        "",
        "names:",
        "  0: dog",
        "",
        "kpt_names:",
        "  0:",
    ]
    for name in TARGET_KPT_NAMES:
        lines.append(f"    - {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manual_queue(output_root: Path, rows: Sequence[Dict[str, str]]) -> None:
    queue_path = output_root / "manual_label_queue.csv"
    fieldnames = ["split", "label", "image", "view", "missing_points"]
    with open(queue_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def convert_split(
    input_root: Path,
    output_root: Path,
    split: str,
    source_schema: str,
    view_source: str,
    manifest: Dict[str, Tuple[str, bool]],
    copy_images: bool,
    limit: int,
    sample_per_clip: int,
) -> Dict[str, int]:
    label_dir = input_root / "labels" / split
    image_dir = input_root / "images" / split
    out_label_dir = output_root / "labels" / split
    out_image_dir = output_root / "images" / split
    out_label_dir.mkdir(parents=True, exist_ok=True)
    if copy_images:
        out_image_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "converted": 0,
        "skipped_no_view": 0,
        "skipped_no_image": 0,
        "skipped_invalid": 0,
    }
    manual_rows: List[Dict[str, str]] = []
    source_kpt_count = len(SOURCE_SCHEMAS[source_schema])

    raw_label_files = sorted(label_dir.rglob("*.txt"))

    side_candidates: List[Tuple[Path, str]] = []
    for label_path in raw_label_files:
        rel_label = label_path.relative_to(label_dir)
        view = infer_view(rel_label, view_source, manifest)
        if view not in {"left", "right"}:
            stats["skipped_no_view"] += 1
            continue
        side_candidates.append((label_path, view))

    if limit > 0:
        side_candidates = side_candidates[:limit]

    if sample_per_clip > 0:
        grouped: Dict[str, List[Tuple[Path, str]]] = {}
        for label_path, view in side_candidates:
            rel_label = label_path.relative_to(label_dir)
            grouped.setdefault(clip_key(rel_label), []).append((label_path, view))

        sampled_candidates: List[Tuple[Path, str]] = []
        for entries in grouped.values():
            entries = sorted(entries, key=lambda item: (frame_index(item[0].relative_to(label_dir)), item[0].name))
            sampled_paths = set(select_evenly_spaced([entry[0] for entry in entries], sample_per_clip))
            sampled_candidates.extend([entry for entry in entries if entry[0] in sampled_paths])
        side_candidates = sorted(sampled_candidates, key=lambda item: item[0])

    for label_path, view in side_candidates:
        rel_label = label_path.relative_to(label_dir)

        image_path = find_image_for_label(image_dir, rel_label)
        if copy_images and image_path is None:
            stats["skipped_no_image"] += 1
            continue

        out_label_path = out_label_dir / rel_label
        out_label_path.parent.mkdir(parents=True, exist_ok=True)

        lines_out: List[str] = []
        missing_union = set()
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parsed = parse_yolo_pose_line(raw_line, source_kpt_count)
            if parsed is None:
                stats["skipped_invalid"] += 1
                continue
            prefix, source_keypoints = parsed
            target_keypoints, missing = convert_keypoints(source_keypoints, source_schema, view)
            lines_out.append(format_yolo_pose_line(prefix, target_keypoints))
            missing_union.update(missing)

        if not lines_out:
            continue

        out_label_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")

        if copy_images and image_path is not None:
            out_image_path = out_image_dir / rel_label.parent / image_path.name
            out_image_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_image_path.exists():
                shutil.copy2(image_path, out_image_path)

        manual_rows.append(
            {
                "split": split,
                "label": str(rel_label),
                "image": str((rel_label.parent / image_path.name) if image_path else rel_label.with_suffix("")),
                "view": view,
                "missing_points": ",".join(sorted(missing_union)),
            }
        )
        stats["converted"] += 1

    write_manual_queue_for_split(output_root, split, manual_rows)
    return stats


def write_manual_queue_for_split(output_root: Path, split: str, rows: Sequence[Dict[str, str]]) -> None:
    queue_path = output_root / f"manual_label_queue_{split}.csv"
    fieldnames = ["split", "label", "image", "view", "missing_points"]
    with open(queue_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def merge_split_queues(output_root: Path, splits: Sequence[str]) -> None:
    rows: List[Dict[str, str]] = []
    for split in splits:
        queue_path = output_root / f"manual_label_queue_{split}.csv"
        if not queue_path.exists():
            continue
        with open(queue_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows.extend(reader)
    write_manual_queue(output_root, rows)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    write_target_yaml(output_root)

    total = {"converted": 0, "skipped_no_view": 0, "skipped_no_image": 0, "skipped_invalid": 0}
    for split in args.splits:
        stats = convert_split(
            input_root=input_root,
            output_root=output_root,
            split=split,
            source_schema=args.source_schema,
            view_source=args.view_source,
            manifest=manifest,
            copy_images=args.copy_images,
            limit=args.limit,
            sample_per_clip=args.sample_per_clip,
        )
        for key in total:
            total[key] += stats[key]
        print(f"[{split}] converted={stats['converted']} skipped_no_view={stats['skipped_no_view']} skipped_no_image={stats['skipped_no_image']} skipped_invalid={stats['skipped_invalid']}")

    merge_split_queues(output_root, args.splits)
    print(f"Wrote schema yaml to: {output_root / 'side14-pose.yaml'}")
    print(f"Wrote manual queue to: {output_root / 'manual_label_queue.csv'}")
    print(
        "Done. "
        f"converted={total['converted']} "
        f"skipped_no_view={total['skipped_no_view']} "
        f"skipped_no_image={total['skipped_no_image']} "
        f"skipped_invalid={total['skipped_invalid']}"
    )


if __name__ == "__main__":
    main()
