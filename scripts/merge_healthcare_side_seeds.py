#!/usr/bin/env python3
"""
Merge multiple healthcare-side-final-23 seed datasets into one training-ready dataset.

Expected source layout for each dataset root:
  - labels/train
  - labels/val
  - train.txt
  - val.txt
  - manual_label_queue.csv

The merged dataset:
  - keeps label files under labels/{split}/{source_name}/...
  - writes train.txt/val.txt by concatenating source image paths
  - writes a merged manual_label_queue.csv with a source column
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

from convert_pose_to_healthcare_side import write_target_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple healthcare-side-final-23 seed datasets")
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="One or more source dataset roots already converted to healthcare-side-final-23",
    )
    parser.add_argument("--output-root", required=True, help="Merged dataset output root")
    parser.add_argument(
        "--source-names",
        nargs="+",
        help="Optional source aliases. Defaults to each source directory name.",
    )
    return parser.parse_args()


def read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_unique_names(source_roots: Sequence[Path], source_names: Sequence[str] | None) -> List[str]:
    if source_names is None:
        names = [root.name for root in source_roots]
    else:
        names = list(source_names)

    if len(names) != len(source_roots):
        raise ValueError("--source-names length must match --sources length")
    if len(set(names)) != len(names):
        raise ValueError("Source names must be unique")
    return names


def copy_label_tree(source_root: Path, output_root: Path, split: str, source_name: str) -> int:
    src_label_dir = source_root / "labels" / split
    dst_label_dir = output_root / "labels" / split / source_name
    count = 0
    if not src_label_dir.exists():
        return count

    for src_label_path in sorted(src_label_dir.rglob("*.txt")):
        rel_label = src_label_path.relative_to(src_label_dir)
        dst_label_path = dst_label_dir / rel_label
        dst_label_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_label_path, dst_label_path)
        count += 1
    return count


def merge_split_index(source_root: Path, split: str) -> List[str]:
    return read_lines(source_root / f"{split}.txt")


def merge_manual_queue(source_root: Path, source_name: str) -> List[Dict[str, str]]:
    queue_path = source_root / "manual_label_queue.csv"
    if not queue_path.exists():
        return []

    rows: List[Dict[str, str]] = []
    with open(queue_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            row["source"] = source_name
            if row.get("label"):
                row["label"] = str(Path(source_name) / row["label"])
            rows.append(row)
    return rows


def write_split_index(output_root: Path, split: str, lines: Sequence[str]) -> None:
    path = output_root / f"{split}.txt"
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_manual_queue(output_root: Path, rows: Sequence[Dict[str, str]]) -> None:
    queue_path = output_root / "manual_label_queue.csv"
    fieldnames = ["source", "split", "label", "image", "view", "missing_points"]
    with open(queue_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    source_roots = [Path(path) for path in args.sources]
    source_names = ensure_unique_names(source_roots, args.source_names)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    write_target_yaml(output_root)

    train_lines: List[str] = []
    val_lines: List[str] = []
    manual_rows: List[Dict[str, str]] = []
    copied_labels = {"train": 0, "val": 0}

    for source_root, source_name in zip(source_roots, source_names):
        copied_labels["train"] += copy_label_tree(source_root, output_root, "train", source_name)
        copied_labels["val"] += copy_label_tree(source_root, output_root, "val", source_name)
        train_lines.extend(merge_split_index(source_root, "train"))
        val_lines.extend(merge_split_index(source_root, "val"))
        manual_rows.extend(merge_manual_queue(source_root, source_name))

    write_split_index(output_root, "train", train_lines)
    write_split_index(output_root, "val", val_lines)
    write_manual_queue(output_root, manual_rows)

    print(
        "Done. "
        f"train_labels={copied_labels['train']} "
        f"val_labels={copied_labels['val']} "
        f"train_images={len(train_lines)} "
        f"val_images={len(val_lines)} "
        f"queue_rows={len(manual_rows)}"
    )


if __name__ == "__main__":
    main()
