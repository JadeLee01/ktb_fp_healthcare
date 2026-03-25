#!/usr/bin/env python3
"""
Build a changed-only training subset from a CVAT YOLO Pose export.

Use case:
  - You labeled only part of a larger task in CVAT (e.g. 150 of 813 images).
  - CVAT exported the whole task.
  - This script compares the export against the original pilot dataset
    and keeps only labels whose contents actually changed.
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from convert_pose_to_healthcare_side import write_target_yaml


@dataclass(frozen=True)
class ExportedLabel:
    split: str
    basename: str
    text: str
    member_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a changed-only subset from a CVAT YOLO Pose export")
    parser.add_argument("--pilot-root", required=True, help="Original nested pilot dataset root")
    parser.add_argument("--export-path", required=True, help="CVAT export zip or extracted directory")
    parser.add_argument("--output-root", required=True, help="Output root for changed-only subset")
    parser.add_argument(
        "--first-n-by-export-order",
        type=int,
        default=0,
        help="If > 0, ignore diffing and keep the first N exported labels in CVAT export order.",
    )
    parser.add_argument(
        "--skip-first-n-by-export-order",
        type=int,
        default=0,
        help="If > 0, skip the first N exported labels before applying first-n or diff logic.",
    )
    return parser.parse_args()


def normalize_label_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def normalize_split(parts: Sequence[str]) -> Optional[str]:
    lowered = [part.lower() for part in parts]
    if "train" in lowered:
        return "train"
    if "val" in lowered or "valid" in lowered or "validation" in lowered:
        return "val"
    return None


def iter_exported_labels(export_path: Path) -> Iterator[ExportedLabel]:
    if export_path.is_dir():
        for path in sorted(export_path.rglob("*.txt")):
            split = normalize_split(path.parts)
            if split is None:
                continue
            if "labels" not in [p.lower() for p in path.parts]:
                continue
            yield ExportedLabel(
                split=split,
                basename=path.name,
                text=path.read_text(encoding="utf-8"),
                member_name=str(path),
            )
        return

    with zipfile.ZipFile(export_path) as zf:
        for name in sorted(zf.namelist()):
            path = Path(name)
            if path.suffix != ".txt":
                continue
            split = normalize_split(path.parts)
            if split is None:
                continue
            if "labels" not in [p.lower() for p in path.parts]:
                continue
            with zf.open(name, "r") as f:
                text = io.TextIOWrapper(f, encoding="utf-8").read()
            yield ExportedLabel(
                split=split,
                basename=path.name,
                text=text,
                member_name=name,
            )


def build_label_map(pilot_root: Path) -> Dict[Tuple[str, str], Path]:
    mapping: Dict[Tuple[str, str], Path] = {}
    for split in ("train", "val"):
        label_root = pilot_root / "labels" / split
        if not label_root.exists():
            continue
        for path in sorted(label_root.rglob("*.txt")):
            key = (split, path.name)
            if key in mapping:
                raise ValueError(f"Duplicate basename in pilot dataset for {key}")
            mapping[key] = path
    return mapping


def image_path_for_label(pilot_root: Path, split: str, label_path: Path) -> Path:
    rel = label_path.relative_to(pilot_root / "labels" / split)
    return pilot_root / "images" / split / rel.with_suffix(".jpg")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_lines(path: Path, lines: Sequence[str]) -> None:
    ensure_parent(path)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_report(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "basename",
                "member_name",
                "target_label",
                "status",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    pilot_root = Path(args.pilot_root).resolve()
    export_path = Path(args.export_path).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not pilot_root.exists():
        raise FileNotFoundError(f"Pilot root not found: {pilot_root}")
    if not export_path.exists():
        raise FileNotFoundError(f"Export path not found: {export_path}")

    label_map = build_label_map(pilot_root)
    write_target_yaml(output_root)

    report_rows: List[Dict[str, str]] = []
    changed = 0
    unchanged = 0
    unmatched = 0
    train_lines: List[str] = []
    val_lines: List[str] = []

    exported_items = list(iter_exported_labels(export_path))
    if args.skip_first_n_by_export_order > 0:
        exported_items = exported_items[args.skip_first_n_by_export_order :]
    if args.first_n_by_export_order > 0:
        exported_items = exported_items[: args.first_n_by_export_order]

    for exported in exported_items:
        key = (exported.split, exported.basename)
        target_label = label_map.get(key)
        if target_label is None:
            unmatched += 1
            report_rows.append(
                {
                    "split": exported.split,
                    "basename": exported.basename,
                    "member_name": exported.member_name,
                    "target_label": "",
                    "status": "unmatched",
                }
            )
            continue

        current_text = normalize_label_text(target_label.read_text(encoding="utf-8"))
        exported_text = normalize_label_text(exported.text)
        rel_label = target_label.relative_to(pilot_root / "labels" / exported.split)

        if args.first_n_by_export_order <= 0 and current_text == exported_text:
            unchanged += 1
            report_rows.append(
                {
                    "split": exported.split,
                    "basename": exported.basename,
                    "member_name": exported.member_name,
                    "target_label": rel_label.as_posix(),
                    "status": "unchanged",
                }
            )
            continue

        src_image = image_path_for_label(pilot_root, exported.split, target_label)
        if not src_image.exists():
            raise FileNotFoundError(f"Image not found for label {target_label}: {src_image}")

        dst_label = output_root / "labels" / exported.split / rel_label
        dst_image = output_root / "images" / exported.split / rel_label.with_suffix(".jpg")
        ensure_parent(dst_label)
        ensure_parent(dst_image)
        dst_label.write_text(exported_text + "\n", encoding="utf-8")
        shutil.copy2(src_image, dst_image)

        rel_image = dst_image.relative_to(output_root).as_posix()
        if exported.split == "train":
            train_lines.append(rel_image)
        else:
            val_lines.append(rel_image)

        changed += 1
        report_rows.append(
            {
                "split": exported.split,
                "basename": exported.basename,
                "member_name": exported.member_name,
                "target_label": rel_label.as_posix(),
                "status": "changed",
            }
        )

    write_lines(output_root / "train.txt", train_lines)
    write_lines(output_root / "val.txt", val_lines)
    write_report(output_root / "changed_report.csv", report_rows)

    print(
        "Done. "
        f"changed={changed} "
        f"unchanged={unchanged} "
        f"unmatched={unmatched} "
        f"train_changed={len(train_lines)} "
        f"val_changed={len(val_lines)} "
        f"mode={'first_n' if args.first_n_by_export_order > 0 else 'diff'} "
        f"skip_first_n={args.skip_first_n_by_export_order}"
    )
    print(f"Wrote changed-only subset to: {output_root}")
    print(f"Wrote report to: {output_root / 'changed_report.csv'}")


if __name__ == "__main__":
    main()
