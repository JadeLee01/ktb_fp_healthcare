#!/usr/bin/env python3
"""
Apply a flat CVAT Ultralytics YOLO Pose export back onto the nested manual pilot dataset.

Typical use:
  python3 scripts/apply_cvat_export_to_manual_pilot.py \
    --pilot-root dataset_healthcare_side_manual_pilot_v1 \
    --export-path /path/to/cvat_export.zip

The script:
  - reads exported labels from labels/{train,val}/*.txt
  - maps each flat basename back to the original nested label path in the pilot dataset
  - backs up the original pilot labels before overwriting
  - prints updated file counts and keypoint visibility stats for the updated subset
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


KEYPOINT_NAMES: List[str] = [
    "nose",
    "withers",
    "t13_spinous_process",
    "iliac_crest",
    "sacrum",
    "tail_base",
    "tail_end",
    "near_front_shoulder",
    "near_front_elbow",
    "near_front_carpus",
    "near_front_paw",
    "far_front_shoulder",
    "far_front_elbow",
    "far_front_carpus",
    "far_front_paw",
    "near_rear_hip",
    "near_rear_stifle",
    "near_rear_hock",
    "near_rear_paw",
    "far_rear_hip",
    "far_rear_stifle",
    "far_rear_hock",
    "far_rear_paw",
]


@dataclass(frozen=True)
class ExportedLabel:
    split: str
    basename: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a flat CVAT export back to the nested manual pilot dataset")
    parser.add_argument("--pilot-root", required=True, help="Nested pilot dataset root (images/, labels/, manual_label_queue.csv)")
    parser.add_argument("--export-path", required=True, help="CVAT export zip or directory")
    parser.add_argument(
        "--backup-root",
        help="Optional backup directory. Defaults to <pilot-root>/_backups/cvat_import_<timestamp>",
    )
    parser.add_argument(
        "--report-csv",
        help="Optional CSV report path. Defaults to <pilot-root>/cvat_import_report_<timestamp>.csv",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be updated")
    return parser.parse_args()


def iter_exported_labels(export_path: Path) -> Iterator[ExportedLabel]:
    if export_path.is_dir():
        for split in ("train", "val"):
            label_dir = export_path / "labels" / split
            if not label_dir.exists():
                continue
            for path in sorted(label_dir.rglob("*.txt")):
                yield ExportedLabel(split=split, basename=path.name, text=path.read_text(encoding="utf-8"))
        return

    with zipfile.ZipFile(export_path) as zf:
        for name in sorted(zf.namelist()):
            path = Path(name)
            if len(path.parts) < 3:
                continue
            if path.parts[0] != "labels":
                continue
            split = path.parts[1]
            if split not in {"train", "val"} or path.suffix != ".txt":
                continue
            with zf.open(name, "r") as f:
                text = io.TextIOWrapper(f, encoding="utf-8").read()
            yield ExportedLabel(split=split, basename=path.name, text=text)


def build_pilot_label_map(pilot_root: Path) -> Dict[Tuple[str, str], Path]:
    mapping: Dict[Tuple[str, str], Path] = {}
    for split in ("train", "val"):
        label_root = pilot_root / "labels" / split
        if not label_root.exists():
            continue
        for path in sorted(label_root.rglob("*.txt")):
            key = (split, path.name)
            if key in mapping:
                raise ValueError(f"Duplicate basename in pilot dataset for {split}: {path.name}")
            mapping[key] = path
    return mapping


def default_backup_root(pilot_root: Path) -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return pilot_root / "_backups" / f"cvat_import_{stamp}"


def default_report_csv(pilot_root: Path) -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return pilot_root / f"cvat_import_report_{stamp}.csv"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def backup_file(src: Path, backup_root: Path, pilot_root: Path) -> Path:
    rel = src.relative_to(pilot_root)
    dst = backup_root / rel
    ensure_parent(dst)
    shutil.copy2(src, dst)
    return dst


def parse_visible_keypoints(label_text: str, expected_kpt: int = 23) -> List[bool]:
    line = ""
    for raw in label_text.splitlines():
        raw = raw.strip()
        if raw:
            line = raw
            break
    if not line:
        return [False] * expected_kpt

    values = line.split()
    min_required = 5 + expected_kpt * 3
    if len(values) < min_required:
        raise ValueError(f"Unexpected label length: got {len(values)}, expected at least {min_required}")

    offset = 5
    visible: List[bool] = []
    for idx in range(expected_kpt):
        x = float(values[offset + idx * 3 + 0])
        y = float(values[offset + idx * 3 + 1])
        v = int(float(values[offset + idx * 3 + 2]))
        visible.append(v > 0 and not (x == 0.0 and y == 0.0))
    return visible


def summarize_visibility(label_texts: Sequence[str]) -> List[Tuple[str, int, float]]:
    total = len(label_texts)
    counts = Counter()
    for text in label_texts:
        visible = parse_visible_keypoints(text, expected_kpt=len(KEYPOINT_NAMES))
        for idx, is_visible in enumerate(visible):
            if is_visible:
                counts[idx] += 1

    rows: List[Tuple[str, int, float]] = []
    for idx, name in enumerate(KEYPOINT_NAMES):
        count = counts[idx]
        ratio = (count / total * 100.0) if total else 0.0
        rows.append((name, count, ratio))
    return rows


def write_report_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "basename", "target_label", "backup_label", "status"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    pilot_root = Path(args.pilot_root).resolve()
    export_path = Path(args.export_path).resolve()
    backup_root = Path(args.backup_root).resolve() if args.backup_root else default_backup_root(pilot_root)
    report_csv = Path(args.report_csv).resolve() if args.report_csv else default_report_csv(pilot_root)

    if not pilot_root.exists():
        raise FileNotFoundError(f"Pilot root not found: {pilot_root}")
    if not export_path.exists():
        raise FileNotFoundError(f"Export path not found: {export_path}")

    label_map = build_pilot_label_map(pilot_root)
    updated_texts: List[str] = []
    report_rows: List[Dict[str, str]] = []
    updated = 0
    unmatched = 0

    for exported in iter_exported_labels(export_path):
        key = (exported.split, exported.basename)
        target = label_map.get(key)
        if target is None:
            unmatched += 1
            report_rows.append(
                {
                    "split": exported.split,
                    "basename": exported.basename,
                    "target_label": "",
                    "backup_label": "",
                    "status": "unmatched",
                }
            )
            continue

        backup_path = backup_root / target.relative_to(pilot_root)
        status = "would_update" if args.dry_run else "updated"
        if not args.dry_run:
            backup_file(target, backup_root, pilot_root)
            target.write_text(exported.text.rstrip() + "\n", encoding="utf-8")
        updated += 1
        updated_texts.append(exported.text)
        report_rows.append(
            {
                "split": exported.split,
                "basename": exported.basename,
                "target_label": str(target.relative_to(pilot_root)),
                "backup_label": str(backup_path.relative_to(pilot_root)) if not args.dry_run else "",
                "status": status,
            }
        )

    write_report_csv(report_csv, report_rows)

    print(
        f"{'Dry run complete' if args.dry_run else 'Import complete'}. "
        f"updated={updated} unmatched={unmatched} report={report_csv}"
    )
    if not args.dry_run:
        print(f"Backups written under: {backup_root}")

    if updated_texts:
        print("\nUpdated subset keypoint visibility:")
        for name, count, ratio in summarize_visibility(updated_texts):
            print(f"  {name:20s} {count:4d}/{len(updated_texts):4d} ({ratio:5.1f}%)")


if __name__ == "__main__":
    main()
