#!/usr/bin/env python3
"""
Build a lightweight train/val split view over an existing YOLO pose dataset.

This script does not copy images or labels.
It writes new train.txt / val.txt files that point to the original image paths.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a train/val split view over an existing YOLO pose dataset")
    parser.add_argument("--input-root", required=True, help="Existing dataset root that already has train.txt/data.yaml")
    parser.add_argument("--output-root", required=True, help="Output split-view dataset root")
    parser.add_argument("--val-count", type=int, default=30, help="How many samples to place into val")
    return parser.parse_args()


def read_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_evenly_spaced_indices(n_items: int, n_select: int) -> List[int]:
    if n_select <= 0 or n_items <= 0:
        return []
    if n_select >= n_items:
        return list(range(n_items))
    if n_select == 1:
        return [n_items // 2]

    step = (n_items - 1) / (n_select - 1)
    selected = []
    for i in range(n_select):
        idx = round(i * step)
        if selected and idx <= selected[-1]:
            idx = selected[-1] + 1
        if idx >= n_items:
            idx = n_items - 1
        selected.append(idx)
    return selected


def write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def rewrite_yaml_path(yaml_text: str, dataset_root: Path) -> str:
    replacement = f"path: {dataset_root}"
    if re.search(r"^path:\s*.*$", yaml_text, flags=re.MULTILINE):
        return re.sub(r"^path:\s*.*$", replacement, yaml_text, count=1, flags=re.MULTILINE)
    return replacement + "\n" + yaml_text


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train_lines = read_lines(input_root / "train.txt")
    val_lines_existing = read_lines(input_root / "val.txt")
    all_lines = train_lines + val_lines_existing
    if not all_lines:
        raise ValueError(f"No samples found in {input_root}")

    val_indices = set(select_evenly_spaced_indices(len(all_lines), args.val_count))
    split_train = [line for idx, line in enumerate(all_lines) if idx not in val_indices]
    split_val = [line for idx, line in enumerate(all_lines) if idx in val_indices]

    data_yaml = output_root / "data.yaml"
    healthcare_yaml = output_root / "healthcare-side-final-23.yaml"

    if (input_root / "data.yaml").exists():
        source_text = (input_root / "data.yaml").read_text(encoding="utf-8")
        data_yaml.write_text(rewrite_yaml_path(source_text, output_root), encoding="utf-8")
    if (input_root / "healthcare-side-final-23.yaml").exists():
        source_text = (input_root / "healthcare-side-final-23.yaml").read_text(encoding="utf-8")
        healthcare_yaml.write_text(rewrite_yaml_path(source_text, output_root), encoding="utf-8")

    # Rewrite split files to absolute paths so the lightweight view can live anywhere.
    split_train_abs = [str((input_root / line).resolve()) if not Path(line).is_absolute() else line for line in split_train]
    split_val_abs = [str((input_root / line).resolve()) if not Path(line).is_absolute() else line for line in split_val]

    write_lines(output_root / "train.txt", split_train_abs)
    write_lines(output_root / "val.txt", split_val_abs)

    print(
        "Done. "
        f"total={len(all_lines)} "
        f"train={len(split_train_abs)} "
        f"val={len(split_val_abs)} "
        f"output={output_root}"
    )


if __name__ == "__main__":
    main()
