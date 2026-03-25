#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2


DEFAULT_DATASET_ROOT = Path(
    "/root/medical_AI/hkh/jacob/ktb_fp_healthcare/60.반려견_보행영상_기반_건강관리_데이터/3.개방데이터/1.데이터"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render overlay preview images from raw AI-Hub healthcare dataset JSON labels."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root containing Training/Validation folders.",
    )
    parser.add_argument(
        "--phase",
        default="Training",
        choices=["Training", "Validation"],
        help="Dataset phase to inspect.",
    )
    parser.add_argument(
        "--view",
        default="Back",
        choices=["Front", "Back", "Left", "Right"],
        help="Camera view folder to render.",
    )
    parser.add_argument(
        "--basename",
        help="Optional exact file stem to render, e.g. SNC_2024_10_22_14_42_14_00025",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of images to render when --basename is not provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/medical_AI/hkh/jacob/ktb_fp_healthcare/debug_aihub_overlays"),
        help="Directory where overlay jpg files will be written.",
    )
    parser.add_argument(
        "--draw-label-text",
        action="store_true",
        help="Draw label names next to each point.",
    )
    return parser.parse_args()


def color_for_label(label: str) -> tuple[int, int, int]:
    seed = sum(ord(ch) for ch in label)
    return (
        64 + (seed * 37) % 192,
        64 + (seed * 67) % 192,
        64 + (seed * 97) % 192,
    )


def collect_jsons(label_root: Path, view: str, basename: str | None) -> list[Path]:
    if basename:
        return sorted(label_root.rglob(f"{basename}.json"))
    return sorted(path for path in label_root.rglob("*.json") if path.parent.name == view)


def resolve_image_path(image_root: Path, label_path: Path) -> Path:
    rel = label_path.relative_to(label_path.parents[3])
    # label_path.parents[3] = .../<phase>/02.라벨링데이터
    candidate = image_root / rel
    return candidate.with_suffix(".jpg")


def render_overlay(image_path: Path, label_path: Path, output_path: Path, draw_label_text: bool) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        return False

    data = json.loads(label_path.read_text())
    annotations = data.get("annotation_info", [])
    h, w = image.shape[:2]

    counts: Counter[str] = Counter()
    for ann in annotations:
        label = ann.get("label", "").strip()
        if not label:
            continue
        try:
            x = float(ann["x"]) * w
            y = float(ann["y"]) * h
        except (KeyError, ValueError, TypeError):
            continue

        counts[label] += 1
        label_name = label if counts[label] == 1 else f"{label}#{counts[label]}"
        color = color_for_label(label)
        center = (int(round(x)), int(round(y)))

        cv2.circle(image, center, 6, color, thickness=-1)
        cv2.circle(image, center, 9, (255, 255, 255), thickness=2)

        if draw_label_text:
            cv2.putText(
                image,
                label_name,
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
                cv2.LINE_AA,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return True


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    image_root = dataset_root / args.phase / "01.원천데이터"
    label_root = dataset_root / args.phase / "02.라벨링데이터"

    json_paths = collect_jsons(label_root, args.view, args.basename)
    if args.basename is None:
        json_paths = json_paths[: args.limit]

    if not json_paths:
        raise FileNotFoundError("No matching JSON labels found for the requested view/basename.")

    written = 0
    skipped_existing = 0
    skipped_unreadable = 0
    for label_path in json_paths:
        if label_path.parent.name != args.view:
            continue
        rel = label_path.relative_to(label_root)
        image_path = (image_root / rel).with_suffix(".jpg")
        if not image_path.exists():
            raise FileNotFoundError(f"Missing source image for label: {label_path}")

        out_name = "__".join(rel.with_suffix("").parts) + ".jpg"
        output_path = args.output_dir / args.phase / args.view / out_name
        if output_path.exists():
            skipped_existing += 1
            continue

        ok = render_overlay(image_path, label_path, output_path, draw_label_text=args.draw_label_text)
        if ok:
            written += 1
        else:
            skipped_unreadable += 1

    print(f"Wrote {written} overlay image(s) to: {args.output_dir / args.phase / args.view}")
    if skipped_existing:
        print(f"Skipped existing overlay image(s): {skipped_existing}")
    if skipped_unreadable:
        print(f"Skipped unreadable source image(s): {skipped_unreadable}")


if __name__ == "__main__":
    main()
