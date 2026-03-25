#!/usr/bin/env python3
"""
Run a trained healthcare-side pose model over a YOLO image dataset and write pseudo labels.

Typical use:
- take the remaining unlabeled subset after the first manual round
- run a checkpoint such as models/for151.pt
- copy images and write YOLO pose labels in the same 23-keypoint schema
- generate a manual_label_queue.csv so the next CVAT correction round can focus on missed points
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from ultralytics import YOLO

from convert_pose_to_healthcare_side import (
    TARGET_KPT_NAMES,
    format_yolo_pose_line,
    parse_yolo_pose_line,
    write_target_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-label a healthcare-side image dataset with a trained YOLO pose model")
    parser.add_argument("--input-root", required=True, help="Input dataset root with train.txt/val.txt and images/")
    parser.add_argument("--output-root", required=True, help="Output dataset root for pseudo labels")
    parser.add_argument("--model-path", required=True, help="Trained YOLO pose checkpoint")
    parser.add_argument("--device", default="0", help="CUDA device id or cpu")
    parser.add_argument("--imgsz", type=int, default=512, help="Inference image size")
    parser.add_argument("--batch", type=int, default=16, help="Inference batch size")
    parser.add_argument("--conf", type=float, default=0.05, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument(
        "--kpt-conf-threshold",
        type=float,
        default=0.2,
        help="Per-keypoint confidence threshold used to mark visibility",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to process",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into the output dataset. Recommended for the next CVAT round.",
    )
    parser.add_argument(
        "--preserve-existing-labels",
        action="store_true",
        help="Keep existing keypoints from input labels and only fill currently missing points from model predictions.",
    )
    parser.add_argument(
        "--fill-point-names",
        nargs="+",
        default=None,
        help=(
            "Optional whitelist of target keypoint names that may be filled from model predictions "
            "when --preserve-existing-labels is enabled. Other missing keypoints remain empty."
        ),
    )
    return parser.parse_args()


def read_index_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_image_path(input_root: Path, line: str) -> Tuple[Path, Path]:
    raw = Path(line)
    abs_path = raw if raw.is_absolute() else (input_root / raw)
    if not abs_path.exists():
        raise FileNotFoundError(f"Image not found: {abs_path}")

    if raw.is_absolute():
        try:
            rel_path = abs_path.relative_to(input_root)
        except ValueError:
            rel_path = Path("images") / abs_path.name
    else:
        rel_path = raw
    return abs_path.resolve(), rel_path


def infer_view(rel_image: Path) -> str:
    lowered_parts = [part.lower() for part in rel_image.parts]
    if "left" in lowered_parts:
        return "left"
    if "right" in lowered_parts:
        return "right"
    return ""


def label_rel_from_image_rel(rel_image: Path, split: str) -> Path:
    parts = rel_image.parts
    if len(parts) >= 2 and parts[0] == "images" and parts[1] == split:
        return Path(*parts[2:]).with_suffix(".txt")
    if len(parts) >= 1 and parts[0] == split:
        return Path(*parts[1:]).with_suffix(".txt")
    return Path(rel_image.name).with_suffix(".txt")


def build_prefix_from_box(cls_id: int, xywhn: Sequence[float]) -> List[str]:
    x, y, w, h = xywhn
    return [str(int(cls_id)), f"{x:.6f}", f"{y:.6f}", f"{w:.6f}", f"{h:.6f}"]


def result_to_keypoints(result, kpt_conf_threshold: float) -> Tuple[List[str] | None, List[Tuple[float, float, int]], float]:
    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return None, [(0.0, 0.0, 0) for _ in TARGET_KPT_NAMES], 0.0

    box_conf = result.boxes.conf
    if box_conf is None or len(box_conf) == 0:
        return None, [(0.0, 0.0, 0) for _ in TARGET_KPT_NAMES], 0.0

    best_idx = int(box_conf.argmax().item())
    score = float(box_conf[best_idx].item())
    cls_id = int(result.boxes.cls[best_idx].item()) if result.boxes.cls is not None else 0
    xywhn = result.boxes.xywhn[best_idx].tolist()
    prefix = build_prefix_from_box(cls_id, xywhn)

    xyn = result.keypoints.xyn
    if xyn is None or len(xyn) <= best_idx:
        return prefix, [(0.0, 0.0, 0) for _ in TARGET_KPT_NAMES], score

    conf = result.keypoints.conf
    conf_row = conf[best_idx].tolist() if conf is not None and len(conf) > best_idx else [1.0] * len(TARGET_KPT_NAMES)
    xy_row = xyn[best_idx].tolist()

    keypoints: List[Tuple[float, float, int]] = []
    for idx, (xy, kp_conf) in enumerate(zip(xy_row, conf_row)):
        x = float(xy[0])
        y = float(xy[1])
        visible = 2 if float(kp_conf) >= kpt_conf_threshold and x > 0 and y > 0 else 0
        if visible == 0:
            keypoints.append((0.0, 0.0, 0))
        else:
            keypoints.append((x, y, visible))

    if len(keypoints) < len(TARGET_KPT_NAMES):
        keypoints.extend([(0.0, 0.0, 0)] * (len(TARGET_KPT_NAMES) - len(keypoints)))
    elif len(keypoints) > len(TARGET_KPT_NAMES):
        keypoints = keypoints[: len(TARGET_KPT_NAMES)]

    return prefix, keypoints, score


def missing_points_from_keypoints(keypoints: Sequence[Tuple[float, float, int]]) -> str:
    return ",".join(name for name, (_, _, v) in zip(TARGET_KPT_NAMES, keypoints) if v <= 0)


def load_existing_label(input_root: Path, split: str, rel_label: Path) -> Tuple[List[str] | None, List[Tuple[float, float, int]] | None]:
    label_path = input_root / "labels" / split / rel_label
    if not label_path.exists():
        return None, None
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None, None
    parsed = parse_yolo_pose_line(lines[0], len(TARGET_KPT_NAMES))
    if parsed is None:
        return None, None
    return parsed


def merge_keypoints(
    existing_prefix: List[str] | None,
    existing_keypoints: List[Tuple[float, float, int]] | None,
    predicted_prefix: List[str] | None,
    predicted_keypoints: List[Tuple[float, float, int]],
    fill_point_indices: set[int] | None = None,
) -> Tuple[List[str] | None, List[Tuple[float, float, int]]]:
    if existing_keypoints is None:
        return predicted_prefix, predicted_keypoints
    if predicted_prefix is None:
        return existing_prefix, existing_keypoints

    merged = list(existing_keypoints)
    for idx, pred_kpt in enumerate(predicted_keypoints):
        if idx >= len(merged):
            break
        if fill_point_indices is not None and idx not in fill_point_indices:
            continue
        ex_x, ex_y, ex_v = merged[idx]
        pred_x, pred_y, pred_v = pred_kpt
        if ex_v <= 0 and pred_v > 0:
            merged[idx] = (pred_x, pred_y, pred_v)

    return existing_prefix or predicted_prefix, merged


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_lines(path: Path, lines: Sequence[str]) -> None:
    ensure_parent(path)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_manual_queue(output_root: Path, rows: Sequence[Dict[str, str]]) -> None:
    fieldnames = ["split", "label", "image", "view", "missing_points", "score"]
    queue_path = output_root / "manual_label_queue.csv"
    with open(queue_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    write_target_yaml(output_root)
    model = YOLO(args.model_path)
    fill_point_indices = None
    if args.fill_point_names is not None:
        name_to_idx = {name: idx for idx, name in enumerate(TARGET_KPT_NAMES)}
        unknown = [name for name in args.fill_point_names if name not in name_to_idx]
        if unknown:
            raise ValueError(f"Unknown fill point names: {unknown}")
        fill_point_indices = {name_to_idx[name] for name in args.fill_point_names}

    all_queue_rows: List[Dict[str, str]] = []
    split_index_lines: Dict[str, List[str]] = {split: [] for split in args.splits}

    for split in args.splits:
        index_lines = read_index_lines(input_root / f"{split}.txt")
        if not index_lines:
            continue

        resolved: List[Tuple[Path, Path]] = [resolve_image_path(input_root, line) for line in index_lines]
        abs_paths = [str(abs_path) for abs_path, _ in resolved]
        rel_paths = [rel_path for _, rel_path in resolved]

        results = model.predict(
            source=abs_paths,
            device=args.device,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            max_det=1,
            verbose=False,
            stream=False,
        )

        if len(results) != len(rel_paths):
            raise RuntimeError(f"Prediction count mismatch: results={len(results)} images={len(rel_paths)}")

        for result, rel_image in zip(results, rel_paths):
            prefix, keypoints, score = result_to_keypoints(result, args.kpt_conf_threshold)
            rel_image_out = rel_image
            rel_label = label_rel_from_image_rel(rel_image, split)
            if args.preserve_existing_labels:
                existing_prefix, existing_keypoints = load_existing_label(input_root, split, rel_label)
                prefix, keypoints = merge_keypoints(
                    existing_prefix,
                    existing_keypoints,
                    prefix,
                    keypoints,
                    fill_point_indices=fill_point_indices,
                )

            dst_label = output_root / "labels" / split / rel_label
            ensure_parent(dst_label)
            if prefix is None:
                dst_label.write_text("", encoding="utf-8")
            else:
                dst_label.write_text(format_yolo_pose_line(prefix, keypoints) + "\n", encoding="utf-8")

            if args.copy_images:
                dst_image = output_root / rel_image_out
                ensure_parent(dst_image)
                shutil.copy2(input_root / rel_image if not rel_image.is_absolute() else rel_image, dst_image)
                split_index_lines[split].append(rel_image_out.as_posix())
            else:
                split_index_lines[split].append(str((input_root / rel_image).resolve()))

            all_queue_rows.append(
                {
                    "split": split,
                    "label": rel_label.as_posix(),
                    "image": rel_image_out.as_posix(),
                    "view": infer_view(rel_image),
                    "missing_points": missing_points_from_keypoints(keypoints),
                    "score": f"{score:.6f}",
                }
            )

    for split, lines in split_index_lines.items():
        write_lines(output_root / f"{split}.txt", lines)
    write_manual_queue(output_root, all_queue_rows)

    total_images = sum(len(lines) for lines in split_index_lines.values())
    print(f"Done. total_images={total_images} output={output_root}")
    print(f"Wrote queue to: {output_root / 'manual_label_queue.csv'}")


if __name__ == "__main__":
    main()
