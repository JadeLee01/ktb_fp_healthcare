#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


TARGET_KPT_COUNT = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build smaller CVAT-friendly Ultralytics YOLO Pose zip chunks from an existing side14 dataset."
    )
    parser.add_argument("--dataset-root", required=True, help="Root directory of the side14 dataset.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Subset to chunk.")
    parser.add_argument("--chunk-size", type=int, default=500, help="Images per chunk zip.")
    parser.add_argument("--output-dir", required=True, help="Directory to write chunk zips into.")
    parser.add_argument(
        "--queue-csv",
        default="",
        help="Optional manual queue CSV for the same split. If set, a per-chunk queue CSV is written next to each zip.",
    )
    return parser.parse_args()


def list_subset_images(dataset_root: Path, split: str) -> list[str]:
    image_dir = dataset_root / "images" / split
    return sorted(
        path.relative_to(dataset_root).as_posix()
        for path in image_dir.rglob("*")
        if path.is_file()
    )


def build_data_yaml() -> str:
    return "\n".join(
        [
            "path: ./",
            "train: train.txt",
            "",
            f"kpt_shape: [{TARGET_KPT_COUNT}, 3]",
            "",
            "names:",
            "  0: dog",
            "",
        ]
    )


def chunk_rows(rows: list[str], chunk_size: int) -> list[list[str]]:
    return [rows[idx : idx + chunk_size] for idx in range(0, len(rows), chunk_size)]


def load_queue_rows(queue_csv: Path) -> list[dict[str, str]]:
    if not queue_csv.exists():
        return []
    with open(queue_csv, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_chunk_queue_csv(queue_rows: list[dict[str, str]], image_set: set[str], dst_csv: Path) -> None:
    filtered_rows = [row for row in queue_rows if row.get("image", "") in image_set]
    fieldnames = ["split", "label", "image", "view", "missing_points"]
    with open(dst_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in filtered_rows:
            writer.writerow(row)


def write_chunk_zip(dataset_root: Path, chunk_images: list[str], output_zip: Path) -> None:
    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("data.yaml", build_data_yaml())
        zf.writestr("train.txt", "\n".join(chunk_images) + "\n")

        for rel_image_path in chunk_images:
            image_abs_path = dataset_root / rel_image_path
            if not image_abs_path.exists():
                raise FileNotFoundError(f"Missing image: {image_abs_path}")
            zf.write(image_abs_path, rel_image_path)

            label_rel_path = Path(rel_image_path.replace("images/", "labels/", 1)).with_suffix(".txt")
            label_abs_path = dataset_root / label_rel_path
            if not label_abs_path.exists():
                raise FileNotFoundError(f"Missing label: {label_abs_path}")
            zf.write(label_abs_path, label_rel_path.as_posix())


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_subset_images(dataset_root, args.split)
    if not images:
        raise FileNotFoundError(f"No images found for split '{args.split}' in {dataset_root}")

    queue_rows = load_queue_rows(Path(args.queue_csv).resolve()) if args.queue_csv else []
    chunks = chunk_rows(images, args.chunk_size)
    total_chunks = len(chunks)
    width = max(2, len(str(total_chunks)))

    for index, chunk_images in enumerate(chunks, start=1):
        stem = f"{args.split}_chunk_{index:0{width}d}_of_{total_chunks:0{width}d}"
        output_zip = output_dir / f"{stem}.zip"
        write_chunk_zip(dataset_root, chunk_images, output_zip)

        if queue_rows:
            queue_csv_path = output_dir / f"{stem}_manual_queue.csv"
            write_chunk_queue_csv(queue_rows, set(chunk_images), queue_csv_path)

        print(f"[{index}/{total_chunks}] wrote {output_zip}")

    print(
        "Done. "
        f"split={args.split} images={len(images)} chunk_size={args.chunk_size} chunks={total_chunks} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
