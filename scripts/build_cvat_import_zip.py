#!/usr/bin/env python3
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


DEFAULT_KPT_COUNT = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a CVAT-friendly Ultralytics YOLO Pose zip from an existing side14 dataset root."
    )
    parser.add_argument("--dataset-root", required=True, help="Root directory of the side14 dataset.")
    parser.add_argument("--output-zip", required=True, help="Destination zip path.")
    parser.add_argument(
        "--kpt-count",
        type=int,
        default=DEFAULT_KPT_COUNT,
        help="Number of keypoints to write into data.yaml kpt_shape.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        choices=["train", "val", "test"],
        help="Subset directories to include in the zip.",
    )
    return parser.parse_args()


def list_subset_images(dataset_root: Path, split: str) -> list[str]:
    image_dir = dataset_root / "images" / split
    if not image_dir.exists():
        return []
    return sorted(
        path.relative_to(dataset_root).as_posix()
        for path in image_dir.rglob("*")
        if path.is_file()
    )


def build_data_yaml(splits: list[str], kpt_count: int) -> str:
    lines = [
        "path: ./",
    ]
    for split in splits:
        lines.append(f"{split}: {split}.txt")
        lines.extend(
        [
            "",
            f"kpt_shape: [{kpt_count}, 3]",
            "",
            "names:",
            "  0: dog",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_zip = Path(args.output_zip).resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("data.yaml", build_data_yaml(args.splits, args.kpt_count))

        for split in args.splits:
            subset_images = list_subset_images(dataset_root, split)
            zf.writestr(f"{split}.txt", "\n".join(subset_images) + ("\n" if subset_images else ""))

            for rel_path in subset_images:
                abs_path = dataset_root / rel_path
                zf.write(abs_path, rel_path)

                label_rel_path = Path(rel_path.replace("images/", "labels/", 1)).with_suffix(".txt")
                label_abs_path = dataset_root / label_rel_path
                if not label_abs_path.exists():
                    raise FileNotFoundError(f"Missing label for image: {label_abs_path}")
                zf.write(label_abs_path, label_rel_path.as_posix())

    print(f"Wrote CVAT import zip: {output_zip}")


if __name__ == "__main__":
    main()
