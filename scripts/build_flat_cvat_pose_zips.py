#!/usr/bin/env python3
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build flat-basename CVAT helper zips for pose datasets when task frame paths were flattened."
    )
    parser.add_argument("--dataset-root", required=True, help="Root directory of the dataset.")
    parser.add_argument("--split", required=True, choices=["train", "val"], help="Dataset split to export.")
    parser.add_argument("--images-zip", required=True, help="Output zip for task data images only.")
    parser.add_argument(
        "--annotations-zip",
        required=True,
        help="Output zip for Ultralytics YOLO Pose annotation import.",
    )
    parser.add_argument(
        "--kpt-count",
        type=int,
        required=True,
        help="Number of keypoints to write into data.yaml kpt_shape.",
    )
    return parser.parse_args()


def collect_image_paths(dataset_root: Path, split: str) -> list[Path]:
    image_dir = dataset_root / "images" / split
    return sorted(path for path in image_dir.rglob("*.jpg") if path.is_file())


def ensure_unique_basenames(paths: list[Path]) -> None:
    seen: dict[str, Path] = {}
    for path in paths:
        basename = path.name
        prev = seen.get(basename)
        if prev is not None:
            raise ValueError(
                f"Duplicate basename detected: {basename}\n"
                f" - {prev}\n"
                f" - {path}"
            )
        seen[basename] = path


def build_data_yaml(split: str, kpt_count: int) -> str:
    return "\n".join(
        [
            "path: ./",
            f"{split}: {split}.txt",
            "",
            f"kpt_shape: [{kpt_count}, 3]",
            "",
            "names:",
            "  0: dog",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    images_zip = Path(args.images_zip).resolve()
    annotations_zip = Path(args.annotations_zip).resolve()

    image_paths = collect_image_paths(dataset_root, args.split)
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {dataset_root / 'images' / args.split}")
    ensure_unique_basenames(image_paths)

    images_zip.parent.mkdir(parents=True, exist_ok=True)
    annotations_zip.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(images_zip, "w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
        for image_path in image_paths:
            zf.write(image_path, Path("images") / args.split / image_path.name)

    with ZipFile(annotations_zip, "w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("data.yaml", build_data_yaml(args.split, args.kpt_count))
        zf.writestr(
            f"{args.split}.txt",
            "\n".join(f"images/{args.split}/{path.name}" for path in image_paths) + "\n",
        )

        for image_path in image_paths:
            basename = image_path.name
            rel_label = image_path.relative_to(dataset_root / "images")
            label_path = (dataset_root / "labels" / rel_label).with_suffix(".txt")
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for image: {image_path}")
            zf.write(image_path, Path("images") / args.split / basename)
            zf.write(label_path, Path("labels") / args.split / Path(basename).with_suffix(".txt"))

    print(f"Wrote images zip: {images_zip}")
    print(f"Wrote annotations zip: {annotations_zip}")


if __name__ == "__main__":
    main()
