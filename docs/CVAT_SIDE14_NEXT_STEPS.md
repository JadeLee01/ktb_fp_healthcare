# CVAT Side14 Next Steps

## What Is Ready
- Side14 schema file: `/root/medical_AI/hkh/jacob/ktb_fp_healthcare/side14-pose.yaml`
- Conversion script: `/root/medical_AI/hkh/jacob/ktb_fp_healthcare/scripts/convert_pose_to_side14.py`
- CVAT seed dataset: `/root/medical_AI/hkh/jacob/ktb_fp_healthcare/dataset_side14_cvat_seed`
- Legacy full zip: `/root/medical_AI/hkh/jacob/ktb_fp_healthcare/dataset_side14_cvat_seed.zip`
- Legacy train-only zip: `/root/medical_AI/hkh/jacob/ktb_fp_healthcare/dataset_side14_cvat_seed_train.zip`
- Preferred clean full zip for CVAT import: `/root/medical_AI/hkh/jacob/ktb_fp_healthcare/dataset_side14_cvat_seed_cvat_import.zip`
- Preferred clean train-only zip for CVAT import: `/root/medical_AI/hkh/jacob/ktb_fp_healthcare/dataset_side14_cvat_seed_train_cvat_import.zip`
- Manual queue CSV: `/root/medical_AI/hkh/jacob/ktb_fp_healthcare/dataset_side14_cvat_seed/manual_label_queue.csv`

Current seed dataset size:
- `train`: 3346 images
- `val`: 2567 images
- total: 5913 images
- `dataset_side14_cvat_seed.zip`: 2.44 GiB
- `dataset_side14_cvat_seed_train.zip`: 1.38 GiB

The seed dataset was built from the existing `integrated34` labels with:
- only `Left` and `Right` views
- `2` evenly spaced frames per clip
- automatic mapping into the side14 schema

## Why CVAT Is Fine
The local CVAT docs in this workspace show that `Ultralytics YOLO Pose 1.0` is supported for import.

Relevant local references:
- `/root/medical_AI/hkh/jacob/cvat/site/content/en/docs/dataset_management/formats/_index.md`
- `/root/medical_AI/hkh/jacob/cvat/site/content/en/docs/dataset_management/formats/format-yolo-ultralytics.md`

## What You Should Label
Focus on correcting or adding these points first:
- `withers`
- `t13_spinous_process`
- `sacrum`
- `near_rear_hip`
- `far_rear_hip`

The file `manual_label_queue.csv` already lists which images are missing which points.

## Recommended CVAT Workflow
1. Start with the `train` split only.
2. Create one task from `/root/medical_AI/hkh/jacob/ktb_fp_healthcare/dataset_side14_cvat_seed/images/train`.
3. Define the `dog` skeleton label manually before importing annotations.
4. Import annotations in `Ultralytics YOLO Pose 1.0` format from the preferred clean zip.
5. Use `manual_label_queue.csv` as your checklist for missing points.
6. Correct the missing landmarks only. Do not redraw the full skeleton unless the imported points are clearly wrong.
7. Export annotations again as `Ultralytics YOLO Pose 1.0`.
8. Replace or merge the corrected labels back into the seed dataset.
9. Train on the corrected `train` split.
10. Keep `val` for later or annotate it after the train split workflow is stable.

## Suggested Task Size
Do not upload all 3346 train images as one huge correction batch if you are working alone.

Recommended:
- split into chunks of `400` to `800` images
- finish one chunk end-to-end
- export and verify
- then continue with the next chunk

## Verification After Export
After CVAT export, verify:
- keypoint count is exactly `14`
- labels still use the same point order as `side14-pose.yaml`
- missing points remain `0 0 0`
- no file/image name mismatches were introduced

## If Import Does Not Behave As Expected
Use this fallback:
1. Create the CVAT project first and define the `dog` skeleton manually.
2. Create the CVAT task with images only.
3. Import the generated annotations separately as `Ultralytics YOLO Pose 1.0` using the preferred clean zip.
4. If your CVAT build still handles skeleton import differently, keep the generated labels as reference and correct only the missing points in CVAT.

## Practical First Pass
For the first annotation pass, prioritize:
- left-view clips with clearly visible torso
- larger dogs with more stable gait visibility
- frames where rear hip landmarks are least occluded

This will make the first corrected training round much cleaner than trying to fix every difficult frame immediately.
