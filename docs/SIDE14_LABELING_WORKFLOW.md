# Side14 Labeling Workflow

## Goal
Create a side-view gait dataset with 14 keypoints that match the healthcare service better than the current mixed-view schemas.

## Final Keypoints
1. `withers`
2. `t13_spinous_process`
3. `sacrum`
4. `tail_base`
5. `near_rear_hip`
6. `near_rear_stifle`
7. `near_rear_hock`
8. `near_rear_paw`
9. `far_rear_hip`
10. `far_rear_stifle`
11. `far_rear_hock`
12. `far_rear_paw`
13. `near_front_paw`
14. `far_front_paw`

## Automatic Mapping
Use `scripts/convert_pose_to_side14.py` to bootstrap labels.

### From `ultralytics24`
| side14 target | source point | note |
| --- | --- | --- |
| `withers` | `withers` | copy when present |
| `t13_spinous_process` | none | manual |
| `sacrum` | none | manual |
| `tail_base` | `tail_start` | auto |
| `near_rear_hip` | none | manual |
| `near_rear_stifle` | `rear_left_knee` or `rear_right_knee` | depends on left/right view |
| `near_rear_hock` | `rear_left_elbow` or `rear_right_elbow` | current pipeline uses this as hock proxy |
| `near_rear_paw` | `rear_left_paw` or `rear_right_paw` | depends on view |
| `far_rear_hip` | none | manual |
| `far_rear_stifle` | opposite rear knee | depends on view |
| `far_rear_hock` | opposite rear elbow | depends on view |
| `far_rear_paw` | opposite rear paw | depends on view |
| `near_front_paw` | `front_left_paw` or `front_right_paw` | depends on view |
| `far_front_paw` | opposite front paw | depends on view |

### From `integrated34`
| side14 target | source point |
| --- | --- |
| `withers` | `Withers` |
| `t13_spinous_process` | `T13 Spinous precess` |
| `sacrum` | `Sacrum` |
| `tail_base` | `Tail start` |
| `near_rear_hip` | `L_Femoral greater trochanter` or `R_Femoral greater trochanter` |
| `near_rear_stifle` | `L_Femorotibial joint` or `R_Femorotibial joint` |
| `near_rear_hock` | `L_Lateral malleolus of the distal tibia` or `R_Lateral malleolus of the distal tibia` |
| `near_rear_paw` | `L_Distal lateral aspect of the fifth metatarsus` or `R_Distal lateral aspect of the fifth metatarsus` |
| `far_rear_hip` | opposite femoral greater trochanter |
| `far_rear_stifle` | opposite femorotibial joint |
| `far_rear_hock` | opposite lateral malleolus |
| `far_rear_paw` | opposite distal fifth metatarsus |
| `near_front_paw` | `L_Distal lateral aspect of fifth metacarpal bone` or `R_Distal lateral aspect of fifth metacarpal bone` |
| `far_front_paw` | opposite distal fifth metacarpal bone |

## View Rule
`near` and `far` are based on camera side, not dog anatomy labels.

- If the visible side is the dog's left side, then `near = left`, `far = right`
- If the visible side is the dog's right side, then `near = right`, `far = left`

The conversion script needs this left/right side information from one of these sources:
- a CSV manifest
- parent folder names such as `Left/` and `Right/`
- file names containing `Left` or `Right`
- a fixed value if the whole split contains only one side

## Manual Points
If the source is `ultralytics24`, these points still need manual work in most images:
- `t13_spinous_process`
- `sacrum`
- `near_rear_hip`
- `far_rear_hip`
- `withers` only when the original `withers` is missing

If the source is `integrated34`, some of these can be copied automatically, but you still need to review them because many samples contain `0 0 0`.

## Labeling Tool Choice
You do not need Roboflow specifically.

Recommended default: `CVAT`
- good for local use
- supports keypoints
- avoids cloud upload for large datasets
- practical when you want to correct auto-generated labels

Roboflow is still acceptable when:
- you want a quicker UI setup
- your team already uses it
- cloud upload is acceptable

Editing YOLO pose `.txt` files directly is not practical for this task.

## Practical Workflow
1. Filter side-view walking images only.
2. Prepare left/right side metadata.
3. Run the conversion script to create `side14` labels with automatic points filled.
4. Import the generated dataset into a labeling tool.
5. Correct `withers` when missing.
6. Add `t13_spinous_process`, `sacrum`, `near_rear_hip`, and `far_rear_hip`.
7. Export the corrected YOLO pose labels.
8. Train the pose model on the new `side14` dataset.

## Example Commands
Using a manifest:

```bash
python scripts/convert_pose_to_side14.py \
  --input-root /path/to/source_dataset \
  --output-root /path/to/dataset_side14 \
  --source-schema ultralytics24 \
  --splits train val \
  --view-source manifest \
  --manifest /path/to/side_views.csv \
  --copy-images
```

Using folder names:

```bash
python scripts/convert_pose_to_side14.py \
  --input-root /path/to/source_dataset \
  --output-root /path/to/dataset_side14 \
  --source-schema integrated34 \
  --splits train val \
  --view-source parent \
  --copy-images
```

## Manifest Format
CSV columns:

```text
path,view,include
images/train/dog_0001,left,1
images/train/dog_0002,right,1
images/train/dog_0003,left,0
```

- `path`: relative path or stem
- `view`: `left` or `right`
- `include`: optional, use `0` to skip a file

## Landmark Rules
Use the same anatomical rule for every frame.

- `withers`: highest dorsal point at the shoulder region
- `t13_spinous_process`: thoracolumbar transition point on the dorsal midline
- `sacrum`: dorsal pelvis midpoint just before the tail base
- `tail_base`: the proximal tail root
- `rear_hip`: greater trochanter region, not the iliac crest

When the far-side point is fully occluded, leave it missing instead of guessing.
