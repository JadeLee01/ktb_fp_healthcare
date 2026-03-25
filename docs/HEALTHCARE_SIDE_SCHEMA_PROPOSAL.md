# Healthcare Side Schema Proposal

## Goal
Build a single side-view gait keypoint schema for the healthcare service.

This schema should support:
- the 5 healthcare metrics in `healthcare-service`
- user-facing keypoint overlay videos
- mixed-source training from:
  - Ultralytics 24-keypoint dog pose
  - AI-Hub dog gait healthcare data (`Left` / `Right` only)
  - manual correction labels

The important rule is:
- source datasets do not need to match each other
- both source datasets only need to project into one target schema
- unknown points stay `0 0 0`

## View Rule
The target schema should use `near` / `far`, not anatomical `left` / `right`.

- if the dog is seen from the left side: `near = left`, `far = right`
- if the dog is seen from the right side: `near = right`, `far = left`

This keeps the model aligned with the actual service input: side-view gait videos.

## Recommended Final Target Schema
Recommended final schema: `healthcare-side-final-23`

1. `nose`
2. `withers`
3. `t13_spinous_process`
4. `iliac_crest`
5. `sacrum`
6. `tail_base`
7. `tail_end`
8. `near_front_shoulder`
9. `near_front_elbow`
10. `near_front_carpus`
11. `near_front_paw`
12. `far_front_shoulder`
13. `far_front_elbow`
14. `far_front_carpus`
15. `far_front_paw`
16. `near_rear_hip`
17. `near_rear_stifle`
18. `near_rear_hock`
19. `near_rear_paw`
20. `far_rear_hip`
21. `far_rear_stifle`
22. `far_rear_hock`
23. `far_rear_paw`

### Why These 23 Points
- `nose`: better overlay readability, head direction, future head-bob signal
- `withers`, `t13_spinous_process`, `iliac_crest`, `sacrum`, `tail_base`, `tail_end`: trunk axis, pelvic motion, stability
- rear chain (`hip`, `stifle`, `hock`, `paw`): mobility, rhythm, patella-related signals
- front chain (`shoulder`, `elbow`, `carpus`, `paw`): balance, compensation patterns, clearer overlay

## Mapping Status Definitions
- `exact`: same or essentially same anatomical landmark
- `proxy`: usable approximation for seed pretraining, but not ideal as gold supervision
- `missing`: no reliable source point in that dataset subset

## Source Mapping
Below, `AI-Hub Left/Right` means only the side-view subset, not `Front` / `Back`.

Important observation from the raw AI-Hub `Left` / `Right` JSONs:
- they consistently contain:
  - `Acromion/Greater tubercle`
  - `Distal lateral aspect of fifth metacarpal bone`
  - `Distal lateral aspect of the fifth metatarsus`
  - `Dorsal scapular spine`
  - `Ear`
  - `Femoral greater trochanter`
  - `Femorotibial joint`
  - `Iliac crest`
  - `Lateral humeral epicondyle`
  - `Lateral malleolus of the distal tibia`
  - `T13 Spinous precess`
  - `Ulnar styloid process`
- they do not reliably contain:
  - `Nose`
  - `Withers`
  - `Sacrum`
  - `Tail start`
  - `Tail end`

This means AI-Hub side-view data is still useful, but mainly for:
- `t13`
- `iliac_crest`
- near-side limb chains

It is not a complete source for head and tail anchors in the current raw form.

### Head And Trunk

| Target | Ultralytics24 | Status | AI-Hub Left/Right | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| `nose` | `nose` | exact | none | missing | AI-Hub raw side-view labels use `Ear`, not `Nose` |
| `withers` | `withers` | exact | none | missing | AI-Hub raw side-view labels use `Dorsal scapular spine`, not `Withers` |
| `t13_spinous_process` | none | missing | `T13 Spinous precess` | exact | AI-Hub advantage |
| `iliac_crest` | none | missing | `Iliac crest` | exact | AI-Hub advantage |
| `sacrum` | none | missing | none | missing | Not present in observed raw AI-Hub `Left/Right` JSONs |
| `tail_base` | `tail_start` | exact | none | missing | Not present in observed raw AI-Hub `Left/Right` JSONs |
| `tail_end` | `tail_end` | exact | none | missing | Not present in observed raw AI-Hub `Left/Right` JSONs |

### Near Front Leg

| Target | Ultralytics24 | Status | AI-Hub Left/Right | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| `near_front_shoulder` | none | missing | `Acromion/Greater tubercle` | exact | Requires AI-Hub or manual labels |
| `near_front_elbow` | `front_<near>_elbow` | exact | `Lateral humeral epicondyle` | exact | Good cross-dataset match |
| `near_front_carpus` | `front_<near>_knee` | proxy | `Ulnar styloid process` | exact | Ultralytics front `knee` is the distal front joint proxy |
| `near_front_paw` | `front_<near>_paw` | exact | `Distal lateral aspect of fifth metacarpal bone` | exact | Strong overlay/balance point |

### Far Front Leg

| Target | Ultralytics24 | Status | AI-Hub Left/Right | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| `far_front_shoulder` | none | missing | none | missing | Mostly manual if needed |
| `far_front_elbow` | `front_<far>_elbow` | exact | none | missing | U24 helps here |
| `far_front_carpus` | `front_<far>_knee` | proxy | none | missing | U24 proxy only |
| `far_front_paw` | `front_<far>_paw` | exact | none | missing | U24 helps here |

### Near Rear Leg

| Target | Ultralytics24 | Status | AI-Hub Left/Right | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| `near_rear_hip` | none | missing | `Femoral greater trochanter` | exact | Critical for better mobility/patella logic |
| `near_rear_stifle` | `rear_<near>_knee` | exact | `Femorotibial joint` | exact | Strong cross-dataset match |
| `near_rear_hock` | `rear_<near>_elbow` | proxy | `Lateral malleolus of the distal tibia` | exact | Existing pipeline already uses this proxy |
| `near_rear_paw` | `rear_<near>_paw` | exact | `Distal lateral aspect of the fifth metatarsus` | exact | Strong cross-dataset match |

### Far Rear Leg

| Target | Ultralytics24 | Status | AI-Hub Left/Right | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| `far_rear_hip` | none | missing | none | missing | Mostly manual if needed |
| `far_rear_stifle` | `rear_<far>_knee` | exact | none | missing | U24 helps here |
| `far_rear_hock` | `rear_<far>_elbow` | proxy | none | missing | U24 proxy only |
| `far_rear_paw` | `rear_<far>_paw` | exact | none | missing | U24 helps here |

## What This Means In Practice
There is no requirement that every dataset supervises every point.

The practical split is:
- Ultralytics24 contributes:
  - nose
  - withers
  - tail base / tail end
  - both-side paws
  - both-side rear stifles
  - both-side rear hocks as proxy
  - both-side front elbows
  - both-side front carpus as proxy
- AI-Hub Left/Right contributes:
  - t13
  - iliac crest
  - near-side shoulder / elbow / carpus / paw
  - near-side rear hip / stifle / hock / paw
  - dorsal scapular spine and ear exist in raw data, but are not currently part of `healthcare-side-final-23`
- Manual labels are mainly needed for:
  - `nose`
  - `withers`
  - `sacrum`
  - `tail_base`
  - `tail_end`
  - `far_front_shoulder`
  - `far_rear_hip`
  - cleanup on `t13`
  - any frames where proxy points should be upgraded to better supervision

## Recommended Training Strategy
Do not start from full manual labeling.

### Stage 1: Freeze The Target Schema
Freeze `healthcare-side-final-23` first.

Do not change the schema while labeling is already in progress.

### Stage 2: Build Seed Labels From Each Source
Create two converters:
- `Ultralytics24 -> healthcare-side-final-23`
- `AI-Hub Left/Right -> healthcare-side-final-23`

Rules:
- map `exact` points directly
- map `proxy` points when useful
- write `0 0 0` for `missing`

### Stage 3: Merge Seed Datasets
Merge:
- Ultralytics side-view subset
- AI-Hub `Left` / `Right` subset

Keep metadata:
- source dataset
- view (`left` / `right`)
- which target points were auto-filled
- which target points remain missing

### Stage 4: Manual Labeling Queue
Label only the points that matter most first.

Priority order:
1. `t13_spinous_process`
2. `sacrum`
3. `near_rear_hip`
4. `far_rear_hip`
5. `near_front_shoulder`
6. `far_front_shoulder`
7. cleanup for `withers`

If a far-side point is truly occluded, keep it missing.
Do not hallucinate hidden anatomy.

### Stage 5: Train In Two Phases
Recommended:
1. pretrain on merged auto-projected seed data
2. fine-tune on healthcare-specific manual correction data

This avoids paying the full manual labeling cost up front.

## Immediate Next Tasks
The next concrete tasks should be:

1. Freeze the target schema names and order.
2. Write a mapping table in code for:
   - Ultralytics24
   - AI-Hub Left/Right
3. Generate a seed dataset in the target schema.
4. Export a `manual_label_queue.csv` for the remaining points.
5. Label a small pilot batch first:
   - around 500 to 1500 frames
   - left and right side mixed
   - prioritize clean gait cycles
6. Train a first healthcare model on the pilot set.
7. Update `healthcare-service` to consume the new schema after model quality is confirmed.

## Current Converter
The current target-schema converters are:

- `scripts/convert_pose_to_healthcare_side.py`
- `scripts/convert_aihub_raw_to_healthcare_side.py`
- `scripts/merge_healthcare_side_seeds.py`

Key behavior:
- supports `ultralytics24` and `integrated34`
- projects both sources into `healthcare-side-final-23`
- writes missing points as `0 0 0`
- writes `manual_label_queue.csv`
- writes `flip_idx` for near/far-aware horizontal augmentation
- if `--copy-images` is not used, `train.txt` and `val.txt` point to the original source images with absolute paths
- supports `--invariant-only` for `ultralytics24` when reliable left/right view metadata is unavailable

Raw AI-Hub direct converter behavior:
- reads original JSON annotations from `Training/02.라벨링데이터` and `Validation/02.라벨링데이터`
- matches them to the original images under `01.원천데이터`
- uses only `Left` and `Right` folders
- bypasses the old `integrated34` intermediate dataset entirely
- can optionally map `Dorsal scapular spine -> withers` as a proxy

Seed merge behavior:
- merges multiple already-converted `healthcare-side-final-23` datasets
- keeps label files separated by `source_name`
- concatenates `train.txt` and `val.txt`
- writes a merged `manual_label_queue.csv` with a `source` column

### Example: AI-Hub Left/Right Smoke Test
```bash
python3 scripts/convert_pose_to_healthcare_side.py \
  --input-root dataset_yolo \
  --output-root /tmp/healthcare_side_smoke_integrated34 \
  --source-schema integrated34 \
  --splits train val \
  --view-source filename \
  --limit 100 \
  --sample-per-clip 2
```

`dataset_yolo` uses filenames like `Left_*` and `Right_*`, so `--view-source filename` is the simplest mode there.

### Example: Raw AI-Hub Left/Right Direct Conversion
```bash
python3 scripts/convert_aihub_raw_to_healthcare_side.py \
  --input-root "60.반려견_보행영상_기반_건강관리_데이터" \
  --output-root /tmp/healthcare_side_aihub_raw_lr_seed \
  --splits train val
```

This is now the preferred AI-Hub path because it avoids depending on the old multi-view `integrated34` intermediate dataset.

### Example: Merge AI-Hub And Future Ultralytics Seeds
```bash
python3 scripts/merge_healthcare_side_seeds.py \
  --sources \
    dataset_healthcare_side_aihub_raw_lr_seed \
    /path/to/dataset_healthcare_side_ultralytics24_seed \
  --source-names \
    aihub_raw \
    ultralytics24 \
  --output-root dataset_healthcare_side_merged_seed
```

### Example: Future Ultralytics24 Conversion
```bash
python3 scripts/convert_pose_to_healthcare_side.py \
  --input-root /path/to/ultralytics24_dataset \
  --output-root /tmp/healthcare_side_from_u24 \
  --source-schema ultralytics24 \
  --splits train val \
  --view-source manifest \
  --manifest /path/to/side_view_manifest.csv
```

For Ultralytics24, a manifest is usually safer because the raw dataset often does not encode left/right view clearly enough in filenames alone.

### Example: Ultralytics24 Invariant-Only Conversion
```bash
python3 scripts/convert_pose_to_healthcare_side.py \
  --input-root datasets/dog-pose \
  --output-root dataset_healthcare_side_ultralytics24_invariant_seed \
  --source-schema ultralytics24 \
  --splits train val \
  --view-source auto \
  --invariant-only
```

This mode keeps only view-invariant points such as:
- `nose`
- `withers`
- `tail_base`
- `tail_end`

It is useful when the dataset lacks reliable `left` / `right` side-view metadata, which is the current case for the downloaded Ultralytics dog-pose dataset.

## Important Non-Goals
Avoid these for now:
- mixing `Front` and `Back` frames into the same side-view healthcare model
- forcing a perfect one-to-one match between the source datasets
- trying to fully supervise all 23 points before the first experiment

The correct approach is:
- one target schema
- two source converters
- missing points allowed
- manual labeling only where the sources are weak
