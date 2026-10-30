# Synthetic Floorplan Dataset

This project generates synthetic architectural floorplans for training and evaluating computer-vision models. Each sample has a clean image, a degraded/scanned image, a binary wall mask, and JSON annotations. The files are paired by sample ID; for example, `floorplan_00001` exists in all four output folders.

The saved dataset currently contains 20 samples. Images include multiple canvas sizes, rectangular and non-rectangular footprints, 10-30 apartments, variable room subdivisions, optional shared access halls, and other architectural details. Generation is randomized and does not use a fixed seed, so rerunning it replaces the saved samples with a new dataset.

## Repository Layout

```text
main.py
viewer.html
synthetic_floorplans/
  annotations/   floorplan_XXXXX.json
  clean/         floorplan_XXXXX.png
  images/        floorplan_XXXXX.jpg
  wall_masks/    floorplan_XXXXX.png
```

Open `viewer.html` through a local web server to inspect samples with annotation overlays. The viewer can switch between clean, scanned, and wall-mask images.

## Generate Samples

Install the runtime dependencies and run the generator from the project directory:

```powershell
python -m pip install numpy opencv-python
python main.py
```

The generator writes or replaces files with matching IDs under `synthetic_floorplans/`. It does not create train/validation/test splits.

Important settings near the top of `main.py`:

| Setting | Meaning |
| --- | --- |
| `NUM_IMAGES` | Number of samples to generate. |
| `MIN_APARTMENTS`, `MAX_APARTMENTS` | Apartment count range per building. |
| `MAX_ROOMS` | Maximum rooms retained per plan. |
| `SHARED_ACCESS_HALL_PERCENTAGE` | Percent chance that a plan uses one elongated shared hall. Current value: `70`. |
| `DRAWING_SCALE_PROFILES` | Canvas sizes, nominal scales, and footprint occupancy profiles. |
| `MIN_WINDOWS`, `MAX_WINDOWS` | Requested window count range; placement may yield fewer if no collision-free wall spans remain. |

`SHARED_ACCESS_HALL_PERCENTAGE` is the setting used by the generator. `SHARED_ACCESS_HALL_PROBABILITY` is also present in the current source but is not read by generation.

## Sample Pairing

For a sample ID such as `floorplan_00001`:

- `clean/floorplan_00001.png` is the clean generated drawing.
- `images/floorplan_00001.jpg` is the same drawing with synthetic scan degradation.
- `wall_masks/floorplan_00001.png` is the clean-coordinate wall segmentation mask.
- `annotations/floorplan_00001.json` contains the ground-truth geometry, relationships, measurements, and degradation metadata.

Image dimensions vary by sample. Read `drawing_scale.canvas_width` and `drawing_scale.canvas_height`, or inspect the image itself; do not assume one fixed resolution. OpenCV reads color images in BGR order. Wall masks are single-channel, with `0` for background and `255` for walls.

## Annotation Schema

All coordinates are integer pixel coordinates in the clean image's coordinate system, where `x` increases rightward and `y` downward. Bounding boxes use `[x_min, y_min, x_max, y_max]`. Building outlines are polygons represented as lists of `[x, y]` points. `building_bbox` is the axis-aligned extent of the building; `building_outline` is the actual footprint and is needed for L-, T-, chamfered, or angled plans.

Every annotation file includes these top-level fields:

| Field | Contents |
| --- | --- |
| `building_bbox`, `building_outline`, `footprint_style` | Building extent and footprint polygon/type. |
| `wall_style` | Interior/exterior rendering style, BGR colors, and thicknesses. |
| `drawing_scale` | Nominal scale, canvas dimensions, pixels per meter, DPI assumption, and footprint occupancy. |
| `building_area_px2`, `building_area_m2` | Gross footprint area. |
| `interior_wall_segments`, `interior_wall_length_px`, `interior_wall_length_m` | Unique shared interior-wall segments and total length. |
| `exterior_wall_lengths` | Raster-derived inner- and outer-face perimeter lengths, in pixels and meters. |
| `total_room_area_m2` | Sum of net room areas. |
| `apartment_count`, `apartments` | Apartment boxes and lists of their room IDs. |
| `rooms` | Room boxes, apartment relationship, measurements, and optional visible name. |
| `doors`, `core_doors`, `access_doors` | Internal room doors, apartment-to-core doors, and apartment-to-shared-hall doors. |
| `shared_access_halls`, `common_cores` | Optional shared facilities. These arrays are empty when absent. |
| `entrances`, `windows`, `balconies`, `porches`, `ladders`, `stairs` | Additional architectural features and their boxes. Arrays may be empty. |
| `degradation` | Scan effects and exact geometric transform matrices, when applied. |

### Room and Apartment Targets

Each room has `room_id`, `apartment_id`, `bbox`, `name_displayed`, `measurement_displayed`, and `measurements`. A `name` key is included **only when the room name was actually drawn in the image**. Missing `name` means "no visible room-name target," not an unknown or negative room class. For room-name recognition/detection, train only on rooms whose `name` key is present. For room-region detection, all room boxes are available regardless of name visibility.

Each `measurements` object contains clear width/height and area in pixels and meters: `width_px`, `height_px`, `area_px2`, `width_m`, `height_m`, and `area_m2`. Only some rooms have measurement lines printed; `measurement_displayed` identifies them. Measurement metadata is still computed for every room.

Apartment objects contain `apartment_id`, `bbox`, and `room_ids`. Shared access hall objects contain `hall_id`, `bbox`, `orientation`, `apartment_ids`, and `area_m2`. A hall-bearing plan has one elongated shared room and one `access_doors` entry for each apartment. A `common_cores` object is an alternate shared stairwell/lift core used by some plans; its doors and stair symbols refer to it through `common_core_id`.

Door objects include a `bbox` and relationship fields. Internal `doors` refer to two room IDs in `rooms`; `core_doors` identify the connected apartment, room, and core; `access_doors` identify the connected apartment, room, and hall. All generated rooms have at least one internal or shared-space door connection.

## Coordinate Alignment

`clean` images and `wall_masks` use the annotation coordinates directly. The scanned image has the same canvas dimensions, but it may have been rotated and then perspective-warped. Apply the saved matrices in this order to each annotation point:

1. If `degradation.rotation_matrix` is not `null`, apply its 2x3 affine matrix.
2. If `degradation.perspective_matrix` is not `null`, apply its 3x3 homography, including homogeneous-coordinate division.

For a box under rotation or perspective, transform all four corners. Keep the resulting quadrilateral for polygon-capable models, or take the min/max of transformed corners to derive a new axis-aligned box. Downscaling is resized back to the original canvas size, so it requires no additional coordinate transform. Other degradation fields describe visual noise/effects and do not change coordinates.

Wall masks are saved in clean coordinates and are not independently scanned. To train segmentation on scanned images, warp the corresponding mask with the same matrices using nearest-neighbor interpolation so class values remain `0` and `255`.

## Training Targets

The dataset supports several tasks:

- **Object detection:** rooms, apartments, doors, entrances, core/access doors, windows, balconies, porches, ladders, stairs, shared halls, and common cores. Each item has a `bbox`; relationship fields preserve room/apartment/core connectivity.
- **Room-name detection or recognition:** use only room records containing `name`. Respect `name_displayed`; do not infer a label for omitted names.
- **Wall segmentation:** use `wall_masks/*.png`; internal and exterior walls are foreground, while openings such as doors and windows are cleared from the mask.
- **Footprint segmentation or polygon detection:** use `building_outline`; `building_bbox` alone loses non-rectangular shape information.
- **Measurement prediction:** use the room/building measurement fields as regression targets, considering `drawing_scale.pixels_per_meter` and its 72-DPI assumption.

For normalized YOLO-style `xywh` boxes, with image width `W` and height `H`:

```text
center_x = (x_min + x_max) / (2 * W)
center_y = (y_min + y_max) / (2 * H)
box_width = (x_max - x_min) / W
box_height = (y_max - y_min) / H
```

Map annotation object types to class IDs in your training code; the JSON intentionally retains semantic relationships instead of prescribing one model's class map. Building footprints are polygons, not boxes.

## Minimal Loader

```python
import json
from pathlib import Path

import cv2

root = Path("synthetic_floorplans")
sample_id = "floorplan_00001"

image = cv2.imread(str(root / "clean" / f"{sample_id}.png"))
wall_mask = cv2.imread(
    str(root / "wall_masks" / f"{sample_id}.png"),
    cv2.IMREAD_GRAYSCALE,
)
annotation = json.loads(
    (root / "annotations" / f"{sample_id}.json").read_text(encoding="utf-8")
)

height, width = image.shape[:2]
room_targets = []
for room in annotation["rooms"]:
    x1, y1, x2, y2 = room["bbox"]
    target = {
        "bbox_xywh_normalized": [
            ((x1 + x2) / 2) / width,
            ((y1 + y2) / 2) / height,
            (x2 - x1) / width,
            (y2 - y1) / height,
        ],
        "apartment_id": room["apartment_id"],
    }
    if "name" in room:
        target["name"] = room["name"]
    room_targets.append(target)
```

## Splitting and Caveats

Split by sample ID, keeping each sample's clean image, scanned image, mask, and annotations together. Do not put clean and scanned variants of the same plan in different train/validation/test splits. With the current 20 samples, evaluation is small; increase `NUM_IMAGES` for training-scale experiments. Re-run `python main.py` after changing generation settings; the existing files are not updated automatically. This is synthetic data: validate performance on real plans before drawing conclusions about deployment quality.