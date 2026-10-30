import os
import cv2
import json
import math
import random
import numpy as np
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

OUT_DIR = Path("synthetic_floorplans")
NUM_IMAGES = 20

IMG_W = 1400
IMG_H = 1000

MIN_ROOMS = 10
MAX_ROOMS = 240
MIN_APARTMENTS = 10
MAX_APARTMENTS = 30
SHARED_ACCESS_HALL_PERCENTAGE = 70
SHARED_ACCESS_HALL_PROBABILITY = 0.70

MIN_WINDOWS = 10
MAX_WINDOWS = 50

DRAWING_SCALE_PROFILES = [
    {"label": "1:50", "width": 2800, "height": 2000, "footprint_fraction": 0.88},
    {"label": "1:100", "width": 2200, "height": 1600, "footprint_fraction": 0.76},
    {"label": "1:200", "width": 1800, "height": 1300, "footprint_fraction": 0.64},
]

WALL_THICKNESS = 10
OUTER_WALL_THICKNESS = 18

FONT = cv2.FONT_HERSHEY_SIMPLEX

ROOM_NAMES = [
    "BEDROOM",
    "LIVING ROOM",
    "KITCHEN",
    "BATHROOM",
    "WC",
    "HALL",
    "STORAGE",
    "OFFICE",
    "DINING",
    "CORRIDOR",
    "UTILITY",
    "WARDROBE",
]


# ============================================================
# COLORS
# ============================================================

# OpenCV uses BGR

OUTER_WALL_RED_SHADES = [
    (20, 20, 190),
    (25, 25, 205),
    (30, 30, 220),
    (35, 35, 235),
    (40, 40, 245),
    (45, 45, 210),
]

OUTER_WALL_COLORS = [
    (25, 25, 190),
    (45, 70, 105),
    (55, 105, 65),
    (25, 90, 150),
    (70, 75, 80),
    (35, 115, 145),
]

INNER_WALL_COLORS = [
    (20, 20, 20),
    (35, 35, 35),
    (50, 50, 50),
    (60, 45, 30),
    (45, 65, 75),
    (55, 75, 45),
]

ROOM_FILL_COLORS = [
    (245, 245, 255),
    (255, 248, 235),
    (240, 255, 240),
    (245, 250, 255),
    (255, 245, 255),
    (245, 255, 255),
    (250, 250, 250),
    (245, 250, 245),
]

DOOR_COLORS = [
    (30, 30, 30),
    (70, 45, 20),
    (35, 60, 95),
]

WINDOW_COLORS = [
    (170, 95, 20),
    (205, 120, 30),
    (140, 80, 20),
    (210, 145, 55),
]

TEXT_COLORS = [
    (20, 20, 20),
    (40, 40, 40),
    (60, 60, 60),
]


# ============================================================
# GEOMETRY
# ============================================================

def split_rectangle(rect, depth=0, max_depth=7):
    """
    Recursive BSP subdivision.

    rect = (x1, y1, x2, y2)
    """

    x1, y1, x2, y2 = rect

    w = x2 - x1
    h = y2 - y1

    if depth >= max_depth:
        return [rect]

    if w < 80 or h < 80:
        return [rect]

    if depth > 1 and random.random() < 0.12:
        return [rect]

    rooms = []

    if w / h > 1.25:
        orientation = "vertical"

    elif h / w > 1.25:
        orientation = "horizontal"

    else:
        orientation = random.choice([
            "vertical",
            "horizontal"
        ])

    if orientation == "vertical":

        min_child_width = max(50, min(110, int(w * random.uniform(0.18, 0.36))))
        min_split = x1 + min_child_width
        max_split = x2 - max(50, min(110, int(w * random.uniform(0.18, 0.36))))

        if max_split <= min_split:
            return [rect]

        split = random.randint(
            min_split,
            max_split
        )

        rooms += split_rectangle(
            (
                x1,
                y1,
                split,
                y2
            ),
            depth + 1,
            max_depth
        )

        rooms += split_rectangle(
            (
                split,
                y1,
                x2,
                y2
            ),
            depth + 1,
            max_depth
        )

    else:

        min_child_height = max(50, min(100, int(h * random.uniform(0.18, 0.36))))
        min_split = y1 + min_child_height
        max_split = y2 - max(50, min(100, int(h * random.uniform(0.18, 0.36))))

        if max_split <= min_split:
            return [rect]

        split = random.randint(
            min_split,
            max_split
        )

        rooms += split_rectangle(
            (
                x1,
                y1,
                x2,
                split
            ),
            depth + 1,
            max_depth
        )

        rooms += split_rectangle(
            (
                x1,
                split,
                x2,
                y2
            ),
            depth + 1,
            max_depth
        )

    return rooms


def shared_wall(a, b, tolerance=3):
    """
    Find common wall between two rectangular rooms.

    Returns:
        ("vertical", x, y1, y2)
        ("horizontal", y, x1, x2)
        None
    """

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    # vertical wall
    if (
        abs(ax2 - bx1) <= tolerance
        or abs(bx2 - ax1) <= tolerance
    ):

        x = (
            ax2
            if abs(ax2 - bx1) <= tolerance
            else ax1
        )

        y1 = max(
            ay1,
            by1
        )

        y2 = min(
            ay2,
            by2
        )

        if y2 - y1 > 50:
            return (
                "vertical",
                x,
                y1,
                y2
            )

    # horizontal wall
    if (
        abs(ay2 - by1) <= tolerance
        or abs(by2 - ay1) <= tolerance
    ):

        y = (
            ay2
            if abs(ay2 - by1) <= tolerance
            else ay1
        )

        x1 = max(
            ax1,
            bx1
        )

        x2 = min(
            ax2,
            bx2
        )

        if x2 - x1 > 50:
            return (
                "horizontal",
                y,
                x1,
                x2
            )

    return None


def polygon_area(points):
    area_twice = 0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area_twice += x1 * y2 - x2 * y1
    return abs(area_twice) / 2


def calculate_internal_wall_segments(rooms, pixels_per_meter):
    segments_by_line = {}

    for first_index, first in enumerate(rooms):
        ax1, ay1, ax2, ay2 = first
        for second in rooms[first_index + 1:]:
            bx1, by1, bx2, by2 = second

            if ax2 == bx1 or bx2 == ax1:
                start = max(ay1, by1)
                end = min(ay2, by2)
                if end > start:
                    coordinate = ax2 if ax2 == bx1 else ax1
                    key = ("vertical", coordinate)
                    segments_by_line.setdefault(key, []).append((start, end))

            if ay2 == by1 or by2 == ay1:
                start = max(ax1, bx1)
                end = min(ax2, bx2)
                if end > start:
                    coordinate = ay2 if ay2 == by1 else ay1
                    key = ("horizontal", coordinate)
                    segments_by_line.setdefault(key, []).append((start, end))

    segments = []
    for (orientation, coordinate), intervals in segments_by_line.items():
        intervals.sort()
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        for start, end in merged:
            length_px = end - start
            if orientation == "vertical":
                start_point = [coordinate, start]
                end_point = [coordinate, end]
            else:
                start_point = [start, coordinate]
                end_point = [end, coordinate]
            segments.append({
                "orientation": orientation,
                "start": start_point,
                "end": end_point,
                "length_px": length_px,
                "length_m": round(length_px / pixels_per_meter, 2),
            })

    return segments


def calculate_exterior_wall_face_lengths(wall_mask, pixels_per_meter):
    contours, hierarchy = cv2.findContours(
        wall_mask.copy(),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    inner_length_px = 0.0
    outer_length_px = 0.0
    if hierarchy is not None:
        for index, contour in enumerate(contours):
            length = cv2.arcLength(contour, True)
            if hierarchy[0][index][3] < 0:
                outer_length_px += length
            else:
                inner_length_px += length

    return {
        "inner_side_px": round(inner_length_px, 2),
        "inner_side_m": round(inner_length_px / pixels_per_meter, 2),
        "outer_side_px": round(outer_length_px, 2),
        "outer_side_m": round(outer_length_px / pixels_per_meter, 2),
    }


def create_building_geometry(building, style):
    x1, y1, x2, y2 = building

    if style == "rectangular":
        outline = [
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
        ]
        return outline, [building]

    if style == "L-shaped":
        split_x = x1 + int((x2 - x1) * random.uniform(0.48, 0.62))
        split_y = y1 + int((y2 - y1) * random.uniform(0.42, 0.58))
        outline = [
            (x1, y1), (x2, y1), (x2, split_y),
            (split_x, split_y), (split_x, y2), (x1, y2),
        ]
        return outline, [
            (x1, y1, x2, split_y),
            (x1, split_y, split_x, y2),
        ]

    if style == "T-shaped":
        split_y = y1 + int((y2 - y1) * random.uniform(0.38, 0.52))
        stem_width = int((x2 - x1) * random.uniform(0.36, 0.52))
        stem_x1 = (x1 + x2 - stem_width) // 2
        stem_x2 = stem_x1 + stem_width
        outline = [
            (x1, y1), (x2, y1), (x2, split_y),
            (stem_x2, split_y), (stem_x2, y2),
            (stem_x1, y2), (stem_x1, split_y), (x1, split_y),
        ]
        return outline, [
            (x1, y1, x2, split_y),
            (stem_x1, split_y, stem_x2, y2),
        ]

    if style == "chamfered":
        cuts = [
            random.randint(28, 52)
            for _ in range(4)
        ]
    else:
        cuts = [0, 0, 0, 0]
        cuts[random.randrange(4)] = random.randint(75, 115)

    top_left, top_right, bottom_right, bottom_left = cuts
    outline = [
        (x1 + top_left, y1),
        (x2 - top_right, y1),
        (x2, y1 + top_right),
        (x2, y2 - bottom_right),
        (x2 - bottom_right, y2),
        (x1 + bottom_left, y2),
        (x1, y2 - bottom_left),
        (x1, y1 + top_left),
    ]
    return outline, [building]


def create_apartment_zones(regions, apartment_count):
    areas = [
        (x2 - x1) * (y2 - y1)
        for x1, y1, x2, y2 in regions
    ]
    total_area = sum(areas)
    expected = [
        apartment_count * area / total_area
        for area in areas
    ]
    counts = [max(1, int(value)) for value in expected]

    while sum(counts) < apartment_count:
        index = max(
            range(len(counts)),
            key=lambda i: expected[i] - counts[i]
        )
        counts[index] += 1

    while sum(counts) > apartment_count:
        candidates = [i for i, count in enumerate(counts) if count > 1]
        index = max(
            candidates,
            key=lambda i: counts[i] - expected[i]
        )
        counts[index] -= 1

    zones = []
    apartment_id = 0
    for (x1, y1, x2, y2), count in zip(regions, counts):
        width = x2 - x1
        height = y2 - y1
        rows = max(1, min(count, round(math.sqrt(count * height / width))))
        base_count, extra = divmod(count, rows)

        for row in range(rows):
            row_count = base_count + (1 if row < extra else 0)
            row_y1 = y1 + round(height * row / rows)
            row_y2 = y1 + round(height * (row + 1) / rows)

            for column in range(row_count):
                zone_x1 = x1 + round(width * column / row_count)
                zone_x2 = x1 + round(width * (column + 1) / row_count)
                zones.append({
                    "id": apartment_id,
                    "bbox": (zone_x1, row_y1, zone_x2, row_y2),
                })
                apartment_id += 1

    return zones


def create_shared_access_layout(building, apartment_count):
    x1, y1, x2, y2 = building
    width, height = x2 - x1, y2 - y1
    apartment_zones = []

    if width >= height:
        hall_height = int(height * random.uniform(0.13, 0.19))
        hall_center = y1 + int(height * random.uniform(0.42, 0.58))
        hall_x1 = x1
        hall_x2 = x2
        hall_y1 = hall_center - hall_height // 2
        hall_y2 = hall_y1 + hall_height
        hall = (hall_x1, hall_y1, hall_x2, hall_y2)
        regions = [
            (x1, y1, x2, hall_y1),
            (x1, hall_y2, x2, y2),
        ]
        counts = [
            (apartment_count + 1) // 2,
            apartment_count // 2,
        ]
        for (region_x1, region_y1, region_x2, region_y2), count in zip(regions, counts):
            if count == 0:
                continue
            row_width = region_x2 - region_x1
            for column in range(count):
                zone_x1 = region_x1 + round(row_width * column / count)
                zone_x2 = region_x1 + round(row_width * (column + 1) / count)
                apartment_zones.append({
                    "id": len(apartment_zones),
                    "bbox": (zone_x1, region_y1, zone_x2, region_y2),
                })
    else:
        hall_width = int(width * random.uniform(0.13, 0.19))
        hall_center = x1 + int(width * random.uniform(0.42, 0.58))
        hall_y1 = y1
        hall_y2 = y2
        hall_x1 = hall_center - hall_width // 2
        hall_x2 = hall_x1 + hall_width
        hall = (hall_x1, hall_y1, hall_x2, hall_y2)
        regions = [
            (x1, y1, hall_x1, y2),
            (hall_x2, y1, x2, y2),
        ]
        counts = [
            (apartment_count + 1) // 2,
            apartment_count // 2,
        ]
        for (region_x1, region_y1, region_x2, region_y2), count in zip(regions, counts):
            if count == 0:
                continue
            region_height = region_y2 - region_y1
            for row in range(count):
                zone_y1 = region_y1 + round(region_height * row / count)
                zone_y2 = region_y1 + round(region_height * (row + 1) / count)
                apartment_zones.append({
                    "id": len(apartment_zones),
                    "bbox": (region_x1, zone_y1, region_x2, zone_y2),
                })

    return hall, apartment_zones


def extract_common_core_zones(apartment_zones, core_count, minimum_size):
    remaining = list(apartment_zones)
    cores = []

    for _ in range(core_count):
        candidates = []
        for zone in remaining:
            x1, y1, x2, y2 = zone["bbox"]
            if min(x2 - x1, y2 - y1) < minimum_size:
                continue
            neighbors = [
                other
                for other in remaining
                if other is not zone
                and shared_wall(zone["bbox"], other["bbox"], tolerance=0) is not None
            ]
            if len(neighbors) >= 2:
                candidates.append((zone, neighbors))

        if not candidates:
            break

        zone, neighbors = random.choice(candidates)
        remaining.remove(zone)
        cores.append({
            "id": len(cores),
            "bbox": zone["bbox"],
            "apartment_ids": [neighbor["id"] for neighbor in neighbors],
        })

    id_map = {}
    for new_id, zone in enumerate(remaining):
        id_map[zone["id"]] = new_id
        zone["id"] = new_id

    for core in cores:
        core["apartment_ids"] = [
            id_map[apartment_id]
            for apartment_id in core["apartment_ids"]
            if apartment_id in id_map
        ]

    return cores, remaining


def apply_wall_pattern(img, wall_mask, color, style):
    if style == "solid":
        return

    height, width = wall_mask.shape
    pattern = np.zeros_like(wall_mask)
    spacing = random.randint(10, 17)

    if style in ("diagonal", "crosshatch", "outlined_hatch"):
        for offset in range(-height, width, spacing):
            cv2.line(
                pattern,
                (offset, 0),
                (offset + height, height),
                (255,),
                1,
                cv2.LINE_8
            )

            if style in ("crosshatch", "outlined_hatch"):
                cv2.line(
                    pattern,
                    (offset, height),
                    (offset + height, 0),
                    (255,),
                    1,
                    cv2.LINE_8
                )

    elif style == "brick":
        for row, y in enumerate(range(0, height, spacing)):
            cv2.line(pattern, (0, y), (width, y), (255,), 1, cv2.LINE_8)
            offset = spacing // 2 if row % 2 else 0
            for x in range(offset, width, spacing):
                cv2.line(
                    pattern,
                    (x, y),
                    (x, min(height - 1, y + spacing)),
                    (255,),
                    1,
                    cv2.LINE_8
                )

    elif style == "stipple":
        for y in range(spacing // 2, height, spacing):
            for x in range(spacing // 2, width, spacing):
                cv2.circle(pattern, (x, y), 1, (255,), -1)

    pattern = cv2.bitwise_and(pattern, wall_mask)
    if sum(color) // 3 < 120:
        pattern_color = tuple(min(255, int(channel * 1.7)) for channel in color)
    else:
        pattern_color = tuple(int(channel * 0.62) for channel in color)
    img[pattern > 0] = pattern_color

    if style == "outlined_hatch":
        contours, _ = cv2.findContours(
            wall_mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        outline_color = tuple(max(0, int(channel * 0.45)) for channel in color)
        cv2.drawContours(img, contours, -1, outline_color, 2, cv2.LINE_AA)


# ============================================================
# ROOM FILLS
# ============================================================

def fill_rooms(img, rooms, footprint_mask, wall_thickness):
    """
    Light colored room fills.
    """

    overlay = img.copy()

    for room in rooms:

        x1, y1, x2, y2 = room

        margin = wall_thickness + 2

        if (
            x2 - x1 < margin * 2
            or y2 - y1 < margin * 2
        ):
            continue

        color = random.choice(
            ROOM_FILL_COLORS
        )

        cv2.rectangle(
            overlay,
            (
                x1 + margin,
                y1 + margin
            ),
            (
                x2 - margin,
                y2 - margin
            ),
            color,
            -1
        )

    alpha = random.uniform(
        0.20,
        0.55
    )

    blended = cv2.addWeighted(
        overlay,
        alpha,
        img,
        1.0 - alpha,
        0
    )

    result = img.copy()
    result[footprint_mask > 0] = blended[footprint_mask > 0]
    return result


# ============================================================
# LABELS
# ============================================================

def draw_room_labels(
    img,
    rooms,
    apartment_ids,
    detail_scale=1.0,
    pixels_per_meter=1.0,
    footprint_mask=None,
    wall_thickness=0,
    measurement_room_ids=None,
    room_name_probability=0.75,
    visible_room_ids=None
):
    labels = []

    for i, room in enumerate(rooms):

        x1, y1, x2, y2 = room

        name = random.choice(
            ROOM_NAMES
        )
        name_displayed = (
            i in visible_room_ids
            if visible_room_ids is not None
            else random.random() < room_name_probability
        )

        cx = (
            x1 + x2
        ) // 2

        cy = (
            y1 + y2
        ) // 2

        name_font_scale = random.uniform(0.35, 0.62) * detail_scale

        thickness = random.choice([
            1,
            1,
            1,
            2
        ])

        color = random.choice(
            TEXT_COLORS
        )

        width_px = max(0, x2 - x1 - wall_thickness)
        height_px = max(0, y2 - y1 - wall_thickness)
        if footprint_mask is None:
            area_px2 = width_px * height_px
        else:
            inset = wall_thickness // 2
            inner_x1 = min(x2, x1 + inset)
            inner_y1 = min(y2, y1 + inset)
            inner_x2 = max(inner_x1, x2 - inset)
            inner_y2 = max(inner_y1, y2 - inset)
            area_px2 = int(np.count_nonzero(
                footprint_mask[inner_y1:inner_y2, inner_x1:inner_x2]
            ))
        width_m = width_px / pixels_per_meter
        height_m = height_px / pixels_per_meter
        area_m2 = area_px2 / (pixels_per_meter ** 2)
        show_measurements = (
            measurement_room_ids is not None
            and i in measurement_room_ids
        )
        dimension_text = f"{width_m:.1f} x {height_m:.1f} m"
        area_text = f"{area_m2:.1f} m2"

        text_specs = []
        if name_displayed:
            text_specs.append((name, name_font_scale, thickness))
        if show_measurements:
            text_specs.extend([
                (dimension_text, 0.30 * detail_scale, 1),
                (area_text, 0.30 * detail_scale, 1),
            ])

            inset = wall_thickness // 2
            inner_x1 = x1 + inset
            inner_x2 = x2 - inset
            inner_y1 = y1 + inset
            inner_y2 = y2 - inset
            line_inset = int(18 * detail_scale)

            if inner_x2 - inner_x1 >= int(60 * detail_scale):
                dimension_y = inner_y2 - line_inset
                draw_dimension(
                    img,
                    (inner_x1, dimension_y),
                    (inner_x2, dimension_y),
                    offset=0,
                    pixels_per_meter=pixels_per_meter
                )

            if inner_y2 - inner_y1 >= int(60 * detail_scale):
                dimension_x = inner_x1 + line_inset
                draw_dimension(
                    img,
                    (dimension_x, inner_y1),
                    (dimension_x, inner_y2),
                    offset=0,
                    pixels_per_meter=pixels_per_meter
                )
        if text_specs:
            measured_specs = [
                (text, scale, line_thickness, cv2.getTextSize(text, FONT, scale, line_thickness)[0])
                for text, scale, line_thickness in text_specs
            ]
            max_text_width = max(24, width_px - 12)
            max_text_height = max(20, height_px - 10)
            total_height = sum(size[1] for _, _, _, size in measured_specs)
            total_height += max(0, len(text_specs) - 1) * max(1, int(2 * detail_scale))
            widest_text = max(size[0] for _, _, _, size in measured_specs)
            fit_factor = min(
                1.0,
                max_text_width / max(1, widest_text),
                max_text_height / max(1, total_height)
            )

            fitted_specs = []
            for text, scale, line_thickness in text_specs:
                scale *= fit_factor
                (text_width, text_height), _ = cv2.getTextSize(
                    text,
                    FONT,
                    scale,
                    line_thickness
                )
                fitted_specs.append((text, scale, line_thickness, text_width, text_height))

            line_gap = max(1, int(2 * detail_scale))
            total_height = sum(spec[4] for spec in fitted_specs) + line_gap * max(0, len(fitted_specs) - 1)
            cursor_y = cy - total_height // 2
            for text, scale, line_thickness, text_width, text_height in fitted_specs:
                cv2.putText(
                    img,
                    text,
                    (cx - text_width // 2, cursor_y + text_height),
                    FONT,
                    scale,
                    color,
                    line_thickness,
                    cv2.LINE_AA
                )
                cursor_y += text_height + line_gap

        room_annotation = {
            "room_id": i,
            "apartment_id": apartment_ids[i],
            "name_displayed": name_displayed,
            "bbox": list(room),
            "measurement_displayed": show_measurements,
            "measurements": {
                "width_px": width_px,
                "height_px": height_px,
                "width_m": round(width_m, 2),
                "height_m": round(height_m, 2),
                "area_px2": area_px2,
                "area_m2": round(area_m2, 2),
            }
        }
        if name_displayed:
            room_annotation["name"] = name
        labels.append(room_annotation)

    return labels


# ============================================================
# DIMENSIONS
# ============================================================

def draw_dimension(
    img,
    p1,
    p2,
    offset=25,
    pixels_per_meter=None
):
    x1, y1 = p1
    x2, y2 = p2

    color = random.choice([
        (60, 60, 60),
        (80, 80, 80),
        (100, 100, 100)
    ])

    if abs(y1 - y2) < 5:

        y = y1 + offset

        cv2.line(
            img,
            (x1, y),
            (x2, y),
            color,
            1
        )

        cv2.line(
            img,
            (x1, y - 6),
            (x1, y + 6),
            color,
            1
        )

        cv2.line(
            img,
            (x2, y - 6),
            (x2, y + 6),
            color,
            1
        )

        dist = abs(
            x2 - x1
        )

        distance_label = (
            f"{dist / pixels_per_meter:.1f} m"
            if pixels_per_meter
            else str(dist)
        )

        cv2.putText(
            img,
            distance_label,
            (
                (x1 + x2) // 2 - 15,
                y - 5
            ),
            FONT,
            0.35,
            color,
            1,
            cv2.LINE_AA
        )

    elif abs(x1 - x2) < 5:

        x = x1 + offset

        cv2.line(
            img,
            (x, y1),
            (x, y2),
            color,
            1
        )

        cv2.line(
            img,
            (x - 6, y1),
            (x + 6, y1),
            color,
            1
        )

        cv2.line(
            img,
            (x - 6, y2),
            (x + 6, y2),
            color,
            1
        )

        dist = abs(
            y2 - y1
        )

        distance_label = (
            f"{dist / pixels_per_meter:.1f} m"
            if pixels_per_meter
            else str(dist)
        )

        cv2.putText(
            img,
            distance_label,
            (
                x + 5,
                (y1 + y2) // 2
            ),
            FONT,
            0.35,
            color,
            1,
            cv2.LINE_AA
        )


# ============================================================
# DOORS
# ============================================================

def draw_door(
    img,
    mask,
    wall,
    door_width=65,
    wall_thickness=WALL_THICKNESS
):
    orientation = wall[0]

    door_color = random.choice(
        DOOR_COLORS
    )

    arc_color = tuple(
        min(
            255,
            c + 35
        )
        for c in door_color
    )

    data = {
        "type": "door"
    }

    if orientation == "vertical":

        _, x, y1, y2 = wall

        if y2 - y1 < door_width + 30:
            return None

        cy = random.randint(
            y1 + door_width // 2 + 10,
            y2 - door_width // 2 - 10
        )

        d1 = (
            cy - door_width // 2
        )

        d2 = (
            cy + door_width // 2
        )

        cv2.rectangle(
            img,
            (
                x - wall_thickness,
                d1
            ),
            (
                x + wall_thickness,
                d2
            ),
            (255, 255, 255),
            -1
        )

        cv2.rectangle(
            mask,
            (
                x - wall_thickness,
                d1
            ),
            (
                x + wall_thickness,
                d2
            ),
            0,
            -1
        )

        direction = random.choice([
            -1,
            1
        ])

        cv2.line(
            img,
            (
                x,
                d1
            ),
            (
                x + direction * door_width,
                d1
            ),
            door_color,
            2
        )

        if direction > 0:

            cv2.ellipse(
                img,
                (
                    x,
                    d1
                ),
                (
                    door_width,
                    door_width
                ),
                0,
                0,
                90,
                arc_color,
                1
            )

        else:

            cv2.ellipse(
                img,
                (
                    x,
                    d1
                ),
                (
                    door_width,
                    door_width
                ),
                180,
                0,
                90,
                arc_color,
                1
            )

        data["bbox"] = [
            x - 5,
            d1,
            x + 5,
            d2
        ]

    else:

        _, y, x1, x2 = wall

        if x2 - x1 < door_width + 30:
            return None

        cx = random.randint(
            x1 + door_width // 2 + 10,
            x2 - door_width // 2 - 10
        )

        d1 = (
            cx - door_width // 2
        )

        d2 = (
            cx + door_width // 2
        )

        cv2.rectangle(
            img,
            (
                d1,
                y - wall_thickness
            ),
            (
                d2,
                y + wall_thickness
            ),
            (255, 255, 255),
            -1
        )

        cv2.rectangle(
            mask,
            (
                d1,
                y - wall_thickness
            ),
            (
                d2,
                y + wall_thickness
            ),
            0,
            -1
        )

        direction = random.choice([
            -1,
            1
        ])

        cv2.line(
            img,
            (
                d1,
                y
            ),
            (
                d1,
                y + direction * door_width
            ),
            door_color,
            2
        )

        if direction > 0:

            cv2.ellipse(
                img,
                (
                    d1,
                    y
                ),
                (
                    door_width,
                    door_width
                ),
                90,
                0,
                90,
                arc_color,
                1
            )

        else:

            cv2.ellipse(
                img,
                (
                    d1,
                    y
                ),
                (
                    door_width,
                    door_width
                ),
                270,
                0,
                90,
                arc_color,
                1
            )

        data["bbox"] = [
            d1,
            y - 5,
            d2,
            y + 5
        ]

    return data


def draw_room_connection_doors(
    img,
    wall_mask,
    rooms,
    room_apartment_ids,
    wall_thickness,
    detail_scale
):
    doors = []
    rooms_by_apartment = {}
    for room_id, apartment_id in enumerate(room_apartment_ids):
        rooms_by_apartment.setdefault(apartment_id, []).append(room_id)

    for apartment_id, room_ids in rooms_by_apartment.items():
        if len(room_ids) < 2:
            continue

        edges = []
        for index, first_room_id in enumerate(room_ids):
            for second_room_id in room_ids[index + 1:]:
                wall = shared_wall(
                    rooms[first_room_id],
                    rooms[second_room_id],
                    tolerance=0
                )
                if wall:
                    edges.append((first_room_id, second_room_id, wall, wall[3] - wall[2]))

        connected = {room_ids[0]}
        used_edges = set()
        while len(connected) < len(room_ids):
            candidates = [
                edge
                for edge in edges
                if (edge[0] in connected) != (edge[1] in connected)
                and (edge[0], edge[1]) not in used_edges
            ]
            if not candidates:
                break

            candidates.sort(key=lambda edge: edge[3], reverse=True)
            edge = random.choice(candidates[:min(4, len(candidates))])
            first_room_id, second_room_id, wall, segment_length = edge
            used_edges.add((first_room_id, second_room_id))
            maximum_width = min(
                int(random.randint(50, 80) * detail_scale),
                segment_length - 31
            )
            minimum_width = min(int(36 * detail_scale), maximum_width)
            door = draw_door(
                img,
                wall_mask,
                wall,
                random.randint(minimum_width, maximum_width),
                wall_thickness
            )
            if not door:
                continue

            door["rooms"] = [first_room_id, second_room_id]
            door["apartment_id"] = apartment_id
            door["connects_rooms"] = True
            doors.append(door)
            connected.add(
                second_room_id if first_room_id in connected else first_room_id
            )

        for first_room_id, second_room_id, wall, segment_length in edges:
            if (first_room_id, second_room_id) in used_edges or random.random() >= 0.12:
                continue
            maximum_width = min(
                int(random.randint(50, 80) * detail_scale),
                segment_length - 31
            )
            minimum_width = min(int(36 * detail_scale), maximum_width)
            door = draw_door(
                img,
                wall_mask,
                wall,
                random.randint(minimum_width, maximum_width),
                wall_thickness
            )
            if door:
                door["rooms"] = [first_room_id, second_room_id]
                door["apartment_id"] = apartment_id
                door["connects_rooms"] = True
                doors.append(door)

    return doors


def draw_shared_access_doors(
    img,
    wall_mask,
    hall,
    rooms,
    room_apartment_ids,
    apartment_ids,
    wall_thickness,
    detail_scale
):
    doors = []
    for apartment_id in apartment_ids:
        candidates = []
        for room_id, room in enumerate(rooms):
            if room_apartment_ids[room_id] != apartment_id:
                continue
            wall = shared_wall(hall, room, tolerance=0)
            if wall:
                candidates.append((wall[3] - wall[2], room_id, wall))

        if not candidates:
            continue

        segment_length, room_id, wall = max(candidates, key=lambda item: item[0])
        maximum_width = min(
            int(random.randint(46, 64) * detail_scale),
            segment_length - 31
        )
        minimum_width = min(maximum_width, max(10, int(12 * detail_scale)))
        if maximum_width < minimum_width or maximum_width <= 0:
            continue

        door = draw_door(
            img,
            wall_mask,
            wall,
            random.randint(minimum_width, maximum_width),
            wall_thickness
        )
        if door:
            door["type"] = "shared_access_door"
            door["shared_access_hall_id"] = 0
            door["apartment_id"] = apartment_id
            door["room_id"] = room_id
            doors.append(door)

    return doors


# ============================================================
# WINDOWS
# ============================================================

def available_exterior_center_ranges(start, end, width, occupied, gap):
    half_width = width // 2
    left = start + half_width
    right = end - half_width
    ranges = []

    for occupied_start, occupied_end in sorted(occupied):
        before = occupied_start - half_width - gap
        if before >= left:
            ranges.append((left, min(right, before)))
        left = max(left, occupied_end + half_width + gap)
        if left > right:
            return ranges

    if left <= right:
        ranges.append((left, right))
    return ranges


def draw_window(
    img,
    mask,
    outer_rect,
    outer_wall_thickness=OUTER_WALL_THICKNESS,
    footprint=None,
    detail_scale=1.0,
    occupied_windows=None
):
    x1, y1, x2, y2 = outer_rect

    footprint = footprint or [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
    ]

    side_bounds = {
        "top": [x for x, y in footprint if y == y1],
        "bottom": [x for x, y in footprint if y == y2],
        "left": [y for x, y in footprint if x == x1],
        "right": [y for x, y in footprint if x == x2],
    }

    side_bounds = {
        side: (min(values), max(values))
        for side, values in side_bounds.items()
        if values
    }

    width = int(random.randint(60, 130) * detail_scale)
    minimum_gap = max(
        outer_wall_thickness * 2,
        int(20 * detail_scale)
    )
    occupied_windows = occupied_windows if occupied_windows is not None else {}

    side_options = []
    for side, (start, end) in side_bounds.items():
        center_ranges = available_exterior_center_ranges(
            start,
            end,
            width,
            occupied_windows.get(side, []),
            minimum_gap
        )
        if center_ranges:
            side_options.append((side, center_ranges))

    if not side_options:
        return None

    side, center_ranges = random.choice(side_options)
    center_start, center_end = random.choice(center_ranges)
    center = random.randint(center_start, center_end)

    window_color = random.choice(
        WINDOW_COLORS
    )

    if side in [
        "top",
        "bottom"
    ]:

        cx = center

        xa = (
            cx - width // 2
        )

        xb = (
            cx + width // 2
        )

        y = (
            y1
            if side == "top"
            else y2
        )

        cv2.rectangle(
            img,
            (
                xa,
                y - outer_wall_thickness
            ),
            (
                xb,
                y + outer_wall_thickness
            ),
            (255, 255, 255),
            -1
        )

        cv2.rectangle(
            mask,
            (
                xa,
                y - outer_wall_thickness
            ),
            (
                xb,
                y + outer_wall_thickness
            ),
            0,
            -1
        )

        cv2.line(
            img,
            (
                xa,
                y - 4
            ),
            (
                xb,
                y - 4
            ),
            window_color,
            2
        )

        cv2.line(
            img,
            (
                xa,
                y + 4
            ),
            (
                xb,
                y + 4
            ),
            window_color,
            2
        )

        bbox = [
            xa,
            y - 8,
            xb,
            y + 8
        ]
        occupied_windows.setdefault(side, []).append((xa, xb))

    else:
        cy = center

        ya = (
            cy - width // 2
        )

        yb = (
            cy + width // 2
        )

        x = (
            x1
            if side == "left"
            else x2
        )

        cv2.rectangle(
            img,
            (
                x - outer_wall_thickness,
                ya
            ),
            (
                x + outer_wall_thickness,
                yb
            ),
            (255, 255, 255),
            -1
        )

        cv2.rectangle(
            mask,
            (
                x - outer_wall_thickness,
                ya
            ),
            (
                x + outer_wall_thickness,
                yb
            ),
            0,
            -1
        )

        cv2.line(
            img,
            (
                x - 4,
                ya
            ),
            (
                x - 4,
                yb
            ),
            window_color,
            2
        )

        cv2.line(
            img,
            (
                x + 4,
                ya
            ),
            (
                x + 4,
                yb
            ),
            window_color,
            2
        )

        bbox = [
            x - 8,
            ya,
            x + 8,
            yb
        ]
        occupied_windows.setdefault(side, []).append((ya, yb))

    return {
        "type": "window",
        "side": side,
        "bbox": bbox
    }


def get_exterior_edges(footprint):
    xs = [point[0] for point in footprint]
    ys = [point[1] for point in footprint]
    center_x = (min(xs) + max(xs)) // 2
    center_y = (min(ys) + max(ys)) // 2
    edges = []

    for index, (x1, y1) in enumerate(footprint):
        x2, y2 = footprint[(index + 1) % len(footprint)]
        if y1 == y2 and abs(x2 - x1) >= 140:
            edges.append({
                "side": "top" if y1 < center_y else "bottom",
                "start": min(x1, x2),
                "end": max(x1, x2),
                "coordinate": y1,
            })
        elif x1 == x2 and abs(y2 - y1) >= 140:
            edges.append({
                "side": "left" if x1 < center_x else "right",
                "start": min(y1, y2),
                "end": max(y1, y2),
                "coordinate": x1,
            })

    return edges


def draw_balcony(
    img,
    mask,
    edge,
    wall_thickness,
    detail_scale=1.0,
    occupied_openings=None
):
    width = int(random.randint(100, 160) * detail_scale)
    depth = int(random.randint(45, 75) * detail_scale)
    if occupied_openings is None:
        occupied_openings = {}

    side = edge["side"]
    gap = max(wall_thickness * 2, int(20 * detail_scale))
    center_ranges = available_exterior_center_ranges(
        edge["start"],
        edge["end"],
        width,
        occupied_openings.get(side, []),
        gap
    )
    if not center_ranges:
        return None

    center_start, center_end = random.choice(center_ranges)
    center = random.randint(center_start, center_end)
    color = random.choice([
        (225, 225, 225),
        (215, 225, 230),
        (230, 220, 205),
    ])
    rail_color = random.choice(INNER_WALL_COLORS)
    half_width = width // 2
    coordinate = edge["coordinate"]

    if side in ("top", "bottom"):
        x1, x2 = center - half_width, center + half_width
        direction = -1 if side == "top" else 1
        outer = coordinate + direction * depth
        cv2.rectangle(img, (x1, coordinate), (x2, outer), color, -1)
        cv2.line(img, (x1, outer), (x2, outer), rail_color, 2)
        cv2.line(img, (x1, coordinate), (x1, outer), rail_color, 2)
        cv2.line(img, (x2, coordinate), (x2, outer), rail_color, 2)
        opening = int(28 * detail_scale)
        cv2.rectangle(
            img,
            (center - opening, coordinate - wall_thickness),
            (center + opening, coordinate + wall_thickness),
            (255, 255, 255),
            -1
        )
        cv2.rectangle(
            mask,
            (center - opening, coordinate - wall_thickness),
            (center + opening, coordinate + wall_thickness),
            0,
            -1
        )
        bbox = [x1, min(coordinate, outer), x2, max(coordinate, outer)]
    else:
        y1, y2 = center - half_width, center + half_width
        direction = -1 if side == "left" else 1
        outer = coordinate + direction * depth
        cv2.rectangle(img, (coordinate, y1), (outer, y2), color, -1)
        cv2.line(img, (outer, y1), (outer, y2), rail_color, 2)
        cv2.line(img, (coordinate, y1), (outer, y1), rail_color, 2)
        cv2.line(img, (coordinate, y2), (outer, y2), rail_color, 2)
        opening = int(28 * detail_scale)
        cv2.rectangle(
            img,
            (coordinate - wall_thickness, center - opening),
            (coordinate + wall_thickness, center + opening),
            (255, 255, 255),
            -1
        )
        cv2.rectangle(
            mask,
            (coordinate - wall_thickness, center - opening),
            (coordinate + wall_thickness, center + opening),
            0,
            -1
        )
        bbox = [min(coordinate, outer), y1, max(coordinate, outer), y2]

    occupied_openings.setdefault(side, []).append((center - half_width, center + half_width))

    return {
        "type": "balcony",
        "side": side,
        "bbox": bbox,
    }


def draw_entrance(img, mask, edge, wall_thickness, detail_scale, occupied_openings):
    side = edge["side"]
    door_width = int(random.randint(72, 100) * detail_scale)
    porch_width = door_width + int(random.randint(34, 64) * detail_scale)
    porch_depth = int(random.randint(38, 68) * detail_scale)
    gap = max(wall_thickness * 2, int(24 * detail_scale))
    center_ranges = available_exterior_center_ranges(
        edge["start"],
        edge["end"],
        porch_width,
        occupied_openings.get(side, []),
        gap
    )
    if not center_ranges:
        return None

    center_start, center_end = random.choice(center_ranges)
    center = random.randint(center_start, center_end)
    porch_half = porch_width // 2
    door_half = door_width // 2
    porch_start, porch_end = center - porch_half, center + porch_half
    door_start, door_end = center - door_half, center + door_half
    coordinate = edge["coordinate"]
    porch_color = random.choice([
        (228, 230, 225),
        (220, 226, 228),
        (232, 224, 212),
    ])
    line_color = random.choice(INNER_WALL_COLORS)

    if side in ("top", "bottom"):
        inside_direction = 1 if side == "top" else -1
        outside = coordinate - inside_direction * porch_depth
        cv2.rectangle(
            img,
            (porch_start, min(coordinate, outside)),
            (porch_end, max(coordinate, outside)),
            porch_color,
            -1
        )
        cv2.rectangle(
            img,
            (porch_start, min(coordinate, outside)),
            (porch_end, max(coordinate, outside)),
            line_color,
            1
        )
        step_count = random.randint(2, 4)
        for step in range(1, step_count + 1):
            distance = round(porch_depth * step / (step_count + 1))
            step_y = outside + inside_direction * distance
            cv2.line(img, (porch_start, step_y), (porch_end, step_y), line_color, 1)

        cv2.rectangle(
            img,
            (door_start, coordinate - wall_thickness),
            (door_end, coordinate + wall_thickness),
            (255, 255, 255),
            -1
        )
        cv2.rectangle(
            mask,
            (door_start, coordinate - wall_thickness),
            (door_end, coordinate + wall_thickness),
            0,
            -1
        )
        hinge_x = door_start
        cv2.line(
            img,
            (hinge_x, coordinate),
            (hinge_x, coordinate + inside_direction * door_width),
            line_color,
            2
        )
        if side == "top":
            arc_start, arc_end = 0, 90
        else:
            arc_start, arc_end = 270, 360
        cv2.ellipse(
            img,
            (hinge_x, coordinate),
            (door_width, door_width),
            0,
            arc_start,
            arc_end,
            line_color,
            1
        )
        entrance_bbox = [door_start, coordinate - wall_thickness, door_end, coordinate + wall_thickness]
        porch_bbox = [porch_start, min(coordinate, outside), porch_end, max(coordinate, outside)]
    else:
        inside_direction = 1 if side == "left" else -1
        outside = coordinate - inside_direction * porch_depth
        cv2.rectangle(
            img,
            (min(coordinate, outside), porch_start),
            (max(coordinate, outside), porch_end),
            porch_color,
            -1
        )
        cv2.rectangle(
            img,
            (min(coordinate, outside), porch_start),
            (max(coordinate, outside), porch_end),
            line_color,
            1
        )
        step_count = random.randint(2, 4)
        for step in range(1, step_count + 1):
            distance = round(porch_depth * step / (step_count + 1))
            step_x = outside + inside_direction * distance
            cv2.line(img, (step_x, porch_start), (step_x, porch_end), line_color, 1)

        cv2.rectangle(
            img,
            (coordinate - wall_thickness, door_start),
            (coordinate + wall_thickness, door_end),
            (255, 255, 255),
            -1
        )
        cv2.rectangle(
            mask,
            (coordinate - wall_thickness, door_start),
            (coordinate + wall_thickness, door_end),
            0,
            -1
        )
        hinge_y = door_start
        cv2.line(
            img,
            (coordinate, hinge_y),
            (coordinate + inside_direction * door_width, hinge_y),
            line_color,
            2
        )
        if side == "left":
            arc_start, arc_end = 270, 360
        else:
            arc_start, arc_end = 180, 270
        cv2.ellipse(
            img,
            (coordinate, hinge_y),
            (door_width, door_width),
            0,
            arc_start,
            arc_end,
            line_color,
            1
        )
        entrance_bbox = [coordinate - wall_thickness, door_start, coordinate + wall_thickness, door_end]
        porch_bbox = [min(coordinate, outside), porch_start, max(coordinate, outside), porch_end]

    occupied_openings.setdefault(side, []).append((porch_start, porch_end))
    return (
        {"type": "entrance", "side": side, "bbox": entrance_bbox},
        {"type": "porch", "side": side, "bbox": porch_bbox},
    )


def draw_ladder(img, room, room_id, apartment_id, detail_scale=1.0):
    x1, y1, x2, y2 = room
    ladder_width = min(int(36 * detail_scale), (x2 - x1) // 4)
    ladder_height = min(int(100 * detail_scale), (y2 - y1) // 2)
    if ladder_width < 16 or ladder_height < 40:
        return None

    right = x2 - int(18 * detail_scale)
    left = right - ladder_width
    top = y1 + int(18 * detail_scale)
    bottom = top + ladder_height
    color = random.choice(INNER_WALL_COLORS)
    cv2.line(img, (left, top), (left, bottom), color, 2)
    cv2.line(img, (right, top), (right, bottom), color, 2)
    rung_count = max(4, ladder_height // max(1, int(14 * detail_scale)))
    for rung in range(rung_count + 1):
        y = top + round(ladder_height * rung / rung_count)
        cv2.line(img, (left, y), (right, y), color, 1)

    return {
        "type": "ladder",
        "room_id": room_id,
        "apartment_id": apartment_id,
        "bbox": [left, top, right, bottom],
    }


def draw_staircase(img, room, room_id, apartment_id, detail_scale=1.0):
    x1, y1, x2, y2 = room
    room_width, room_height = x2 - x1, y2 - y1
    orientation = random.choice(["horizontal", "vertical"])
    stair_width = int(room_width * random.uniform(0.30, 0.52))
    stair_height = int(room_height * random.uniform(0.30, 0.52))
    if min(stair_width, stair_height) < int(36 * detail_scale):
        return None

    left = x1 + (room_width - stair_width) // 2
    top = y1 + (room_height - stair_height) // 2
    right, bottom = left + stair_width, top + stair_height
    color = random.choice(INNER_WALL_COLORS)
    cv2.rectangle(img, (left, top), (right, bottom), color, 1)
    tread_count = random.randint(5, 9)

    if orientation == "horizontal":
        for tread in range(1, tread_count):
            y = top + round(stair_height * tread / tread_count)
            cv2.line(img, (left, y), (right, y), color, 1)
        center_y = (top + bottom) // 2
        cv2.arrowedLine(
            img,
            (left + 8, center_y),
            (right - 8, center_y),
            color,
            1,
            tipLength=0.18
        )
    else:
        for tread in range(1, tread_count):
            x = left + round(stair_width * tread / tread_count)
            cv2.line(img, (x, top), (x, bottom), color, 1)
        center_x = (left + right) // 2
        cv2.arrowedLine(
            img,
            (center_x, top + 8),
            (center_x, bottom - 8),
            color,
            1,
            tipLength=0.18
        )

    return {
        "type": "staircase",
        "room_id": room_id,
        "apartment_id": apartment_id,
        "orientation": orientation,
        "treads": tread_count,
        "bbox": [left, top, right, bottom],
    }


def draw_common_core(img, core, detail_scale=1.0):
    x1, y1, x2, y2 = core["bbox"]
    stair = draw_staircase(
        img,
        core["bbox"],
        core["id"],
        None,
        detail_scale
    )
    if stair:
        stair.pop("room_id", None)
        stair.pop("apartment_id", None)
        stair["common_core_id"] = core["id"]

    lift_bbox = None
    lift_size = int(min(x2 - x1, y2 - y1) * 0.22)
    if lift_size >= int(24 * detail_scale) and random.random() < 0.65:
        margin = int(10 * detail_scale)
        lift_x2 = x2 - margin
        lift_x1 = lift_x2 - lift_size
        lift_y1 = y1 + margin
        lift_y2 = lift_y1 + lift_size
        color = random.choice(INNER_WALL_COLORS)
        cv2.rectangle(img, (lift_x1, lift_y1), (lift_x2, lift_y2), color, 1)
        cv2.line(img, (lift_x1, lift_y1), (lift_x2, lift_y2), color, 1)
        cv2.line(img, (lift_x2, lift_y1), (lift_x1, lift_y2), color, 1)
        lift_bbox = [lift_x1, lift_y1, lift_x2, lift_y2]

    core_annotation = {
        "type": "stairwell_core",
        "core_id": core["id"],
        "apartment_ids": core["apartment_ids"],
        "bbox": [x1, y1, x2, y2],
        "lift_bbox": lift_bbox,
    }
    return core_annotation, stair


# ============================================================
# PAPER TEXTURE
# ============================================================

def add_paper_texture(img):
    h, w = img.shape[:2]

    sigma = random.uniform(
        2,
        8
    )

    noise = np.random.normal(
        0,
        sigma,
        (
            h,
            w,
            1
        )
    )

    textured = (
        img.astype(
            np.float32
        )
        + noise
    )

    gx = np.linspace(
        random.uniform(
            -10,
            0
        ),
        random.uniform(
            0,
            10
        ),
        w
    )

    gy = np.linspace(
        random.uniform(
            -10,
            0
        ),
        random.uniform(
            0,
            10
        ),
        h
    )

    gradient = (
        gy[:, None]
        + gx[None, :]
    )[..., None]

    textured += gradient

    return np.clip(
        textured,
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# NOISE
# ============================================================

def add_gaussian_noise(
    img,
    sigma=None
):
    if sigma is None:

        sigma = random.uniform(
            3,
            22
        )

    noise = np.random.normal(
        0,
        sigma,
        img.shape
    )

    result = (
        img.astype(
            np.float32
        )
        + noise
    )

    return np.clip(
        result,
        0,
        255
    ).astype(
        np.uint8
    )


def add_salt_pepper_noise(
    img,
    amount=None
):
    if amount is None:

        amount = random.uniform(
            0.0005,
            0.012
        )

    result = img.copy()

    h, w = result.shape[:2]

    number = int(
        h
        * w
        * amount
    )

    ys = np.random.randint(
        0,
        h,
        number
    )

    xs = np.random.randint(
        0,
        w,
        number
    )

    result[
        ys,
        xs
    ] = 255

    ys = np.random.randint(
        0,
        h,
        number
    )

    xs = np.random.randint(
        0,
        w,
        number
    )

    result[
        ys,
        xs
    ] = 0

    return result


def add_speckle_noise(img):
    noise = np.random.normal(
        0,
        random.uniform(
            0.01,
            0.08
        ),
        img.shape
    )

    img_float = img.astype(
        np.float32
    )

    result = (
        img_float
        + img_float * noise
    )

    return np.clip(
        result,
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# SCANNER ARTIFACTS
# ============================================================

def add_scan_lines(img):
    result = img.copy()

    h, w = result.shape[:2]

    num_lines = random.randint(
        5,
        35
    )

    for _ in range(
        num_lines
    ):

        y = random.randint(
            0,
            h - 1
        )

        thickness = random.choice([
            1,
            1,
            1,
            2,
            3
        ])

        delta = random.randint(
            -40,
            -5
        )

        y1 = max(
            0,
            y - thickness
        )

        y2 = min(
            h,
            y + thickness + 1
        )

        roi = result[
            y1:y2
        ].astype(
            np.int16
        )

        roi += delta

        result[
            y1:y2
        ] = np.clip(
            roi,
            0,
            255
        ).astype(
            np.uint8
        )

    return result


def add_vertical_scan_noise(img):
    result = img.copy()

    h, w = result.shape[:2]

    num_lines = random.randint(
        1,
        12
    )

    for _ in range(
        num_lines
    ):

        x = random.randint(
            0,
            w - 1
        )

        width = random.randint(
            1,
            4
        )

        delta = random.randint(
            -30,
            20
        )

        x1 = max(
            0,
            x - width
        )

        x2 = min(
            w,
            x + width + 1
        )

        roi = result[
            :,
            x1:x2
        ].astype(
            np.int16
        )

        roi += delta

        result[
            :,
            x1:x2
        ] = np.clip(
            roi,
            0,
            255
        ).astype(
            np.uint8
        )

    return result


# ============================================================
# STAINS
# ============================================================

def add_stains(img):
    result = img.copy()
    overlay = result.copy()

    h, w = result.shape[:2]

    number = random.randint(
        2,
        15
    )

    for _ in range(
        number
    ):

        cx = random.randint(
            0,
            w - 1
        )

        cy = random.randint(
            0,
            h - 1
        )

        radius_x = random.randint(
            15,
            150
        )

        radius_y = random.randint(
            10,
            100
        )

        color = random.choice([
            (210, 220, 225),
            (220, 225, 230),
            (190, 200, 210),
            (225, 225, 215),
            (200, 210, 215),
        ])

        cv2.ellipse(
            overlay,
            (
                cx,
                cy
            ),
            (
                radius_x,
                radius_y
            ),
            random.randint(
                0,
                180
            ),
            0,
            360,
            color,
            -1
        )

    alpha = random.uniform(
        0.03,
        0.18
    )

    return cv2.addWeighted(
        overlay,
        alpha,
        result,
        1.0 - alpha,
        0
    )


# ============================================================
# SCRATCHES
# ============================================================

def add_random_scratches(img):
    result = img.copy()

    h, w = result.shape[:2]

    count = random.randint(
        1,
        12
    )

    for _ in range(
        count
    ):

        x1 = random.randint(
            0,
            w - 1
        )

        y1 = random.randint(
            0,
            h - 1
        )

        length = random.randint(
            20,
            300
        )

        angle = random.uniform(
            0,
            2 * math.pi
        )

        x2 = int(
            x1
            + math.cos(angle)
            * length
        )

        y2 = int(
            y1
            + math.sin(angle)
            * length
        )

        color = random.choice([
            (180, 180, 180),
            (210, 210, 210),
            (120, 120, 120),
            (235, 235, 235),
        ])

        cv2.line(
            result,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            color,
            random.choice([
                1,
                1,
                2
            ])
        )

    return result


# ============================================================
# BRIGHTNESS / CONTRAST
# ============================================================

def random_brightness_contrast(img):
    alpha = random.uniform(
        0.65,
        1.35
    )

    beta = random.uniform(
        -30,
        35
    )

    result = (
        img.astype(
            np.float32
        )
        * alpha
        + beta
    )

    return np.clip(
        result,
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# BLUR
# ============================================================

def add_motion_blur(img):
    size = random.choice([
        3,
        5,
        7,
        9
    ])

    kernel = np.zeros(
        (
            size,
            size
        ),
        dtype=np.float32
    )

    mode = random.randint(
        0,
        3
    )

    if mode == 0:
        kernel[
            size // 2,
            :
        ] = 1

    elif mode == 1:
        kernel[
            :,
            size // 2
        ] = 1

    elif mode == 2:

        for i in range(
            size
        ):
            kernel[
                i,
                i
            ] = 1

    else:

        for i in range(
            size
        ):
            kernel[
                i,
                size - 1 - i
            ] = 1

    kernel /= np.sum(
        kernel
    )

    return cv2.filter2D(
        img,
        -1,
        kernel
    )


def add_defocus_blur(img):
    k = random.choice([
        3,
        5,
        7,
        9
    ])

    return cv2.GaussianBlur(
        img,
        (
            k,
            k
        ),
        random.uniform(
            0.8,
            2.5
        )
    )


# ============================================================
# DOWNSCALE
# ============================================================

def random_downscale(img):
    """
    Low resolution simulation.

    Original:
        1400 x 1000

    Can temporarily become e.g.
        230 x 164

    and then be resized back.
    """

    h, w = img.shape[:2]

    scale = random.choice([
        random.uniform(
            0.12,
            0.25
        ),
        random.uniform(
            0.25,
            0.45
        ),
        random.uniform(
            0.45,
            0.70
        ),
    ])

    new_w = max(
        64,
        int(
            w * scale
        )
    )

    new_h = max(
        64,
        int(
            h * scale
        )
    )

    down_interpolation = random.choice([
        cv2.INTER_AREA,
        cv2.INTER_LINEAR,
        cv2.INTER_NEAREST,
    ])

    small = cv2.resize(
        img,
        (
            new_w,
            new_h
        ),
        interpolation=down_interpolation
    )

    up_interpolation = random.choice([
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_NEAREST,
    ])

    restored = cv2.resize(
        small,
        (
            w,
            h
        ),
        interpolation=up_interpolation
    )

    return restored, scale


# ============================================================
# ROTATION
# ============================================================

def random_rotation(img):
    h, w = img.shape[:2]

    angle = random.uniform(
        -4.0,
        4.0
    )

    matrix = cv2.getRotationMatrix2D(
        (
            w // 2,
            h // 2
        ),
        angle,
        1
    )

    result = cv2.warpAffine(
        img,
        matrix,
        (
            w,
            h
        ),
        borderValue=(
            245,
            245,
            245
        )
    )

    return result, angle, matrix


# ============================================================
# PERSPECTIVE
# ============================================================

def random_perspective(img):
    h, w = img.shape[:2]

    amount = random.uniform(
        0.005,
        0.035
    )

    dx = int(
        w * amount
    )

    dy = int(
        h * amount
    )

    src = np.float32([
        [
            0,
            0
        ],
        [
            w - 1,
            0
        ],
        [
            w - 1,
            h - 1
        ],
        [
            0,
            h - 1
        ]
    ])

    dst = np.float32([
        [
            random.randint(
                0,
                dx
            ),
            random.randint(
                0,
                dy
            )
        ],
        [
            w - 1 - random.randint(
                0,
                dx
            ),
            random.randint(
                0,
                dy
            )
        ],
        [
            w - 1 - random.randint(
                0,
                dx
            ),
            h - 1 - random.randint(
                0,
                dy
            )
        ],
        [
            random.randint(
                0,
                dx
            ),
            h - 1 - random.randint(
                0,
                dy
            )
        ],
    ])

    matrix = cv2.getPerspectiveTransform(
        src,
        dst
    )

    result = cv2.warpPerspective(
        img,
        matrix,
        (
            w,
            h
        ),
        borderValue=(
            245,
            245,
            245
        )
    )

    return result, matrix


# ============================================================
# PARTIAL FADING
# ============================================================

def add_partial_fading(img):
    result = img.copy()

    h, w = result.shape[:2]

    white = np.ones_like(
        result
    ) * 255

    count = random.randint(
        1,
        8
    )

    for _ in range(
        count
    ):

        x1 = random.randint(
            0,
            w - 1
        )

        y1 = random.randint(
            0,
            h - 1
        )

        x2 = min(
            w,
            x1 + random.randint(
                50,
                400
            )
        )

        y2 = min(
            h,
            y1 + random.randint(
                20,
                180
            )
        )

        alpha = random.uniform(
            0.05,
            0.35
        )

        roi = result[
            y1:y2,
            x1:x2
        ]

        white_roi = white[
            y1:y2,
            x1:x2
        ]

        result[
            y1:y2,
            x1:x2
        ] = cv2.addWeighted(
            roi,
            1.0 - alpha,
            white_roi,
            alpha,
            0
        )

    return result


# ============================================================
# COLOR MISREGISTRATION
# ============================================================

def random_channel_shift(img):
    b, g, r = cv2.split(
        img
    )

    h, w = img.shape[:2]

    max_shift = random.randint(
        1,
        4
    )

    def shift(channel):

        dx = random.randint(
            -max_shift,
            max_shift
        )

        dy = random.randint(
            -max_shift,
            max_shift
        )

        matrix = np.float32([
            [
                1,
                0,
                dx
            ],
            [
                0,
                1,
                dy
            ]
        ])

        return cv2.warpAffine(
            channel,
            matrix,
            (
                w,
                h
            ),
            borderValue=255
        )

    return cv2.merge([
        shift(b),
        shift(g),
        shift(r)
    ])


# ============================================================
# JPEG
# ============================================================

def add_jpeg_artifacts(
    img,
    quality=None
):
    if quality is None:

        quality = random.randint(
            15,
            75
        )

    success, encoded = cv2.imencode(
        ".jpg",
        img,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            quality
        ]
    )

    if not success:
        return img

    return cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR
    )


# ============================================================
# RANDOM OCCLUSIONS
# ============================================================

def add_random_occlusions(img):
    result = img.copy()

    h, w = result.shape[:2]

    count = random.randint(
        1,
        5
    )

    for _ in range(
        count
    ):

        x1 = random.randint(
            0,
            w - 20
        )

        y1 = random.randint(
            0,
            h - 20
        )

        x2 = min(
            w - 1,
            x1 + random.randint(
                10,
                180
            )
        )

        y2 = min(
            h - 1,
            y1 + random.randint(
                5,
                80
            )
        )

        color = random.choice([
            (255, 255, 255),
            (235, 235, 235),
            (225, 225, 225),
        ])

        cv2.rectangle(
            result,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            color,
            -1
        )

    return result


# ============================================================
# DEGRADATION PIPELINE
# ============================================================

def degrade_image(img):
    output = img.copy()

    info = {
        "paper_texture": False,
        "brightness_contrast": False,
        "stains": False,
        "scan_lines": False,
        "vertical_scan_noise": False,
        "scratches": False,
        "partial_fading": False,
        "gaussian_noise": False,
        "salt_pepper": False,
        "speckle_noise": False,
        "rotation": None,
        "rotation_matrix": None,
        "perspective": False,
        "perspective_matrix": None,
        "downscale": None,
        "defocus_blur": False,
        "motion_blur": False,
        "channel_shift": False,
        "occlusions": False,
        "jpeg_quality": None,
    }

    # ------------------------------------------
    # PAPER
    # ------------------------------------------

    if random.random() < 0.85:

        output = add_paper_texture(
            output
        )

        info[
            "paper_texture"
        ] = True

    # ------------------------------------------
    # BRIGHTNESS / CONTRAST
    # ------------------------------------------

    if random.random() < 0.50:

        output = random_brightness_contrast(
            output
        )

        info[
            "brightness_contrast"
        ] = True

    # ------------------------------------------
    # STAINS
    # ------------------------------------------

    if random.random() < 0.35:

        output = add_stains(
            output
        )

        info[
            "stains"
        ] = True

    # ------------------------------------------
    # SCAN LINES
    # ------------------------------------------

    if random.random() < 0.30:

        output = add_scan_lines(
            output
        )

        info[
            "scan_lines"
        ] = True

    if random.random() < 0.15:

        output = add_vertical_scan_noise(
            output
        )

        info[
            "vertical_scan_noise"
        ] = True

    # ------------------------------------------
    # SCRATCHES
    # ------------------------------------------

    if random.random() < 0.25:

        output = add_random_scratches(
            output
        )

        info[
            "scratches"
        ] = True

    # ------------------------------------------
    # FADING
    # ------------------------------------------

    if random.random() < 0.30:

        output = add_partial_fading(
            output
        )

        info[
            "partial_fading"
        ] = True

    # ------------------------------------------
    # NOISE
    # ------------------------------------------

    if random.random() < 0.40:

        output = add_gaussian_noise(
            output
        )

        info[
            "gaussian_noise"
        ] = True

    if random.random() < 0.18:

        output = add_salt_pepper_noise(
            output
        )

        info[
            "salt_pepper"
        ] = True

    if random.random() < 0.20:

        output = add_speckle_noise(
            output
        )

        info[
            "speckle_noise"
        ] = True

    # ------------------------------------------
    # GEOMETRIC
    # ------------------------------------------

    if random.random() < 0.45:

        output, angle, matrix = random_rotation(
            output
        )

        info[
            "rotation"
        ] = angle
        info[
            "rotation_matrix"
        ] = matrix.tolist()

    if random.random() < 0.20:

        output, matrix = random_perspective(
            output
        )

        info[
            "perspective"
        ] = True
        info[
            "perspective_matrix"
        ] = matrix.tolist()

    # ------------------------------------------
    # LOW RESOLUTION
    # ------------------------------------------

    if random.random() < 0.65:

        output, scale = random_downscale(
            output
        )

        info[
            "downscale"
        ] = scale

    # ------------------------------------------
    # BLUR
    # ------------------------------------------

    if random.random() < 0.30:

        output = add_defocus_blur(
            output
        )

        info[
            "defocus_blur"
        ] = True

    if random.random() < 0.12:

        output = add_motion_blur(
            output
        )

        info[
            "motion_blur"
        ] = True

    # ------------------------------------------
    # COLOR SHIFT
    # ------------------------------------------

    if random.random() < 0.12:

        output = random_channel_shift(
            output
        )

        info[
            "channel_shift"
        ] = True

    # ------------------------------------------
    # OCCLUSIONS
    # ------------------------------------------

    if random.random() < 0.10:

        output = add_random_occlusions(
            output
        )

        info[
            "occlusions"
        ] = True

    # ------------------------------------------
    # JPEG
    # ------------------------------------------

    if random.random() < 0.75:

        quality = random.randint(
            15,
            75
        )

        output = add_jpeg_artifacts(
            output,
            quality
        )

        info[
            "jpeg_quality"
        ] = quality

    # ------------------------------------------
    # EXTREME:
    # DOWN SCALE -> JPEG -> UPSCALE
    # ------------------------------------------

    if random.random() < 0.20:

        output, scale = random_downscale(
            output
        )

        quality = random.randint(
            10,
            35
        )

        output = add_jpeg_artifacts(
            output,
            quality
        )

        info[
            "downscale"
        ] = scale

        info[
            "jpeg_quality"
        ] = quality

    return output, info


# ============================================================
# FLOORPLAN GENERATOR
# ============================================================

def generate_floorplan():

    scale_profile = random.choice(
        DRAWING_SCALE_PROFILES
    )
    canvas_width = scale_profile["width"]
    canvas_height = scale_profile["height"]
    drawing_scale = scale_profile["label"]
    scale_denominator = int(drawing_scale.split(":")[1])
    pixels_per_meter = 72.0 / (scale_denominator * 0.0254)
    detail_scale = min(
        canvas_width / IMG_W,
        canvas_height / IMG_H
    )

    # Slightly non-white background
    base = random.randint(
        245,
        255
    )

    img = np.ones(
        (
            canvas_height,
            canvas_width,
            3
        ),
        dtype=np.uint8
    ) * base

    wall_mask = np.zeros(
        (
            canvas_height,
            canvas_width
        ),
        dtype=np.uint8
    )

    footprint_fraction = scale_profile["footprint_fraction"]
    margin_x = int(
        canvas_width
        * (1.0 - footprint_fraction)
        * random.uniform(0.42, 0.58)
    )
    margin_y = int(
        canvas_height
        * (1.0 - footprint_fraction)
        * random.uniform(0.42, 0.58)
    )

    building = (
        margin_x,
        margin_y,
        canvas_width - margin_x,
        canvas_height - margin_y
    )

    wall_thickness = int(random.choice([6, 8, 10, 12, 14]) * detail_scale)

    outer_wall_thickness = int(random.choice([12, 16, 20, 24, 28]) * detail_scale)

    has_shared_access_hall = random.random() < SHARED_ACCESS_HALL_PERCENTAGE / 100
    footprint_styles = ["rectangular", "chamfered", "angled"]
    if not has_shared_access_hall:
        footprint_styles.extend(["L-shaped", "T-shaped"])
    footprint_style = random.choice(footprint_styles)

    footprint, building_regions = create_building_geometry(
        building,
        footprint_style
    )

    footprint_points = np.asarray(
        footprint,
        dtype=np.int32
    )

    footprint_mask = np.zeros(
        (canvas_height, canvas_width),
        dtype=np.uint8
    )

    cv2.fillPoly(
        footprint_mask,
        [footprint_points],
        (255,)
    )

    target_apartment_count = random.randint(MIN_APARTMENTS, MAX_APARTMENTS)
    shared_access_hall = None
    common_core_zones = []
    if has_shared_access_hall:
        shared_access_hall, apartment_zones = create_shared_access_layout(
            building,
            target_apartment_count
        )
    else:
        desired_core_count = random.choices(
            [0, 1, 2],
            weights=[0.62, 0.30, 0.08]
        )[0]
        apartment_zones = create_apartment_zones(
            building_regions,
            target_apartment_count + desired_core_count
        )
        common_core_zones, apartment_zones = extract_common_core_zones(
            apartment_zones,
            desired_core_count,
            int(120 * detail_scale)
        )
    apartment_count = len(apartment_zones)

    rooms = []
    room_apartment_ids = []
    for apartment in apartment_zones:
        apartment_rooms = split_rectangle(
            apartment["bbox"],
            max_depth=random.choice([1, 2, 2, 3, 3])
        )
        rooms.extend(apartment_rooms)
        room_apartment_ids.extend([apartment["id"]] * len(apartment_rooms))

    rooms = rooms[:MAX_ROOMS]
    room_apartment_ids = room_apartment_ids[:MAX_ROOMS]
    interior_wall_segments = calculate_internal_wall_segments(
        rooms
        + ([shared_access_hall] if shared_access_hall else [])
        + [core["bbox"] for core in common_core_zones],
        pixels_per_meter
    )
    interior_wall_length_px = sum(
        segment["length_px"]
        for segment in interior_wall_segments
    )
    building_area_px2 = polygon_area(footprint)

    # ------------------------------------------
    # ROOM COLOR FILLS
    # ------------------------------------------

    img = fill_rooms(
        img,
        rooms + ([shared_access_hall] if shared_access_hall else []),
        footprint_mask,
        wall_thickness
    )

    # ------------------------------------------
    # INTERNAL WALLS
    # ------------------------------------------

    inner_wall_color = random.choice(
        INNER_WALL_COLORS
    )

    inner_wall_mask = np.zeros_like(
        wall_mask
    )

    interior_layer = img.copy()

    for room in rooms:

        x1, y1, x2, y2 = room

        cv2.rectangle(
            img,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            inner_wall_color,
            wall_thickness
        )

        cv2.rectangle(
            inner_wall_mask,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            255,
            wall_thickness
        )

    if shared_access_hall:
        hx1, hy1, hx2, hy2 = shared_access_hall
        cv2.rectangle(
            inner_wall_mask,
            (hx1, hy1),
            (hx2, hy2),
            255,
            wall_thickness
        )

    inner_wall_mask = cv2.bitwise_and(
        inner_wall_mask,
        footprint_mask
    )

    img[footprint_mask > 0] = interior_layer[footprint_mask > 0]
    inner_wall_layer = np.zeros_like(img)
    inner_wall_layer[:] = inner_wall_color
    cv2.copyTo(inner_wall_layer, inner_wall_mask, img)

    # ------------------------------------------
    # OUTER WALL
    # ------------------------------------------

    bx1, by1, bx2, by2 = building

    outer_wall_color = random.choice(
        OUTER_WALL_COLORS
    )

    cv2.polylines(
        img,
        [footprint_points],
        True,
        outer_wall_color,
        outer_wall_thickness,
        cv2.LINE_AA
    )

    outer_wall_mask = np.zeros_like(
        wall_mask
    )

    cv2.polylines(
        outer_wall_mask,
        [footprint_points],
        True,
        (255,),
        outer_wall_thickness,
        cv2.LINE_8
    )

    exterior_wall_faces = calculate_exterior_wall_face_lengths(
        outer_wall_mask,
        pixels_per_meter
    )

    wall_mask = cv2.bitwise_or(
        inner_wall_mask,
        outer_wall_mask
    )

    inner_wall_style = random.choice([
        "solid",
        "diagonal",
        "crosshatch",
        "brick",
        "stipple",
        "outlined_hatch"
    ])

    outer_wall_style = random.choice([
        "solid",
        "diagonal",
        "crosshatch",
        "brick",
        "stipple",
        "outlined_hatch"
    ])

    apply_wall_pattern(
        img,
        inner_wall_mask,
        inner_wall_color,
        inner_wall_style
    )

    apply_wall_pattern(
        img,
        outer_wall_mask,
        outer_wall_color,
        outer_wall_style
    )

    annotations = {
        "building_bbox": list(
            building
        ),
        "building_outline": [
            list(point)
            for point in footprint
        ],
        "footprint_style": footprint_style,
        "wall_style": {
            "interior": inner_wall_style,
            "exterior": outer_wall_style,
            "interior_color_bgr": list(inner_wall_color),
            "exterior_color_bgr": list(outer_wall_color),
            "interior_thickness": wall_thickness,
            "exterior_thickness": outer_wall_thickness
        },
        "drawing_scale": {
            "label": drawing_scale,
            "canvas_width": canvas_width,
            "canvas_height": canvas_height,
            "pixels_per_meter": round(pixels_per_meter, 3),
            "measurement_dpi": 72,
            "footprint_occupancy": round(
                (bx2 - bx1) * (by2 - by1)
                / (canvas_width * canvas_height),
                3
            )
        },
        "building_area_px2": round(building_area_px2, 2),
        "building_area_m2": round(
            building_area_px2 / (pixels_per_meter ** 2),
            2
        ),
        "interior_wall_length_px": round(interior_wall_length_px, 2),
        "interior_wall_length_m": round(
            interior_wall_length_px / pixels_per_meter,
            2
        ),
        "interior_wall_segments": interior_wall_segments,
        "exterior_wall_lengths": exterior_wall_faces,
        "apartment_count": apartment_count,
        "apartments": [
            {
                "apartment_id": apartment["id"],
                "bbox": list(apartment["bbox"]),
                "room_ids": [
                    room_id
                    for room_id, apartment_id in enumerate(room_apartment_ids)
                    if apartment_id == apartment["id"]
                ]
            }
            for apartment in apartment_zones
        ],
        "common_cores": [],
        "shared_access_halls": [],
        "rooms": [],
        "doors": [],
        "core_doors": [],
        "access_doors": [],
        "entrances": [],
        "windows": [],
        "balconies": [],
        "porches": [],
        "ladders": [],
        "stairs": [],
        "degradation": {}
    }

    if shared_access_hall:
        hall_x1, hall_y1, hall_x2, hall_y2 = shared_access_hall
        hall_orientation = (
            "horizontal"
            if hall_x2 - hall_x1 > hall_y2 - hall_y1
            else "vertical"
        )
        annotations["shared_access_halls"].append({
            "hall_id": 0,
            "type": "shared_access_hall",
            "bbox": list(shared_access_hall),
            "orientation": hall_orientation,
            "apartment_ids": [apartment["id"] for apartment in apartment_zones],
            "area_m2": round(
                (hall_x2 - hall_x1) * (hall_y2 - hall_y1)
                / (pixels_per_meter ** 2),
                2
            ),
        })

    for core in common_core_zones:
        core_annotation, staircase = draw_common_core(
            img,
            core,
            detail_scale
        )
        annotations["common_cores"].append(core_annotation)
        if staircase:
            annotations["stairs"].append(staircase)

    # ------------------------------------------
    # DOORS
    # ------------------------------------------

    annotations["doors"] = draw_room_connection_doors(
        img,
        wall_mask,
        rooms,
        room_apartment_ids,
        wall_thickness,
        detail_scale
    )

    if shared_access_hall:
        annotations["access_doors"] = draw_shared_access_doors(
            img,
            wall_mask,
            shared_access_hall,
            rooms,
            room_apartment_ids,
            [apartment["id"] for apartment in apartment_zones],
            wall_thickness,
            detail_scale
        )

    for core in common_core_zones:
        for apartment_id in core["apartment_ids"]:
            shared_walls = []
            for room_id, room in enumerate(rooms):
                if room_apartment_ids[room_id] != apartment_id:
                    continue
                wall = shared_wall(core["bbox"], room, tolerance=0)
                if wall:
                    segment_length = wall[3] - wall[2]
                    shared_walls.append((segment_length, room_id, wall))

            if not shared_walls:
                continue

            segment_length, room_id, wall = max(shared_walls, key=lambda item: item[0])
            door_width = min(
                int(64 * detail_scale),
                segment_length - 31
            )
            if door_width < int(30 * detail_scale):
                continue

            door = draw_door(
                img,
                wall_mask,
                wall,
                door_width,
                wall_thickness
            )
            if door:
                door["type"] = "core_door"
                door["common_core_id"] = core["id"]
                door["apartment_id"] = apartment_id
                door["room_id"] = room_id
                annotations["core_doors"].append(door)

    # ------------------------------------------
    # BALCONIES
    # ------------------------------------------

    exterior_edges = get_exterior_edges(footprint)
    occupied_exterior = {
        side: []
        for side in ("top", "bottom", "left", "right")
    }
    if exterior_edges:
        entrance_count = random.choices(
            [1, 2, 3],
            weights=[0.58, 0.32, 0.10]
        )[0]
        for _ in range(entrance_count):
            for edge in random.sample(exterior_edges, len(exterior_edges)):
                result = draw_entrance(
                    img,
                    wall_mask,
                    edge,
                    outer_wall_thickness,
                    detail_scale,
                    occupied_exterior
                )
                if result:
                    entrance, porch = result
                    annotations["entrances"].append(entrance)
                    annotations["porches"].append(porch)
                    break

        for _ in range(random.randint(2, min(10, apartment_count))):
            balcony = draw_balcony(
                img,
                wall_mask,
                random.choice(exterior_edges),
                outer_wall_thickness,
                detail_scale,
                occupied_exterior
            )
            if balcony:
                annotations["balconies"].append(balcony)

    # ------------------------------------------
    # WINDOWS
    # ------------------------------------------

    num_windows = random.randint(
        MIN_WINDOWS,
        MAX_WINDOWS
    )

    for _ in range(
        num_windows
    ):
        window = draw_window(
            img,
            wall_mask,
            building,
            outer_wall_thickness,
            footprint,
            detail_scale,
            occupied_exterior
        )

        if window:
            annotations[
                "windows"
            ].append(
                window
            )

    # ------------------------------------------
    # LADDERS
    # ------------------------------------------

    stair_candidates = sorted(
        range(len(rooms)),
        key=lambda room_id: (
            (rooms[room_id][2] - rooms[room_id][0])
            * (rooms[room_id][3] - rooms[room_id][1])
        ),
        reverse=True
    )
    stair_count = random.choice([0, 1, 1, 2, 2, 3])
    stair_pool = stair_candidates[:max(stair_count, round(len(stair_candidates) * 0.35))]
    for room_id in random.sample(stair_pool, min(stair_count, len(stair_pool))):
        staircase = draw_staircase(
            img,
            rooms[room_id],
            room_id,
            room_apartment_ids[room_id],
            detail_scale
        )
        if staircase:
            annotations["stairs"].append(staircase)

    ladder_count = random.randint(1, min(3, len(rooms)))
    for room_id in random.sample(range(len(rooms)), ladder_count):
        ladder = draw_ladder(
            img,
            rooms[room_id],
            room_id,
            room_apartment_ids[room_id],
            detail_scale
        )
        if ladder:
            annotations["ladders"].append(ladder)

    # ------------------------------------------
    # ROOM LABELS
    # ------------------------------------------

    measurement_count = min(
        max(1, round(len(rooms) * 0.25)),
        max(1, len(rooms) - 1)
    )
    measurement_candidates = sorted(
        range(len(rooms)),
        key=lambda room_id: (
            (rooms[room_id][2] - rooms[room_id][0])
            * (rooms[room_id][3] - rooms[room_id][1])
        ),
        reverse=True
    )
    candidate_pool_size = max(
        measurement_count,
        round(len(measurement_candidates) * 0.55)
    )
    measurement_room_ids = set(random.sample(
        measurement_candidates[:candidate_pool_size],
        measurement_count
    ))

    room_name_probability = random.uniform(0.40, 0.85)
    visible_room_name_count = min(
        max(1, round(len(rooms) * room_name_probability)),
        max(1, len(rooms) - 1)
    )
    visible_room_ids = set(random.sample(
        range(len(rooms)),
        visible_room_name_count
    ))
    labels = draw_room_labels(
        img,
        rooms,
        room_apartment_ids,
        detail_scale,
        pixels_per_meter,
        footprint_mask,
        wall_thickness,
        measurement_room_ids,
        room_name_probability,
        visible_room_ids
    )

    annotations[
        "rooms"
    ] = labels
    annotations["room_name_probability"] = round(room_name_probability, 2)
    annotations["rooms_with_displayed_names"] = len(visible_room_ids)
    annotations["rooms_with_displayed_measurements"] = len(
        measurement_room_ids
    )
    annotations["total_room_area_m2"] = round(
        sum(room["measurements"]["area_m2"] for room in labels),
        2
    )

    # ------------------------------------------
    # GLOBAL DIMENSIONS
    # ------------------------------------------

    if random.random() < 0.90:

        draw_dimension(
            img,
            (
                bx1,
                by1
            ),
            (
                bx2,
                by1
            ),
            offset=-45,
            pixels_per_meter=pixels_per_meter
        )

        draw_dimension(
            img,
            (
                bx1,
                by1
            ),
            (
                bx1,
                by2
            ),
            offset=-55,
            pixels_per_meter=pixels_per_meter
        )

    # ------------------------------------------
    # ROOM DIMENSIONS
    # ------------------------------------------

    sample_count = min(
        len(rooms),
        random.randint(
            1,
            5
        )
    )

    if sample_count > 0:

        for room in random.sample(
            rooms,
            sample_count
        ):

            x1, y1, x2, y2 = room

            if random.random() < 0.50:

                draw_dimension(
                    img,
                    (
                        x1,
                        y2
                    ),
                    (
                        x2,
                        y2
                    ),
                    offset=-20,
                    pixels_per_meter=pixels_per_meter
                )

    # ------------------------------------------
    # TECHNICAL TEXT
    # ------------------------------------------

    if random.random() < 0.85:

        texts = [
            "GROUND FLOOR PLAN",
            "FIRST FLOOR PLAN",
            f"SCALE {drawing_scale}",
            "A-101",
            "A-102",
            "PROJECT No. 024",
            "ARCHITECTURAL PLAN",
            "REV. 01",
            "NOT FOR CONSTRUCTION"
        ]

        random.shuffle(
            texts
        )

        y = 40

        count = random.randint(
            1,
            4
        )

        for text in texts[:count]:

            cv2.putText(
                img,
                text,
                (
                    40,
                    y
                ),
                FONT,
                random.uniform(
                    0.40,
                    0.70
                ),
                random.choice(
                    TEXT_COLORS
                ),
                random.choice([
                    1,
                    1,
                    2
                ]),
                cv2.LINE_AA
            )

            y += random.randint(
                25,
                35
            )

    # ------------------------------------------
    # CONSTRUCTION LINES
    # ------------------------------------------

    for _ in range(
        random.randint(
            0,
            12
        )
    ):

        color = random.choice([
            (180, 180, 180),
            (195, 195, 195),
            (210, 210, 210),
            (160, 160, 160)
        ])

        if random.random() < 0.50:

            y = random.randint(
                by1,
                by2
            )

            cv2.line(
                img,
                (
                    bx1 - 30,
                    y
                ),
                (
                    bx2 + 30,
                    y
                ),
                color,
                1
            )

        else:

            x = random.randint(
                bx1,
                bx2
            )

            cv2.line(
                img,
                (
                    x,
                    by1 - 30
                ),
                (
                    x,
                    by2 + 30
                ),
                color,
                1
            )

    cv2.putText(
        img,
        f"SCALE {drawing_scale}",
        (int(40 * detail_scale), canvas_height - int(24 * detail_scale)),
        FONT,
        0.45 * detail_scale,
        random.choice(TEXT_COLORS),
        max(1, int(detail_scale)),
        cv2.LINE_AA
    )

    return (
        img,
        wall_mask,
        annotations
    )


# ============================================================
# DATASET CREATION
# ============================================================

def create_dataset():

    images_dir = (
        OUT_DIR
        / "images"
    )

    clean_dir = (
        OUT_DIR
        / "clean"
    )

    masks_dir = (
        OUT_DIR
        / "wall_masks"
    )

    annotations_dir = (
        OUT_DIR
        / "annotations"
    )

    for directory in [
        images_dir,
        clean_dir,
        masks_dir,
        annotations_dir
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    for i in range(
        NUM_IMAGES
    ):

        clean_img, mask, annotations = (
            generate_floorplan()
        )

        degraded_img, degradation_info = (
            degrade_image(
                clean_img
            )
        )

        annotations[
            "degradation"
        ] = degradation_info

        filename = (
            f"floorplan_{i:05d}"
        )

        # --------------------------------------
        # CLEAN
        # --------------------------------------

        cv2.imwrite(
            str(
                clean_dir
                / f"{filename}.png"
            ),
            clean_img
        )

        # --------------------------------------
        # DEGRADED
        # --------------------------------------

        cv2.imwrite(
            str(
                images_dir
                / f"{filename}.jpg"
            ),
            degraded_img,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

        # --------------------------------------
        # WALL MASK
        # --------------------------------------

        cv2.imwrite(
            str(
                masks_dir
                / f"{filename}.png"
            ),
            mask
        )

        # --------------------------------------
        # JSON
        # --------------------------------------

        with open(
            annotations_dir
            / f"{filename}.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                annotations,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"[{i + 1:04d}/{NUM_IMAGES:04d}] "
            f"{filename} "
            f"| rooms={len(annotations['rooms'])} "
            f"| doors={len(annotations['doors'])} "
            f"| windows={len(annotations['windows'])}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    create_dataset()