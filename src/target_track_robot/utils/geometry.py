"""Pure geometry functions (pure functions) operating on bounding boxes.

No dependencies on OpenCV or hardware — designed to be fast and easily unit-testable.

Box representation used throughout the project: (x, y, w, h)
where (x, y) is the top-left corner and w, h are width and height.
"""

from __future__ import annotations

from typing import NamedTuple


class Box(NamedTuple):
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)

def box_height_ratio(box: tuple[int, int, int, int], frame_height: int) -> float:
    """Ratio of box height to frame height — used as a proxy for distance."""
    if frame_height <= 0:
        return 0.0
    _, _, _, h = box
    return h / float(frame_height)


def horizontal_offset_ratio(box: tuple[int, int, int, int], frame_width: int) -> float:
    """Horizontal offset of the box centre from the frame centre, as a fraction of frame width.

    Positive => box is right of centre. Negative => box is left of centre.
    """
    if frame_width <= 0:
        return 0.0
    cx, _ = box_center(box)
    frame_center_x = frame_width / 2.0
    return (cx - frame_center_x) / float(frame_width)


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Intersection over Union between two boxes — used to associate detections with trackers."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, aw) * max(0, ah)
    area_b = max(0, bw) * max(0, bh)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / float(union)


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def is_box_in_lower_region(
    box: tuple[int, int, int, int], frame_height: int, y_ratio_threshold: float
) -> bool:
    """Returns True if the box's bottom edge is in the lower portion of the frame (i.e. close to the robot)."""
    if frame_height <= 0:
        return False
    _, y, _, h = box
    bottom = y + h
    return (bottom / float(frame_height)) >= y_ratio_threshold


def is_box_in_path_zone(
    box: tuple[int, int, int, int],
    frame_width: int,
    x_margin_ratio: float,
) -> bool:
    """Returns True if the box centre is within the robot's forward path zone (around frame centre horizontally)."""
    if frame_width <= 0:
        return False
    cx, _ = box_center(box)
    frame_center = frame_width / 2.0
    margin = frame_width * x_margin_ratio
    return (frame_center - margin) <= cx <= (frame_center + margin)
