from target_track_robot.utils import geometry


def test_box_center():
    assert geometry.box_center((10, 10, 20, 40)) == (20.0, 30.0)


def test_box_height_ratio():
    assert geometry.box_height_ratio((0, 0, 50, 240), 480) == 0.5
    assert geometry.box_height_ratio((0, 0, 50, 240), 0) == 0.0


def test_horizontal_offset_ratio_center():
    # Box whose centre is exactly at the middle of a 640-wide frame -> offset = 0
    box = (270, 0, 100, 100)  # center x = 320
    assert geometry.horizontal_offset_ratio(box, 640) == 0.0


def test_horizontal_offset_ratio_right():
    box = (500, 0, 100, 100)  # center x = 550 -> right of centre
    offset = geometry.horizontal_offset_ratio(box, 640)
    assert offset > 0


def test_horizontal_offset_ratio_left():
    box = (0, 0, 100, 100)  # center x = 50 -> left of centre
    offset = geometry.horizontal_offset_ratio(box, 640)
    assert offset < 0


def test_iou_identical_boxes():
    box = (10, 10, 50, 50)
    assert geometry.iou(box, box) == 1.0


def test_iou_no_overlap():
    assert geometry.iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_iou_partial_overlap():
    box_a = (0, 0, 10, 10)
    box_b = (5, 5, 10, 10)
    value = geometry.iou(box_a, box_b)
    assert 0.0 < value < 1.0


def test_clamp():
    assert geometry.clamp(5, 0, 10) == 5
    assert geometry.clamp(-5, 0, 10) == 0
    assert geometry.clamp(50, 0, 10) == 10


def test_is_box_in_lower_region():
    frame_h = 480
    low_box  = (0, 400, 50, 70)  # bottom = 470 -> 97% -> in the lower region
    high_box = (0, 0,   50, 50)  # bottom = 50  -> 10%
    assert geometry.is_box_in_lower_region(low_box, frame_h, 0.55) is True
    assert geometry.is_box_in_lower_region(high_box, frame_h, 0.55) is False


def test_is_box_in_path_zone():
    frame_w = 640
    center_box   = (300, 0, 40, 40)  # center x = 320 = exact frame centre
    far_left_box = (0,   0, 10, 10)  # center x = 5
    assert geometry.is_box_in_path_zone(center_box, frame_w, 0.3) is True
    assert geometry.is_box_in_path_zone(far_left_box, frame_w, 0.3) is False
