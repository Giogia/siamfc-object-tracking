import numpy as np


def region_to_bbox(region, center=True):

    n = len(region)
    assert n == 4 or n == 8, 'Groundtruth region format is invalid, should have 4 or 8 entries.'

    return rectangle(region, center) if n == 4 else polygon(region, center)


# Assuming bounding boxes are saved with 0-indexing
def rectangle(region, center):

    if center:
        top_left_corner_x, top_left_corner_y, width, height = region
        center_x = top_left_corner_x + width / 2
        center_y = top_left_corner_y + height / 2

        return center_x, center_y, width, height

    return region


def polygon(region, center):

    center_x = np.mean(region[::2])
    center_y = np.mean(region[1::2])

    x1 = np.min(region[::2])
    x2 = np.max(region[::2])
    y1 = np.min(region[1::2])
    y2 = np.max(region[1::2])

    a1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
    a2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(a1 / a2)
    width = s * (x2 - x1) + 1
    height = s * (y2 - y1) + 1

    if center:
        return center_x, center_y, width, height

    return center_x - width / 2, center_y - height / 2, width, height
