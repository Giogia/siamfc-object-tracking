import numpy as np


def region_to_bbox(region, center=True):

    n = len(region)
    assert n == 4 or n == 8, 'Groundtruth region format is invalid, should have 4 or 8 entries.'

    if n == 4:
        return rectangle(region, center)
    else:
        return polygon(region, center)


# Assuming bounding boxes are saved with 0-indexing
def rectangle(region, center):

    if center:
        x, y, w, h = region
        cx = x + w / 2
        cy = y + h / 2

        return cx, cy, w, h

    else:
        return region


def polygon(region, center):

    cx = np.mean(region[::2])
    cy = np.mean(region[1::2])
    x1 = np.min(region[::2])
    x2 = np.max(region[::2])
    y1 = np.min(region[1::2])
    y2 = np.max(region[1::2])

    a1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
    a2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(a1 / a2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1

    if center:
        return cx, cy, w, h
    else:
        return cx - w / 2, cy - h / 2, w, h
