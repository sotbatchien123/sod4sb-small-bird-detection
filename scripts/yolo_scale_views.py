"""Shared geometry for Augmentation, Slicing and Copy-Paste. Coordinates are xyxy pixels."""
import numpy as np

OUTPUT = (2560, 1440)


def xyxy(annotations):
    boxes = np.asarray([a['bbox'] for a in annotations], dtype=np.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def intersect(boxes, window):
    out = boxes.copy()
    out[:, :2] = np.maximum(out[:, :2], window[:2])
    out[:, 2:] = np.minimum(out[:, 2:], window[2:])
    return out, np.maximum(out[:, 2:] - out[:, :2], 0).prod(axis=1)


def tile_windows(width, height, tile=(2560, 1440), overlap=.2):
    if not 0 <= overlap < 1:
        raise ValueError('Invalid overlap')
    def starts(size, crop):
        if crop > size:
            raise ValueError('Tile exceeds image')
        values = list(range(0, size-crop+1, max(1, round(crop*(1-overlap)))))
        if values[-1] != size-crop:
            values.append(size-crop)
        return values
    return [(x, y, x+tile[0], y+tile[1])
            for y in starts(height, tile[1]) for x in starts(width, tile[0])]

