"""Path resolution and baseline full-image flip for the re-extracted Drone data."""
from pathlib import Path
import cv2
import numpy as np


def image_path(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe annotation path')
    direct = root/'images'/relative
    nested = root/'images'/relative.parts[0]/relative
    matches = [p for p in (direct, nested) if p.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f'Missing or ambiguous image: {name}')
    return matches[0]


class BaselineViews:
    def __call__(self, raw, boxes, seed, image_id):
        h, w = raw.shape[:2]
        canvas = cv2.resize(raw, (2560, 1440))
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        boxes /= np.array([w, h, w, h], dtype=np.float32)
        if np.random.RandomState(seed).rand() < .5:
            canvas = canvas[:, ::-1]
            boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]
        norm = boxes.copy()
        norm[:, :2] = (boxes[:, :2] + boxes[:, 2:]) / 2
        norm[:, 2:] = boxes[:, 2:] - boxes[:, :2]
        if not np.isfinite(norm).all() or not ((norm >= 0) & (norm <= 1)).all():
            raise RuntimeError('Invalid boxes')
        return canvas, norm, dict(fallback=False, window=[0, 0, w, h],
                                 retained_boxes=len(norm), pasted_count=0,
                                 photometric_applied=False)
