"""Data access for SOD4SB: COCO annotation files, image lists and frames.

Expected layout (see data/README):
    mva2023_sod4bird_train/images/*.jpg                       MVA 2023 frames (from the organisers)
    data/mva_audited_v1/annotations/split_{train,val}_coco.json   audited labels (python scripts/prepare_fixed50.py)
    drone2021/{images,annotations}                            Drone2021, for pretraining only
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT/'mva2023_sod4bird_train/images'
ANN_DIR = ROOT/'data/mva_audited_v1/annotations'


def ann_path(split='val'):
    """Audited annotation file of the 'train' or 'val' split."""
    return ANN_DIR/f'split_{split}_coco.json'


def load_annotations(path=None, split='val'):
    """Return (images, boxes): images is the COCO image list, boxes maps image_id -> (n, 4) float32 xyxy array."""
    ann = json.loads(Path(path or ann_path(split)).read_text(encoding='utf-8'))
    boxes = {im['id']: [] for im in ann['images']}
    for a in ann['annotations']:
        x, y, w, h = a['bbox']; boxes[a['image_id']].append((x, y, x + w, y + h))
    return ann['images'], {k: np.asarray(v, np.float32).reshape(-1, 4) for k, v in boxes.items()}


def list_images(folder):
    """Every .jpg/.jpeg/.png in a folder as COCO-style image records with ids in file order (for unlabelled frames)."""
    files = sorted(f for f in Path(folder).iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    return [dict(id=i, file_name=f.name) for i, f in enumerate(files)]


def load_image(file_name, folder=IMAGES):
    """Read one frame as an HxWx3 BGR uint8 array (OpenCV order, as the detectors expect)."""
    import cv2
    img = cv2.imread(str(Path(folder)/file_name))
    if img is None:
        raise FileNotFoundError(Path(folder)/file_name)
    return img
