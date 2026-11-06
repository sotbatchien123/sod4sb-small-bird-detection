"""COCO evaluation used throughout the paper: AP50, AP75 and AR100 (at most 100 detections per image).

    python evaluate.py results/val_predictions.json                       # against the audited validation labels
    python evaluate.py preds.json --ann data/mva_audited_v1/annotations/split_val_coco.json --by-size
"""
import argparse, contextlib, io, json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
VAL_ANN = ROOT/'data/mva_audited_v1/annotations/split_val_coco.json'
SIZE_BUCKETS = {'8-16': (8, 16), '16-32': (16, 32), '32-64': (32, 64)}   # sqrt(area) in pixels


def evaluate(preds, ann=VAL_ANN, image_ids=None, area_rng=None):
    """preds: list of COCO result dicts (image_id, category_id, bbox=[x, y, w, h], score) or a path to such a JSON file.
    image_ids: restrict to these images (default: every image of the annotation file).
    area_rng: optional (min, max) of sqrt(area) in pixels, for the scale-wise analysis.
    Returns {'AP50', 'AP75', 'AR100'} as fractions."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if isinstance(preds, (str, Path)):
        preds = json.loads(Path(preds).read_text(encoding='utf-8'))
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ann))
        if not preds:
            return dict(AP50=0., AP75=0., AR100=0.)
        ev = COCOeval(gt, gt.loadRes(preds), 'bbox')
        ev.params.imgIds = list(image_ids) if image_ids is not None else gt.getImgIds()
        ev.params.maxDets = [1, 10, 100]
        if area_rng is not None:
            ev.params.areaRng = [[area_rng[0]**2, area_rng[1]**2]] * 4   # all four COCO ranges -> this bucket
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return dict(AP50=float(ev.stats[1]), AP75=float(ev.stats[2]), AR100=float(ev.stats[8]))


def report(m):
    return '  '.join(f'{k} {v*100:.1f}%' for k, v in m.items())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('predictions')
    p.add_argument('--ann', default=str(VAL_ANN))
    p.add_argument('--by-size', action='store_true', help='also report the three sqrt(area) buckets of the paper')
    args = p.parse_args()
    preds = json.loads(Path(args.predictions).read_text(encoding='utf-8'))
    print('all     ', report(evaluate(preds, args.ann)))
    if args.by_size:
        for name, rng in SIZE_BUCKETS.items():
            print(f'{name:8s}', report(evaluate(preds, args.ann, area_rng=rng)))
