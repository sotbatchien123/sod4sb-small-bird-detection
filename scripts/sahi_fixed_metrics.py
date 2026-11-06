"""AP50 / AP75 / AR100 of the corrected SAHI merge rules reported in the paper (merge_tune.py saved only AP50 and AP).
Re-applies each chosen rule to the saved raw boxes (CPU, no GPU pass) with the same code as merge_tune.py.

python scripts/sahi_fixed_metrics.py   ->  reports/sahi_tta/sahi_fixed_metrics.json
"""
import contextlib, io, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import merge_tune as mt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

RUNS = {  # raw file -> rule shown as "Fixed" in the paper (best AP50 in merge_tune.json)
    'slicing_m_e30_sahi_z1.5_raw.npz': 'edge_tilew0.5_nms0.5',
    'slicing_m_e30_sahi_tta_z1.5_raw.npz': 'edge_tilew0.5_nms0.5',
    'slicing_m_e30_sahi_z1_raw.npz': 'edge_nms0.5',
}


def evaluate(preds, ids):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(mt.ANN)); ev = COCOeval(gt, gt.loadRes(preds), 'bbox')
        ev.params.imgIds = ids; ev.params.maxDets = [1, 10, 100]; ev.evaluate(); ev.accumulate(); ev.summarize()
    return dict(AP50=ev.stats[1], AP75=ev.stats[2], AR100=ev.stats[8])


def run(name, rule):
    path = ROOT/'reports/sahi_tta'/name
    raw = np.load(path); ann = json.loads(mt.ANN.read_text(encoding='utf-8'))
    ids = [im['id'] for im in ann['images']]; cat = ann['categories'][0]['id']
    flip = 'tta' in name; full_parts = 2 if flip else 1; per_tile = 2 if flip else 1
    tiles = (len(raw[f'{ids[0]}__parts']) - full_parts)//per_tile
    wins = mt.windows(3840, 2160, float(name.split('_z')[1].split('_')[0])); assert len(wins) == tiles
    w = float(rule.split('tilew')[1].split('_')[0]) if 'tilew' in rule else None
    preds = []
    for i in ids:
        b = raw[str(i)]; lens = raw[f'{i}__parts']; offs = np.cumsum(np.r_[0, lens]); nfull = int(lens[:full_parts].sum())
        kept = []
        for j in range(full_parts, len(lens)):
            t = mt.edge_drop(b[offs[j]:offs[j+1]], wins[(j-full_parts)//per_tile])
            if w is not None: t = t.copy(); t[:, 4] *= w
            kept.append(t)
        m = mt.nms(np.concatenate([b[:nfull]] + kept), .5)
        preds += [dict(image_id=i, category_id=cat, bbox=[float(x1), float(y1), float(x2-x1), float(y2-y1)], score=float(s))
                  for x1, y1, x2, y2, s in m if x2 > x1 and y2 > y1]
    return evaluate(preds, ids)


if __name__ == '__main__':
    out = {}
    for name, rule in RUNS.items():
        r = run(name, rule); out[name.replace('_raw.npz', '')] = dict(rule=rule, **r)
        print(name, rule, {k: round(v*100, 1) for k, v in r.items()}, flush=True)
    (ROOT/'reports/sahi_tta/sahi_fixed_metrics.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
