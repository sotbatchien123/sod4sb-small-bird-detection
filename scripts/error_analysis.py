"""Error analysis of one or more saved prediction files on the audited MVA validation set (CPU only).

For each predictions file (COCO results, e.g. reports/sahi_tta/<name>_predictions.json):
  - AP50 and recall@IoU0.5 by bird size, sqrt(area) buckets 0-8, 8-16, 16-32, 32-64, 64+ px;
  - at the score threshold with the best F1 (greedy IoU>=0.5 matching per image): TP, FP, FN, precision, recall;
  - example panels (crops around birds, x3, GT green / TP blue / FP red / FN yellow) for the slide deck.
Writes reports/error_analysis/<name>.json and <name>_{fn,fp,tp}.png; prints a comparison table.

python scripts/error_analysis.py reports/sahi_tta/baseline_m_e30_full_predictions.json reports/sahi_tta/slicing_m_e30_full_predictions.json
"""
import contextlib, io, json, sys
from pathlib import Path
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from paths import rel

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT/'data/mva_audited_v1/annotations/split_val_coco.json'
IMAGES = ROOT/'mva2023_sod4bird_train/images'
OUT = ROOT/'reports/error_analysis'
BUCKETS = [('0-8', 0, 8), ('8-16', 8, 16), ('16-32', 16, 32), ('32-64', 32, 64), ('64+', 64, 1e5)]
COLORS = dict(gt=(0, 200, 0), tp=(255, 120, 0), fp=(0, 0, 255), fn=(0, 220, 255))  # BGR


def by_size(preds):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANN)); ev = COCOeval(gt, gt.loadRes(preds), 'bbox')
        ev.params.areaRng = [[0, 1e10]] + [[lo*lo, hi*hi] for _, lo, hi in BUCKETS]
        ev.params.areaRngLbl = ['all'] + [b[0] for b in BUCKETS]; ev.params.maxDets = [1, 10, 100]
        ev.evaluate(); ev.accumulate()
    out = {}
    for a, label in enumerate(ev.params.areaRngLbl):
        pr = ev.eval['precision'][0, :, 0, a, 2]; rc = ev.eval['recall'][0, 0, a, 2]
        n = sum(1 for g in gt.dataset['annotations'] if ev.params.areaRng[a][0] <= g['area'] < ev.params.areaRng[a][1])
        out[label] = dict(AP50=float(pr[pr > -1].mean()) if (pr > -1).any() else None, recall50=float(rc) if rc > -1 else None, gt_count=n)
    return out


def iou(a, b):
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix*iy; area = lambda x: (x[:, 2]-x[:, 0])*(x[:, 3]-x[:, 1])
    return inter/np.maximum(area(a)[:, None]+area(b)[None, :]-inter, 1e-9)


def match(gts, preds, thr):
    """Greedy matching at IoU>=0.5 of predictions with score>=thr: returns per image (tp, fp, fn) boxes."""
    res = {}
    for i, g in gts.items():
        p = preds.get(i, np.zeros((0, 5))); p = p[p[:, 4] >= thr]; p = p[np.argsort(-p[:, 4])]
        used = np.zeros(len(g), bool); tp, fp = [], []
        o = iou(p[:, :4], g) if len(p) and len(g) else np.zeros((len(p), len(g)))
        for k in range(len(p)):
            cand = np.where((o[k] >= .5) & ~used)[0] if len(g) else []
            if len(cand): j = cand[o[k, cand].argmax()]; used[j] = True; tp.append(p[k])
            else: fp.append(p[k])
        res[i] = (np.array(tp).reshape(-1, 5), np.array(fp).reshape(-1, 5), g[~used])
    return res


def panel(items, path, n=24, crop=96, scale=3):
    """items: list of (image_id, box xyxy, kind, gt boxes, preds of that image) -> grid of crops."""
    tiles = []
    for image_id, box, kind, g, p in items[:n]:
        img = cv2.imread(str(IMAGES/f'{image_id:05d}.jpg')) if False else cv2.imread(str(IMAGES/FILES[image_id]))
        cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
        x0 = int(np.clip(cx-crop/2, 0, img.shape[1]-crop)); y0 = int(np.clip(cy-crop/2, 0, img.shape[0]-crop))
        c = cv2.resize(img[y0:y0+crop, x0:x0+crop], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        for b in g: cv2.rectangle(c, tuple(int((v-o)*scale) for v, o in zip(b[:2], (x0, y0))), tuple(int((v-o)*scale) for v, o in zip(b[2:4], (x0, y0))), COLORS['gt'], 1)
        b = box; cv2.rectangle(c, (int((b[0]-x0)*scale), int((b[1]-y0)*scale)), (int((b[2]-x0)*scale), int((b[3]-y0)*scale)), COLORS[kind], 2)
        label = kind.upper() + (f' {b[4]:.2f}' if len(b) > 4 else '')
        cv2.putText(c, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .5, COLORS[kind], 1, cv2.LINE_AA)
        tiles.append(c)
    if not tiles: return
    while len(tiles) % 6: tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r+6]) for r in range(0, len(tiles), 6)]
    cv2.imwrite(str(path), np.vstack(rows))


def analyse(path):
    preds_json = json.loads(Path(path).read_text(encoding='utf-8'))
    name = Path(path).name.replace('_predictions.json', '')
    sizes = by_size(preds_json)
    gts = {i: np.zeros((0, 4)) for i in FILES}
    for a in ANN_JSON['annotations']:
        x, y, w, h = a['bbox']; gts[a['image_id']] = np.vstack([gts[a['image_id']], [x, y, x+w, y+h]])
    preds = {}
    for p in preds_json:
        x, y, w, h = p['bbox']; preds.setdefault(p['image_id'], []).append([x, y, x+w, y+h, p['score']])
    preds = {k: np.array(v) for k, v in preds.items()}
    best = None
    for thr in np.arange(.05, .9, .025):
        m = match(gts, preds, thr); tp = sum(len(v[0]) for v in m.values()); fp = sum(len(v[1]) for v in m.values()); fn = sum(len(v[2]) for v in m.values())
        f1 = 2*tp/max(1, 2*tp+fp+fn)
        if best is None or f1 > best['F1']: best = dict(threshold=float(thr), TP=tp, FP=fp, FN=fn, precision=tp/max(1, tp+fp), recall=tp/max(1, tp+fn), F1=f1)
    m = match(gts, preds, best['threshold'])
    rng = np.random.RandomState(0)
    for kind, idx in (('fn', 2), ('fp', 1), ('tp', 0)):
        items = [(i, b, kind, gts[i], preds.get(i)) for i, v in m.items() for b in v[idx]]
        if kind == 'fp': items.sort(key=lambda t: -t[1][4])  # most confident false positives first
        else: rng.shuffle(items)
        panel(items, OUT/f'{name}_{kind}.png')
    result = dict(source=rel(path), by_size=sizes, best_f1=best)
    (OUT/f'{name}.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    return name, result


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    ANN_JSON = json.loads(ANN.read_text(encoding='utf-8'))
    FILES = {im['id']: im['file_name'] for im in ANN_JSON['images']}
    rows = [analyse(p) for p in sys.argv[1:]]
    labels = [b[0] for b in BUCKETS]
    print('AP50 by bird size (sqrt area px) | best-F1 operating point')
    print(f"{'run':38s} " + ' '.join(f'{l:>7s}' for l in ['all']+labels) + '   thr   prec   rec    F1   FP    FN')
    for name, r in rows:
        s = r['by_size']; b = r['best_f1']
        cells = ' '.join(f"{s[l]['AP50']*100:7.1f}" if s[l]['AP50'] is not None else '      -' for l in ['all']+labels)
        print(f"{name:38s} {cells}  {b['threshold']:.2f} {b['precision']*100:5.1f} {b['recall']*100:5.1f} {b['F1']*100:5.1f} {b['FP']:5d} {b['FN']:5d}")
    print('GT count per bucket:', {l: rows[0][1]['by_size'][l]['gt_count'] for l in labels})
