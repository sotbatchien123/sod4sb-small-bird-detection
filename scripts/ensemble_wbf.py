"""Ensemble saved COCO-format predictions of several detectors on the audited MVA val set (CPU only).

Rules compared (per image, all models' boxes pooled):
  nms_<iou>            : concatenate, NMS
  wbf_i<iou>_w<w>      : weighted boxes fusion; model k has weight w_k (first model gets w, others 1);
                         fused box = score*weight-weighted mean of members, fused score =
                         sum over models of (weight * best member score of that model) / sum of weights
Selection bias: every rule is also scored on odd / even image ids; choose on one half, report the other.
Writes reports/ensemble/<name>.json (all rules + held-out check) and <name>_<best>_predictions.json.

python scripts/ensemble_wbf.py --name slicing_tta+gfl reports/sahi_tta/slicing_m_e30_tta_predictions.json work_dirs/gfl_drone12_mva7_1440/mva_epoch_38.bbox.json
"""
import argparse, contextlib, io, json
from pathlib import Path
import numpy as np
import torch, torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT/'data/mva_audited_v1/annotations/split_val_coco.json'
OUT = ROOT/'reports/ensemble'
MAX_DET = 300


def load(path):
    per = {}
    for p in json.loads(Path(path).read_text(encoding='utf-8')):
        x, y, w, h = p['bbox']; per.setdefault(p['image_id'], []).append([x, y, x+w, y+h, p['score']])
    return {k: np.array(v, np.float32) for k, v in per.items()}


def iou_one(f, box):
    ix = np.clip(np.minimum(f[:, 2], box[2]) - np.maximum(f[:, 0], box[0]), 0, None)
    iy = np.clip(np.minimum(f[:, 3], box[3]) - np.maximum(f[:, 1], box[1]), 0, None)
    inter = ix*iy
    return inter/np.maximum((f[:, 2]-f[:, 0])*(f[:, 3]-f[:, 1]) + (box[2]-box[0])*(box[3]-box[1]) - inter, 1e-9)


def wbf(model_boxes, weights, iou):
    """model_boxes: list of (n_k,5) arrays. Returns fused (m,5)."""
    allb = np.concatenate([np.c_[b, np.full(len(b), k)] for k, b in enumerate(model_boxes) if len(b)]) if any(len(b) for b in model_boxes) else np.zeros((0, 6))
    if not len(allb): return np.zeros((0, 5))
    allb = allb[np.argsort(-allb[:, 4])]
    fused = np.zeros((0, 4)); members = []
    for box in allb:
        if len(fused):
            o = iou_one(fused, box); j = int(o.argmax())
            if o[j] > iou:
                members[j].append(box); m = np.array(members[j]); w = m[:, 4]*np.array([weights[int(k)] for k in m[:, 5]])
                fused[j] = (m[:, :4]*w[:, None]).sum(0)/w.sum(); continue
        fused = np.vstack([fused, box[:4]]); members.append([box])
    scores = []
    for m in members:
        m = np.array(m); best = {}
        for s, k in zip(m[:, 4], m[:, 5].astype(int)): best[k] = max(best.get(k, 0), s)
        scores.append(sum(weights[k]*s for k, s in best.items())/sum(weights))
    out = np.c_[fused, scores]
    return out[np.argsort(-out[:, 4])][:MAX_DET]


def nms(model_boxes, iou):
    b = np.concatenate([x for x in model_boxes if len(x)]) if any(len(x) for x in model_boxes) else np.zeros((0, 5), np.float32)
    if not len(b): return b
    keep = torchvision.ops.nms(torch.from_numpy(b[:, :4].astype(np.float32)), torch.from_numpy(b[:, 4].astype(np.float32)), iou).numpy()[:MAX_DET]
    return b[keep]


def coco(preds, ids):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANN)); ev = COCOeval(gt, gt.loadRes(preds), 'bbox'); ev.params.imgIds = ids; ev.params.maxDets = [1, 10, 100]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return dict(AP=ev.stats[0], AP50=ev.stats[1], AP75=ev.stats[2], AP_small=ev.stats[3], AR100=ev.stats[8])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--name', required=True); p.add_argument('preds', nargs='+')
    p.add_argument('--first-weights', default='1,1.5,2,3'); p.add_argument('--ious', default='0.5,0.55,0.6')
    args = p.parse_args()
    ann = json.loads(ANN.read_text(encoding='utf-8')); ids = [im['id'] for im in ann['images']]; cat = ann['categories'][0]['id']
    models = [load(x) for x in args.preds]; OUT.mkdir(parents=True, exist_ok=True)
    rules = {}
    for k, path in enumerate(args.preds): rules[f'single_{k}'] = ('single', k)
    for i in (.5, .6): rules[f'nms_{i}'] = ('nms', i)
    for w in map(float, args.first_weights.split(',')):
        for i in map(float, args.ious.split(',')): rules[f'wbf_i{i}_w{w:g}'] = ('wbf', (i, w))
    results, all_preds = {}, {}
    for name, (kind, par) in rules.items():
        preds = []
        for i in ids:
            mb = [m.get(i, np.zeros((0, 5), np.float32)) for m in models]
            if kind == 'single': b = mb[par]
            elif kind == 'nms': b = nms(mb, par)
            else: b = wbf(mb, [par[1]] + [1.]*(len(mb)-1), par[0])
            preds += [dict(image_id=i, category_id=cat, bbox=[float(x1), float(y1), float(x2-x1), float(y2-y1)], score=float(s))
                      for x1, y1, x2, y2, s in b if x2 > x1 and y2 > y1]
        r = coco(preds, ids); r['AP50_odd'] = coco(preds, ids[1::2])['AP50']; r['AP50_even'] = coco(preds, ids[0::2])['AP50']
        results[name] = r; all_preds[name] = preds
        print(f"{name:18s} AP50 {r['AP50']*100:5.1f}  AP {r['AP']*100:5.1f}  AP75 {r['AP75']*100:5.1f}  AP_s {r['AP_small']*100:5.1f}  AR100 {r['AR100']*100:5.1f} | odd {r['AP50_odd']*100:5.1f} even {r['AP50_even']*100:5.1f}", flush=True)
    pick = lambda half: max(results, key=lambda r: results[r][f'AP50_{half}'])
    held = dict(chosen_on_odd=pick('odd'), its_AP50_on_even=results[pick('odd')]['AP50_even'],
                chosen_on_even=pick('even'), its_AP50_on_odd=results[pick('even')]['AP50_odd'],
                single_0_AP50_on_even=results['single_0']['AP50_even'], single_0_AP50_on_odd=results['single_0']['AP50_odd'])
    print('held-out check:', json.dumps({k: (round(v*100, 1) if isinstance(v, float) else v) for k, v in held.items()}))
    best = max(results, key=lambda r: results[r]['AP50'])
    (OUT/f'{args.name}.json').write_text(json.dumps(dict(sources=args.preds, rules=results, held_out=held, best=best), indent=1), encoding='utf-8')
    (OUT/f'{args.name}_{best}_predictions.json').write_text(json.dumps(all_preds[best]), encoding='utf-8')
    print('best:', best)


if __name__ == '__main__':
    main()
