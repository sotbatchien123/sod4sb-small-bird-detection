"""Try other ways of merging the saved raw boxes of eval_sahi_tta.py, on the CPU (no GPU pass).

Raw file layout: per image the concatenated (n,5) boxes and '<id>__parts' with each part's length. Parts are
ordered full image [, full flipped] then per tile [plain [, flipped]] (flipped present for tta / sahi_tta).
Rules compared:
  nms_<iou>       : all parts, NMS (the eval script uses IoU 0.6)
  wbf             : all parts, weighted boxes fusion (IoU 0.55), score = mean over the parts that could vote
  small_tiles_<s> : full-image boxes always; tile boxes only if sqrt(area) < s px (zooming helps small birds)
Selection bias: picking the best rule on the same images inflates it, so every rule is also scored on two
halves (odd / even image ids): choose on one half, report the other.

python scripts/merge_tune.py reports/sahi_tta/slicing_m_e30_sahi_tta_z1.5_raw.npz
"""
import contextlib, io, json, sys
from pathlib import Path
import numpy as np
import torch, torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT/'data/mva_audited_v1/annotations/split_val_coco.json'
MAX_DET = 300


def nms(b, iou):
    if not len(b): return b
    keep = torchvision.ops.nms(torch.from_numpy(b[:, :4]), torch.from_numpy(b[:, 4]), iou).numpy()[:MAX_DET]
    return b[keep]


def wbf(b, voters, iou=.55):
    """Greedy weighted boxes fusion over score-sorted boxes; fused score = sum of scores / voters."""
    if not len(b): return b
    b = b[np.argsort(-b[:, 4])]; fused, members = [], []
    for box in b:
        if fused:
            f = np.array([x[:4] for x in fused])
            ix1 = np.maximum(f[:, 0], box[0]); iy1 = np.maximum(f[:, 1], box[1])
            ix2 = np.minimum(f[:, 2], box[2]); iy2 = np.minimum(f[:, 3], box[3])
            inter = np.clip(ix2-ix1, 0, None)*np.clip(iy2-iy1, 0, None)
            union = (f[:, 2]-f[:, 0])*(f[:, 3]-f[:, 1]) + (box[2]-box[0])*(box[3]-box[1]) - inter
            o = inter/np.maximum(union, 1e-9); j = int(o.argmax())
            if o[j] > iou:
                members[j].append(box); m = np.array(members[j]); w = m[:, 4:5]
                fused[j] = np.concatenate([(m[:, :4]*w).sum(0)/w.sum(), [m[:, 4].sum()]]); continue
        fused.append(box.copy()); members.append([box])
    out = np.array(fused); out[:, 4] = out[:, 4]/max(1, voters)
    return out[np.argsort(-out[:, 4])][:MAX_DET]


def coco(preds, ids):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANN))
        if not preds: return 0., 0.
        ev = COCOeval(gt, gt.loadRes(preds), 'bbox'); ev.params.imgIds = ids; ev.params.maxDets = [1, 10, 100]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats[1], ev.stats[0]


VIEW = (1280, 736)


def windows(w, h, zoom, overlap=.25):
    """Same tile layout as scripts/eval_sahi_tta.py (kept in sync by hand)."""
    sw, sh = round(VIEW[0]/zoom), round(VIEW[1]/zoom)
    def starts(size, win):
        step = max(1, int(win*(1-overlap)))
        s = list(range(0, max(size-win, 0)+1, step))
        if s[-1] != size-win: s.append(size-win)
        return s
    return [(x, y, sw, sh) for y in starts(h, sh) for x in starts(w, sw)]


def edge_drop(t, win, w=3840, h=2160, margin=3.):
    """Drop tile boxes touching an inner tile edge (a bird cut by the tile border); image borders are kept."""
    x, y, sw, sh = win
    bad = np.zeros(len(t), bool)
    if x > 0: bad |= t[:, 0] <= x + margin
    if y > 0: bad |= t[:, 1] <= y + margin
    if x + sw < w: bad |= t[:, 2] >= x + sw - margin
    if y + sh < h: bad |= t[:, 3] >= y + sh - margin
    return t[~bad]


def supported(t, full, iou=.3, min_score=.05):
    """Keep tile boxes that overlap a full-image box of score >= min_score (tiles refine, not discover)."""
    f = full[full[:, 4] >= min_score]
    if not len(t) or not len(f): return t[:0]
    ix = np.clip(np.minimum(t[:, None, 2], f[None, :, 2]) - np.maximum(t[:, None, 0], f[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(t[:, None, 3], f[None, :, 3]) - np.maximum(t[:, None, 1], f[None, :, 1]), 0, None)
    inter = ix*iy; area = lambda b: (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    o = inter/np.maximum(area(t)[:, None] + area(f)[None, :] - inter, 1e-9)
    return t[o.max(1) >= iou]


def main(path):
    raw = np.load(path); ann = json.loads(ANN.read_text(encoding='utf-8'))
    ids = [im['id'] for im in ann['images']]; cat = ann['categories'][0]['id']
    name = Path(path).name; flip = 'tta' in name
    full_parts = 2 if flip else 1; per_tile = 2 if flip else 1
    tiles = max(0, (len(raw[f'{ids[0]}__parts']) - full_parts)//per_tile)
    voters = len(raw[f'{ids[0]}__parts'])
    rules = {f'nms_{i}': None for i in (.5, .6, .7)} | {'wbf': None}
    if tiles:
        zoom = float(name.split('_z')[1].split('_')[0])
        wins = windows(3840, 2160, zoom); assert len(wins) == tiles, (len(wins), tiles)
        rules |= {f'small_tiles_{s}': None for s in (16, 24, 32)}
        rules |= {'edge_nms0.5': None, 'edge_tilew0.7_nms0.5': None, 'edge_tilew0.5_nms0.5': None,
                  'support_edge_nms0.5': None, 'edge_wbf': None, 'support_edge_wbf': None, 'full_only_wbf': None}
    results = {}
    for rule in rules:
        preds = []
        for i in ids:
            b = raw[str(i)].copy(); lens = raw[f'{i}__parts']; nfull = int(lens[:full_parts].sum())
            if tiles and (rule.startswith(('edge', 'support', 'full_only'))):
                full = b[:nfull]; offs = np.cumsum(np.r_[0, lens]); kept = []
                if rule != 'full_only_wbf':
                    for j in range(full_parts, len(lens)):
                        t = b[offs[j]:offs[j+1]]; t = edge_drop(t, wins[(j-full_parts)//per_tile])
                        if 'support' in rule: t = supported(t, full)
                        if 'tilew' in rule: t = t.copy(); t[:, 4] *= float(rule.split('tilew')[1].split('_')[0])
                        kept.append(t)
                b = np.concatenate([full] + kept)
                b = wbf(b, full_parts if rule == 'full_only_wbf' else voters) if rule.endswith('wbf') else nms(b, .5)
            elif rule.startswith('small_tiles_'):
                s = float(rule.split('_')[-1]); t = b[nfull:]
                t = t[np.sqrt((t[:, 2]-t[:, 0])*(t[:, 3]-t[:, 1])) < s]
                b = nms(np.concatenate([b[:nfull], t]), .6)
            elif rule == 'wbf': b = wbf(b, voters)
            else: b = nms(b, float(rule.split('_')[1]))
            preds += [dict(image_id=i, category_id=cat, bbox=[float(x1), float(y1), float(x2-x1), float(y2-y1)], score=float(sc))
                      for x1, y1, x2, y2, sc in b if x2 > x1 and y2 > y1]
        full = coco(preds, ids); odd = coco(preds, ids[1::2]); even = coco(preds, ids[0::2])
        results[rule] = dict(AP50=full[0], AP=full[1], AP50_odd=odd[0], AP50_even=even[0])
        print(f'{rule:16s} AP50 {full[0]*100:5.1f}  AP {full[1]*100:5.1f}  | odd {odd[0]*100:5.1f}  even {even[0]*100:5.1f}', flush=True)
    pick = lambda half: max(results, key=lambda r: results[r][f'AP50_{half}'])
    honest = dict(chosen_on_odd=pick('odd'), its_AP50_on_even=results[pick('odd')]['AP50_even'],
                  chosen_on_even=pick('even'), its_AP50_on_odd=results[pick('even')]['AP50_odd'])
    print('held-out check:', json.dumps({k: (round(v*100, 1) if isinstance(v, float) else v) for k, v in honest.items()}))
    out = Path(path).with_name(Path(path).name.replace('_raw.npz', '_merge_tune.json'))
    out.write_text(json.dumps(dict(source=str(path), rules=results, held_out=honest), indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1])
