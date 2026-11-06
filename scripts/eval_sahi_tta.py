"""Inference-time variants on the audited MVA validation set, each compared alone with plain inference.

  full : the trainers' validation (native 3840x2160 padded to 32, one pass)          -> reference
  tta  : full + horizontally flipped full image, boxes flipped back, merged by NMS     -> TTA alone
  sahi : full + overlapping tiles (source window = view/zoom, resized to the 1280x736
         training view), boxes mapped back, merged by NMS (SAHI with full-image pass)  -> SAHI alone
  sahi_tta : sahi where the full image and every tile are also predicted flipped     -> both together

Same model loading (best.pt EMA weights, FP32), NMS settings (conf 0.001, IoU 0.6, max_det 300) and COCO
metrics (maxDets 100 primary, 300 diagnostic) as scripts/train_yolo_data_methods.py. Raw pre-merge boxes are
saved (val and test) so other merge rules / ensembles can be tried on the CPU without running the GPU again.

python -u scripts/run_qos.py scripts/eval_sahi_tta.py --run baseline_m_e30 --mode sahi [--zoom 1.5] [--limit N]
  --checkpoint PATH : any YOLOv8-M checkpoint with a 'model' state dict (e.g. the Drone pretrain, "pretrain-only")
  --split test      : the MVA public test images (no labels): writes COCO-format predictions, no metrics
  --chunk N         : process N images at a time, each chunk saved under reports/sahi_tta/<name>_chunks/;
                      a rerun with the same arguments loads finished chunks and continues (stop-safe)
Merged predictions are always saved (<name>_predictions.json) for error analysis and submission.
"""
import argparse, contextlib, hashlib, io, json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'.yolo-deps'))
os.environ['YOLO_CONFIG_DIR'] = str(ROOT/'.cache/yolo'); os.environ['YOLO_OFFLINE'] = 'true'
import numpy as np
import cv2
import torch
import torchvision
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.ops import non_max_suppression
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

VIEW = (1280, 736)
KEYS = ['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']
SPLITS = {'val': (ROOT/'data/mva_audited_v1/annotations/split_val_coco.json', ROOT/'mva2023_sod4bird_train/images'),
          'test': (ROOT/'mva2023_sod4bird_pub_test/annotations/public_test_coco_empty_ann.json', ROOT/'mva2023_sod4bird_pub_test/images')}
CONF, IOU, MAX_DET = .001, .6, 300


def windows(w, h, zoom, overlap):
    """Source windows of size VIEW/zoom covering the image with the given overlap (last row/column flush)."""
    sw, sh = round(VIEW[0]/zoom), round(VIEW[1]/zoom)
    def starts(size, win):
        step = max(1, int(win*(1-overlap)))
        s = list(range(0, max(size-win, 0)+1, step))
        if s[-1] != size-win: s.append(size-win)
        return s
    return [(x, y, sw, sh) for y in starts(h, sh) for x in starts(w, sw)]


def coco(preds, image_ids, maxdet, ann):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ann))
        if not preds: return {k: 0.0 for k in KEYS}
        ev = COCOeval(gt, gt.loadRes(preds), 'bbox'); ev.params.imgIds = image_ids
        ev.params.maxDets = [1, 10, maxdet]; ev.evaluate(); ev.accumulate(); ev.summarize()
        values = ev.stats.tolist()
        if maxdet != 100:
            pr = ev.eval['precision'][:, :, :, 0, 2]; values[0] = float(pr[pr > -1].mean()) if (pr > -1).any() else -1.
    return dict(zip(KEYS, values))


def save_raw(path, raw, extra=None):
    arrays = {k: (np.concatenate(v) if v else np.zeros((0, 5), np.float32)) for k, v in raw.items()}
    arrays |= {f'{k}__parts': np.array([len(q) for q in v]) for k, v in raw.items()}
    if extra is not None: arrays['__meta'] = np.array(json.dumps(extra))
    np.savez_compressed(path, **arrays)


def load_raw(path):
    z = np.load(path, allow_pickle=False); raw = {}
    for k in z.files:
        if '__' in k: continue
        lens = z[f'{k}__parts']; b = z[k]; off = np.cumsum(np.r_[0, lens])
        raw[k] = [b[off[j]:off[j+1]] for j in range(len(lens))]
    return raw, (json.loads(str(z['__meta'])) if '__meta' in z.files else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, help='folder under work_dirs/yolo_improvements with best.pt, or a name with --checkpoint')
    p.add_argument('--checkpoint', default=None, help='checkpoint path relative to the project (default: <run>/best.pt)')
    p.add_argument('--split', choices=('val', 'test'), default='val')
    p.add_argument('--mode', choices=('full', 'tta', 'sahi', 'sahi_tta'), required=True)
    p.add_argument('--zoom', type=float, default=1.5)
    p.add_argument('--overlap', type=float, default=.25)
    p.add_argument('--tile-batch', type=int, default=4)
    p.add_argument('--limit', type=int, default=0, help='first N val images only (tests)')
    p.add_argument('--chunk', type=int, default=0, help='images per saved chunk (0: one pass, nothing saved until the end)')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    dev = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    ANN, IMAGES = SPLITS[args.split]
    checkpoint = ROOT/args.checkpoint if args.checkpoint else ROOT/'work_dirs/yolo_improvements'/args.run/'best.pt'
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = DetectionModel('yolov8m.yaml', nc=1, verbose=False)
    model.load_state_dict(ck['model'], strict=True); model = model.to(dev).eval()
    ann = json.loads(ANN.read_text(encoding='utf-8'))
    images = ann['images'][:args.limit or None]; cat = ann['categories'][0]['id']
    SAHI = args.mode in ('sahi', 'sahi_tta'); FLIP = args.mode in ('tta', 'sahi_tta')
    tag = args.mode + (f'_z{args.zoom:g}' if SAHI else '')
    out = ROOT/'reports/sahi_tta'; out.mkdir(parents=True, exist_ok=True)
    name = f'{args.run}_{tag}' + ('_test' if args.split == 'test' else '') + (f'_limit{args.limit}' if args.limit else '')
    ck_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def tensor(img):
        h, w = img.shape[:2]
        canvas = np.full((math.ceil(h/32)*32, math.ceil(w/32)*32, 3), 114, dtype=np.uint8); canvas[:h, :w] = img
        return torch.from_numpy(np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1))).float().div(255)

    def predict(batch):
        """batch: list of HxWx3 uint8 BGR of equal size -> per image a list of (n,5) arrays xyxy+score in its
        pixels: the plain prediction, plus (FLIP) the prediction of the mirrored image flipped back."""
        x = torch.stack([tensor(b) for b in batch]).to(dev)
        det = non_max_suppression(model(x), conf_thres=CONF, iou_thres=IOU, max_det=MAX_DET)
        out = [[d[:, :5].float().cpu().numpy()] for d in det]
        if FLIP:
            xf = torch.stack([tensor(np.ascontiguousarray(b[:, ::-1])) for b in batch]).to(dev)
            for o, b, d in zip(out, batch, non_max_suppression(model(xf), conf_thres=CONF, iou_thres=IOU, max_det=MAX_DET)):
                d = d[:, :5].float().cpu().numpy(); w = b.shape[1]
                d[:, [0, 2]] = w - d[:, [2, 0]]; o.append(d)
        return out

    def merge(parts, w, h):
        boxes = np.concatenate(parts) if parts else np.zeros((0, 5), np.float32)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w); boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
        keep = torchvision.ops.nms(torch.from_numpy(boxes[:, :4]), torch.from_numpy(boxes[:, 4]), IOU).numpy()[:MAX_DET]
        return boxes[keep]

    pool = ThreadPoolExecutor(2)

    def run_images(imgs, done_before):
        """Predict a list of image records -> (merged COCO preds, raw parts per image id)."""
        preds, raw, t0 = [], {}, time.time()
        pending = pool.submit(cv2.imread, str(IMAGES/imgs[0]['file_name']))
        with torch.inference_mode():
            for i, im in enumerate(imgs):
                img = pending.result()
                if i+1 < len(imgs): pending = pool.submit(cv2.imread, str(IMAGES/imgs[i+1]['file_name']))
                h, w = img.shape[:2]; parts = []
                parts += predict([img])[0]
                if SAHI:
                    wins = windows(w, h, args.zoom, args.overlap)
                    for s in range(0, len(wins), args.tile_batch):
                        group = wins[s:s+args.tile_batch]
                        tiles = [cv2.resize(img[y:y+sh, x:x+sw], VIEW, interpolation=cv2.INTER_LINEAR) for x, y, sw, sh in group]
                        for (x, y, sw, sh), ds in zip(group, predict(tiles)):
                            for d in ds:
                                d[:, [0, 2]] = d[:, [0, 2]]*sw/VIEW[0] + x; d[:, [1, 3]] = d[:, [1, 3]]*sh/VIEW[1] + y
                                parts.append(d)
                raw[str(im['id'])] = [part.astype(np.float32) for part in parts]
                for x1, y1, x2, y2, score in merge([q.copy() for q in parts], w, h):
                    preds.append(dict(image_id=im['id'], category_id=cat, bbox=[float(x1), float(y1), float(x2-x1), float(y2-y1)], score=float(score)))
                n = done_before+i+1
                if n % 100 == 0: print(f'{n}/{len(images)} images, {time.time()-t0:.0f}s in this chunk', flush=True)
        return preds, raw

    preds, raw, seconds = [], {}, 0.
    size = args.chunk or len(images)
    chunk_dir = out/f'{name}_chunks'
    if args.chunk: chunk_dir.mkdir(exist_ok=True)
    for c0 in range(0, len(images), size):
        part = images[c0:c0+size]
        cfile = chunk_dir/f'chunk_{c0:05d}_{len(part)}.npz'
        if args.chunk and cfile.exists():
            c_raw, meta = load_raw(cfile)
            assert meta['checkpoint_sha256'] == ck_sha and meta['mode'] == tag, f'{cfile} was made with other settings'
            preds += meta['preds']; raw |= c_raw; seconds += meta['seconds']
            print(f'chunk {c0}: loaded {len(part)} images', flush=True); continue
        t0 = time.time(); c_preds, c_raw = run_images(part, c0); c_sec = time.time()-t0
        preds += c_preds; raw |= c_raw; seconds += c_sec
        if args.chunk:
            tmp = chunk_dir/f'{cfile.stem}.tmp.npz'
            save_raw(tmp, c_raw, dict(preds=c_preds, seconds=c_sec, checkpoint_sha256=ck_sha, mode=tag))
            os.replace(tmp, cfile)
            print(f'chunk {c0}: saved {len(part)} images in {c_sec:.0f}s', flush=True)

    (out/f'{name}_predictions.json').write_text(json.dumps(preds), encoding='utf-8')
    ids = [im['id'] for im in images]
    if args.split == 'val':
        metrics = coco(preds, ids, 100, ANN)
        metrics['dense_300'] = {k.replace('AR100', 'AR300'): v for k, v in coco(preds, ids, 300, ANN).items()}
    else:
        metrics = {k: None for k in KEYS}
    metrics.update(run=args.run, split=args.split, mode=args.mode, zoom=args.zoom if SAHI else None, overlap=args.overlap if SAHI else None,
                   flip_tta=FLIP, tiles_per_image=len(windows(3840, 2160, args.zoom, args.overlap)) if SAHI else 0,
                   checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_epoch=ck.get('epoch'), checkpoint_sha256=ck_sha,
                   annotation_sha256=hashlib.sha256(ANN.read_bytes()).hexdigest(), images=len(images),
                   seconds=seconds, ms_per_image=1000*seconds/max(1, len(images)),
                   merge=f'NMS IoU {IOU}, max_det {MAX_DET}', precision='FP32, TF32 off', device=str(dev))
    (out/f'{name}.json').write_text(json.dumps(metrics, indent=1), encoding='utf-8')
    save_raw(out/f'{name}_raw.npz', {str(im['id']): raw[str(im['id'])] for im in images})
    pct = lambda k: None if metrics[k] is None else round(metrics[k]*100, 1)
    print(json.dumps(dict(run=args.run, split=args.split, mode=tag, AP50=pct('AP50'), AP=pct('AP'), AP75=pct('AP75'),
                          AP_small=pct('AP_small'), ms_per_image=round(metrics['ms_per_image']), minutes=round(seconds/60, 1))), flush=True)


if __name__ == '__main__':
    main()
