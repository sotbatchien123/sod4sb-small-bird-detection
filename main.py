"""Run the final SOD4SB detector (YOLOv8-M slicing + flip TTA + GFL, one WBF) and evaluate it.

    python main.py                                   # MVA validation split: predictions + COCO AP50 / AP75 / AR100
    python main.py --ann none --images path/to/frames --out results.json   # any folder: predictions only
    python main.py --limit 20                        # quick check on the first 20 validation images
"""
import argparse, json, time
from pathlib import Path

from evaluate import evaluate, report
from load_data import list_images
from model import Detector

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    # environmental setting: change the paths below to match your data and weights
    p = argparse.ArgumentParser()
    p.add_argument('--yolo', default='weights/yolov8m_slicing_e30.pt')
    p.add_argument('--gfl', default='weights/gfl_r50_fpn_ep38.pth')
    p.add_argument('--ann', default='data/mva_audited_v1/annotations/split_val_coco.json')
    p.add_argument('--images', default='mva2023_sod4bird_train/images')
    p.add_argument('--out', default='results/val_predictions.json')
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()

    det = Detector(ROOT/args.yolo, ROOT/args.gfl)
    ann_path = ROOT/args.ann
    if args.ann != 'none' and ann_path.exists():   # evaluation mode: images listed in the annotation file
        images = json.loads(ann_path.read_text(encoding='utf-8'))['images']
    else:   # folder mode: every jpg/png, ids in file order
        images, ann_path = list_images(ROOT/args.images), None
    images = images[:args.limit or None]

    preds, t0 = [], time.time()
    for k, im in enumerate(images, 1):
        for x1, y1, x2, y2, s in det(ROOT/args.images/im['file_name']):
            preds.append(dict(image_id=im['id'], category_id=0, bbox=[float(x1), float(y1), float(x2-x1), float(y2-y1)], score=float(s)))
        if k % 50 == 0: print(f'{k}/{len(images)} images, {(time.time()-t0)/k*1000:.0f} ms/image', flush=True)
    out = ROOT/args.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(preds), encoding='utf-8')
    print(f'{len(preds)} boxes on {len(images)} images -> {args.out}')

    if ann_path is not None:
        print(report(evaluate(preds, ann_path, [im['id'] for im in images])))
