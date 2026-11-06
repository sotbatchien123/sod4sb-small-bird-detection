"""Final public-test predictions: one-stage WBF of three sources with equal weights, IoU 0.55
(the configuration chosen on val: reports/ensemble/gfl+plain+flip.json, rule wbf_i0.55_w1, AP50 84.5%).

Sources (same order as on val): GFL epoch 38, YOLO Slicing M plain pass, YOLO Slicing M flipped pass.
Every test image id of the empty-annotation file gets an entry list (possibly empty). Output: COCO results JSON.

python scripts/make_submission.py
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble_wbf import load, wbf

ROOT = Path(__file__).resolve().parents[1]
TEST_ANN = ROOT/'mva2023_sod4bird_pub_test/annotations/public_test_coco_empty_ann.json'
SOURCES = [ROOT/'reports/gfl_predict/gfl_ep38_test_predictions.json',
           ROOT/'reports/sahi_tta/slicing_m_e30_tta_test_plain_predictions.json',
           ROOT/'reports/sahi_tta/slicing_m_e30_tta_test_flip_predictions.json']
OUT = ROOT/'reports/submission'


def main():
    ann = json.loads(TEST_ANN.read_text(encoding='utf-8')); ids = [im['id'] for im in ann['images']]
    size = {im['id']: (im['width'], im['height']) for im in ann['images']}
    cat = ann['categories'][0]['id']
    models = [load(p) for p in SOURCES]
    for p, m in zip(SOURCES, models): print(p.name, 'images with boxes:', len(m))
    preds, empty = [], 0
    for i in ids:
        b = wbf([m.get(i, np.zeros((0, 5), np.float32)) for m in models], [1., 1., 1.], .55)
        # clip to the image (WBF averaging / GFL rescale can overshoot a few px); val AP unchanged by clipping
        W, H = size[i]
        if len(b): b = b.copy(); b[:, [0, 2]] = b[:, [0, 2]].clip(0, W); b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
        empty += not len(b)
        preds += [dict(image_id=i, category_id=cat, bbox=[float(x1), float(y1), float(x2-x1), float(y2-y1)], score=float(s))
                  for x1, y1, x2, y2, s in b if x2 > x1 and y2 > y1]
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT/'final_ensemble_test_predictions.json'
    out.write_text(json.dumps(preds), encoding='utf-8')
    info = dict(images=len(ids), boxes=len(preds), images_without_boxes=empty, rule='WBF IoU 0.55, weights 1:1:1',
                sources=[str(p.relative_to(ROOT)) for p in SOURCES], val_AP50_of_rule=0.845)
    (OUT/'final_ensemble_test_info.json').write_text(json.dumps(info, indent=1), encoding='utf-8')
    print(json.dumps(info))


if __name__ == '__main__':
    main()
