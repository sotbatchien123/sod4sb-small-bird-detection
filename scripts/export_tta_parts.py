"""Split a TTA raw file of eval_sahi_tta.py into COCO-format predictions per pass, plus the WBF merge of both.

  <name>_plain_predictions.json : full image only (same boxes as mode 'full')
  <name>_flip_predictions.json  : flipped full image, boxes flipped back
  <name>_wbf_predictions.json   : WBF of both passes (merge_tune 'wbf', voters 2)
Image ids come from the raw file; category id 0 (MVA bird). Works for val and test raw files.

python scripts/export_tta_parts.py reports/sahi_tta/slicing_m_e30_tta_raw.npz
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_tune import wbf


def to_coco(i, b):
    return [dict(image_id=int(i), category_id=0, bbox=[float(x1), float(y1), float(x2-x1), float(y2-y1)], score=float(s))
            for x1, y1, x2, y2, s in b if x2 > x1 and y2 > y1]


def main(path):
    raw = np.load(path); ids = sorted(int(k) for k in raw.files if '__' not in k)
    out = {'plain': [], 'flip': [], 'wbf': []}
    for i in ids:
        b = raw[str(i)]; lens = raw[f'{i}__parts']; assert len(lens) == 2, 'not a full-image TTA raw file'
        plain, flip = b[:lens[0]], b[lens[0]:]
        out['plain'] += to_coco(i, plain); out['flip'] += to_coco(i, flip); out['wbf'] += to_coco(i, wbf(b.copy(), 2))
    stem = str(path).replace('_raw.npz', '')
    for k, v in out.items():
        Path(f'{stem}_{k}_predictions.json').write_text(json.dumps(v), encoding='utf-8'); print(k, len(v), 'boxes')


if __name__ == '__main__':
    main(sys.argv[1])
