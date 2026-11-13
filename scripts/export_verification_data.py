"""Pack the saved validation predictions used in the paper into one small file for the verification notebook.

Sources: 'baseline' and 'slicing' are the full-frame runs of the two YOLOv8-M models (rows of Tables III and VII);
'plain' and 'flip' are the two views of the flip-TTA run of the slicing model, which feed the flip TTA and the final
ensemble; 'gfl' is GFL R50-FPN at epoch 38. ('slicing' and 'plain' are two GPU runs of the same model and differ by a
few boxes.) Each source is a COCO result list (image_id, bbox=[x, y, w, h], score); it is stored as a float64 array
with columns image_id, x, y, w, h, score in results/verification/val_predictions.npz, so the notebook can recompute
every number without a GPU, images or weights.

    python scripts/export_verification_data.py \
        --baseline ../reports/sahi_tta/baseline_m_e30_full_predictions.json \
        --slicing  ../reports/sahi_tta/slicing_m_e30_full_predictions.json \
        --plain    ../reports/sahi_tta/slicing_m_e30_tta_plain_predictions.json \
        --flip     ../reports/sahi_tta/slicing_m_e30_tta_flip_predictions.json \
        --gfl      ../work_dirs/gfl_drone12_mva7_1440/mva_epoch_38.bbox.json
"""
import argparse, hashlib, json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/verification'


def to_array(path):
    preds = json.loads(Path(path).read_text(encoding='utf-8'))
    return np.array([[p['image_id'], *p['bbox'], p['score']] for p in preds], dtype=np.float64).reshape(-1, 6)


def main():
    p = argparse.ArgumentParser()
    for name in ('baseline', 'slicing', 'plain', 'flip', 'gfl'):
        p.add_argument(f'--{name}', required=True)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    arrays, info = {}, {}
    for name in ('baseline', 'slicing', 'plain', 'flip', 'gfl'):
        src = Path(getattr(args, name))
        arrays[name] = to_array(src)
        info[name] = dict(file=src.name, sha256=hashlib.sha256(src.read_bytes()).hexdigest(), boxes=len(arrays[name]))
    np.savez_compressed(OUT/'val_predictions.npz', **arrays)
    (OUT/'val_predictions_sources.json').write_text(json.dumps(info, indent=1) + '\n', encoding='utf-8')
    print({k: v['boxes'] for k, v in info.items()}, (OUT/'val_predictions.npz').stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
