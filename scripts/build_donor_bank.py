"""Pre-cut every Copy-Paste donor patch once, for the Kaggle runs.

PilotViews.donor_image decodes the full 3840x2160 donor source for every pasted bird (cache of 2), which
is cheap on the local 24-thread CPU but would dominate Kaggle's ~4 vCPUs. The bank stores the exact
decoded pixels of each donor window (lossless), in the same order and with the same donor_hash as
PilotViews, so pasting from the bank gives bit-identical views.

python scripts/build_donor_bank.py [--out kaggle/donor_bank] [--threads 6]
"""
import argparse, hashlib, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'.yolo-deps')); sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
import cv2
from yolo_pilot_views import PilotViews

ANNOTATION = 'data/mva_audited_v1/annotations/split_train_coco.json'
IMAGES = 'mva2023_sod4bird_train'
RECIPE = 'configs/yolo_improvements/copy_paste_m_e30.json'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='kaggle/donor_bank')
    p.add_argument('--threads', type=int, default=6)
    args = p.parse_args()
    cv2.setNumThreads(1)
    ann_path = ROOT/ANNOTATION
    ann = json.loads(ann_path.read_text(encoding='utf-8'))
    recipe = json.loads((ROOT/RECIPE).read_text(encoding='utf-8'))
    views = PilotViews(ROOT/IMAGES, ann, recipe)
    donors = views.donors
    by_image = {}
    for i, d in enumerate(donors): by_image.setdefault(d['image_id'], []).append(i)

    def cut(image_id):
        im = views.images[image_id]
        raw = cv2.imread(str(ROOT/IMAGES/'images'/im['file_name']))
        if raw is None or raw.shape[:2] != (im['height'], im['width']): raise RuntimeError('Invalid image '+im['file_name'])
        out = []
        for i in by_image[image_id]:
            x0, y0, x1, y1 = donors[i]['window']
            out.append((i, np.ascontiguousarray(raw[y0:y1, x0:x1])))
        return out

    patches = [None]*len(donors)
    with ThreadPoolExecutor(args.threads) as pool:
        for k, result in enumerate(pool.map(cut, sorted(by_image))):
            for i, patch in result: patches[i] = patch
            if (k+1) % 1000 == 0: print(f'{k+1}/{len(by_image)} source images', flush=True)
    offsets = np.cumsum([0]+[p.size for p in patches])
    flat = np.concatenate([p.reshape(-1) for p in patches])
    out = ROOT/args.out; out.mkdir(parents=True, exist_ok=True)
    np.save(out/'patches.npy', flat)
    index = dict(
        donor_hash=views.donor_hash, donor_count=len(donors), source_images=len(by_image),
        annotation=ANNOTATION, annotation_sha256=hashlib.sha256(ann_path.read_bytes()).hexdigest(), recipe=RECIPE,
        donor_parameters={k: recipe[k] for k in ('max_donor_side', 'patch_margin')},
        layout='patches.npy is one flat uint8 BGR buffer; donor i is flat[offset[i]:offset[i+1]].reshape(h, w, 3)',
        patches_sha256=hashlib.sha256((out/'patches.npy').read_bytes()).hexdigest(),
        donors=[dict(d, offset=int(offsets[i]), h=int(patches[i].shape[0]), w=int(patches[i].shape[1])) for i, d in enumerate(donors)])
    (out/'index.json').write_text(json.dumps(index), encoding='utf-8')
    # Check: a sample of bank patches equals a fresh decode-and-crop, as PilotViews.donor_image would give.
    rng = np.random.RandomState(0)
    for i in rng.choice(len(donors), size=min(300, len(donors)), replace=False):
        d = donors[int(i)]; im = views.images[d['image_id']]
        raw = cv2.imread(str(ROOT/IMAGES/'images'/im['file_name']))
        x0, y0, x1, y1 = d['window']
        bank = flat[offsets[i]:offsets[i+1]].reshape(y1-y0, x1-x0, 3)
        assert np.array_equal(bank, raw[y0:y1, x0:x1]), d
    print(f'{len(donors)} donors from {len(by_image)} images, {flat.nbytes/2**20:.1f} MiB; '
          f'donor_hash {views.donor_hash[:12]}; 300 sampled patches identical to a fresh crop', flush=True)


if __name__ == '__main__':
    main()
