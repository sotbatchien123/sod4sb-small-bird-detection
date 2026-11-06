"""GFL (ResNet-50 FPN) inference with a saved checkpoint, reproducing the trainer's validation exactly:
model from the run's model_config.py, wrap_fp16_model, per-image BatchNorm (if the run used it), the
config's test pipeline (2560x1440, no flip), batch 1, rescale to original pixels.

  --split val  : MVA validation (audited annotation, same 879 images) -> predictions + COCO metrics
  --split test : MVA public test (9,699 images, no labels)           -> predictions only
  --limit N    : first N images (checks)
Writes reports/gfl_predict/<name>_predictions.json (COCO results, category 0) and <name>.json (metrics, ms/image).

python -u scripts/run_qos.py scripts/gfl_predict.py --split val --limit 20
"""
import argparse, contextlib, hashlib, io, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'third_party/UniverseNet')); sys.path.insert(0, str(ROOT/'scripts'))
os.environ.setdefault('TORCH_HOME', str(ROOT/'.cache/torch'))

SPLITS = {'val': (ROOT/'data/mva_audited_v1/annotations/split_val_coco.json', ROOT/'mva2023_sod4bird_train/images'),
          'test': (ROOT/'mva2023_sod4bird_pub_test/annotations/public_test_coco_empty_ann.json', ROOT/'mva2023_sod4bird_pub_test/images')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='work_dirs/gfl_drone12_mva7_1440')
    p.add_argument('--checkpoint', default='best_mva.pth')
    p.add_argument('--split', choices=('val', 'test'), default='val')
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    import numpy as np, torch
    from gpu_compat import install
    install()
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import wrap_fp16_model
    from mmdet.datasets import build_dataset, build_dataloader
    from mmdet.models import build_detector
    from torch.utils.data import Subset

    run = ROOT/args.run; ckpt = run/args.checkpoint
    exp = json.loads((run/'experiment.json').read_text(encoding='utf-8'))
    per_image_bn = 'per-image' in exp['stages'][-1].get('normalization', '')
    cfg = Config.fromfile(str(run/'model_config.py'))
    cfg.model.backbone.init_cfg = None
    ann, images = SPLITS[args.split]
    data_cfg = cfg.data.val.copy()
    data_cfg.update(ann_file=str(ann), img_prefix=str(images)+'/', test_mode=True, classes=('bird',))
    dataset = build_dataset(data_cfg)
    torch.cuda.set_device(0)
    model = build_detector(cfg.model); model.CLASSES = ('bird',)
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(state['state_dict'], strict=True)
    model.cuda(); wrap_fp16_model(model)
    if per_image_bn:
        from per_image_bn import enable_per_image_bn
        enable_per_image_bn(model)
    model.eval(); parallel = MMDataParallel(model, device_ids=[0])
    subset = Subset(dataset, range(args.limit)) if args.limit else dataset
    # workers_per_gpu=0 as in the trainer: worker processes hang on this Windows setup
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
    results, t0 = [], time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.limit and i >= args.limit: break
            results.extend(parallel(return_loss=False, rescale=True, **batch))
            if (i+1) % 200 == 0: print(f'{i+1}/{len(subset)} images, {time.time()-t0:.0f}s', flush=True)
    seconds = time.time()-t0
    preds = []
    for idx, r in enumerate(results):
        image_id = dataset.data_infos[idx]['id']; boxes = r[0]
        for x1, y1, x2, y2, s in boxes:
            preds.append(dict(image_id=int(image_id), category_id=0, bbox=[float(x1), float(y1), float(x2-x1), float(y2-y1)], score=float(s)))
    out = ROOT/'reports/gfl_predict'; out.mkdir(parents=True, exist_ok=True)
    name = f'gfl_ep{state["meta"].get("epoch")}_{args.split}' + (f'_limit{args.limit}' if args.limit else '')
    (out/f'{name}_predictions.json').write_text(json.dumps(preds), encoding='utf-8')
    info = dict(checkpoint=str(ckpt.relative_to(ROOT)), checkpoint_epoch=state['meta'].get('epoch'),
                checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(), per_image_bn=per_image_bn,
                image_scale=exp['image_scale'], split=args.split, images=len(results), boxes=len(preds),
                seconds=seconds, ms_per_image=1000*seconds/max(1, len(results)))
    if args.split == 'val' and preds:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(str(ann)); ev = COCOeval(gt, gt.loadRes(preds), 'bbox')
            ev.params.imgIds = [dataset.data_infos[i]['id'] for i in range(len(results))]; ev.params.maxDets = [1, 10, 100]
            ev.evaluate(); ev.accumulate(); ev.summarize()
        info.update(AP=ev.stats[0], AP50=ev.stats[1], AP75=ev.stats[2], AP_small=ev.stats[3])
    (out/f'{name}.json').write_text(json.dumps(info, indent=1), encoding='utf-8')
    print(json.dumps({k: (round(v*100, 1) if k.startswith('AP') else v) for k, v in info.items() if k != 'checkpoint_sha256'}), flush=True)


if __name__ == '__main__':
    main()
