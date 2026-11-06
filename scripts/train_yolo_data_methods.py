"""Independent YOLO data-method fine-tunes on MVA (Augmentation, Slicing, Copy-Paste).

New runs start from the completed Drone2021 pretrain anchor with fresh optimizer/schedule. They
never publish over baseline status or use its output directories. GPU smoke test
and full validation are required before declaring this new runner production-ready.

Version 4 (30/10/2026): physical batch 8 of native-resolution 1280x736 views, matching the Drone
pretrain v2. Batch 1 trained BatchNorm on per-image statistics and evaluated on running
statistics. Validation runs on full native images padded to a multiple of 32.
"""
import os, sys, json, time, math, random, hashlib, argparse, traceback, copy, contextlib, io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'.yolo-deps'))
os.environ['YOLO_CONFIG_DIR'] = str(ROOT/'.cache/yolo')
os.environ['MPLCONFIGDIR'] = str(ROOT/'.cache/matplotlib')
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
import cv2
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import ModelEMA
from ultralytics.utils.ops import non_max_suppression
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from atomic_status_io import atomic_json
from paths import rel
from yolo_scale_views import xyxy
from yolo_pilot_views import PilotViews, VIEW

# Physical batch by recipe model_size, no accumulation. M uses 6: at 8 VRAM sits at the 8 GB limit and runs ~35% slower.
BATCHES = {'s': 8, 'm': 6}

PARENTS = {'yolo': 'yolov8s_parallel_to8_1440', 'swin': 'happyday_swin_s_mva5_1440'}
START = {'yolo': 8, 'swin': 5}
KEYS = ['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']

# YOLOv8 -> YOLOv8-P2 layer indices. Backbone and the top-down path to P3 (0-15) are shared; the P2
# branch inserts layers 16-21, so the bottom-up P3->P5 path moves by 6 and Detect gains level 0.
P2_SHIFT = {16: 22, 17: 23, 18: 24, 19: 25, 20: 26, 21: 27}

def p2_key(key):
    """Name in the P2 model of a YOLOv8 state_dict key, or None if P2 has no counterpart."""
    parts = key.split('.')
    layer = int(parts[1])
    if layer <= 15: return key
    if layer in P2_SHIFT: return '.'.join([parts[0], str(P2_SHIFT[layer])]+parts[2:])
    if layer == 22:  # Detect: per-level cv2/cv3 lists move from (P3,P4,P5) to (P2,P3,P4,P5)
        if parts[2] in ('cv2', 'cv3'): return '.'.join([parts[0], '28', parts[2], str(int(parts[3])+1)]+parts[4:])
        return '.'.join([parts[0], '28']+parts[2:])
    return None

def transfer_to_p2(model, weights):
    """Copy every anchor tensor with a same-shape counterpart; new P2 layers keep their random init.
    Detect's class branch (cv3) is sized from the first level's channels (P2 here, P3 in the anchor),
    so it cannot be copied and is the only anchor part allowed to stay random."""
    target = model.state_dict(); copied = set(); skipped = []
    for key, value in weights.items():
        new = p2_key(key)
        if new is None or new not in target or target[new].shape != value.shape: skipped.append(key); continue
        target[new] = value.clone(); copied.add(new)
    unexpected = [k for k in skipped if not k.startswith('model.22.cv3.')]
    if unexpected: raise RuntimeError(f'Anchor tensors without a P2 counterpart: {unexpected[:5]}')
    model.load_state_dict(target, strict=True)
    return dict(copied=len(copied), skipped_class_branch=len(skipped), random=sorted(set(target)-copied))

def build_model(name, size='s', architecture='base'):
    if name == 'yolo':
        model = DetectionModel(f'yolov8{size}{"-p2" if architecture == "p2" else ""}.yaml', nc=1, verbose=False)
        model.args = get_cfg()
    else:
        from happyday_model_fixed50 import HappyDayModel
        model = HappyDayModel()
    return model.cuda()

def grouped_optimizer(model, name, legacy=None):
    """Separate decay by tensor role, migrating each parameter's Adam moments."""
    if legacy is not None:
        old = torch.optim.AdamW(model.parameters())
        old.load_state_dict(legacy)
    decay, no_decay = [], []
    for key, p in model.named_parameters():
        if not p.requires_grad: continue
        excluded = p.ndim <= 1 or key.endswith('.bias') or 'absolute_pos_embed' in key or 'relative_position_bias_table' in key
        (no_decay if excluded else decay).append(p)
    wd = .0005 if name == 'yolo' else .05
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': wd},
                            {'params': no_decay, 'weight_decay': 0.0}], lr=1e-4)
    assert len(decay)+len(no_decay) == sum(p.requires_grad for p in model.parameters())
    assert len({id(p) for p in decay+no_decay}) == len(decay)+len(no_decay)
    if legacy is not None:
        for p in model.parameters():
            if p in old.state: opt.state[p] = old.state[p]
        del old
    return opt

def replace_retry(src, dst):
    for attempt in range(100):
        try: os.replace(src, dst); return
        except PermissionError:
            if attempt == 99: raise
            time.sleep(.1)

def atomic_torch(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    replace_retry(temp, path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['yolo','swin'], required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--stop-after-updates', type=int, default=0)
    p.add_argument('--publish-status', action='store_true')
    p.add_argument('--recipe', required=True)
    p.add_argument('--allow-shared-gpu', action='store_true')
    args = p.parse_args()
    if args.model != 'yolo' or args.publish_status:
        raise ValueError('This runner is YOLO-only; baseline status publishing forbidden')
    config_path = ROOT/args.recipe
    recipe = json.loads(config_path.read_text(encoding='utf-8'))
    if recipe['view_size'] != list(VIEW):
        raise ValueError('Unsupported view geometry')
    if recipe['method'] in ('slicing','combined') and not 0 <= recipe['overlap'] < 1:
        raise ValueError('Invalid slicing overlap')
    args.epochs = recipe['epochs']
    size = recipe.get('model_size', 's')
    if size not in BATCHES: raise ValueError('model_size must be s or m')
    architecture = recipe.get('architecture', 'base')
    if architecture not in ('base', 'p2'): raise ValueError('architecture must be base or p2')
    # P2 needs more VRAM than the plain model; its recipe sets the measured batch explicitly.
    BATCH = recipe.get('batch', BATCHES[size])
    if not 1 <= args.epochs <= 100:
        raise ValueError('Each method is capped at 100 epochs')
    out = (ROOT/recipe['output']).resolve()
    experiment_root = (ROOT/'work_dirs/yolo_improvements').resolve()
    if experiment_root not in out.parents:
        raise ValueError('Output must be a child of work_dirs/yolo_improvements')
    if not args.allow_shared_gpu:
        for alias_name in ('status_yolo.json','status_swin.json','work_dirs/yolo_drone_pretrain/status.json','work_dirs/yolo_drone_m_pretrain/status.json'):
            path = ROOT/alias_name
            if path.exists():
                state = json.loads(path.read_text())
                try: fresh = time.time()-time.mktime(time.strptime(state['updated_at'],'%Y-%m-%d %H:%M:%S')) < 180
                except (KeyError,ValueError): fresh = True
                if fresh and state.get('status') in ('running','evaluating','starting'):
                    raise RuntimeError('Baseline active; defer trial, or explicitly use --allow-shared-gpu')
    out.mkdir(parents=True, exist_ok=True)
    import msvcrt
    lock = (out/'run.lock').open('a+b'); lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    torch.set_num_threads(3); cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Prevent idle sleep while working; this does not block an explicit shutdown.
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 unavailable')
    data = ROOT/'mva2023_sod4bird_train'
    annroot = ROOT/'data/mva_audited_v1/annotations'
    anns, by_image, hashes = {}, {}, {}
    for split in ('train','val'):
        path = annroot/f'split_{split}_coco.json'
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        anns[split] = json.loads(path.read_text(encoding='utf-8'))
        by_image[split] = {}
        for a in anns[split]['annotations']:
            if not a.get('iscrowd',0): by_image[split].setdefault(a['image_id'], []).append(a)
    views = PilotViews(data, anns['train'], recipe)
    train_images = anns['train']['images'][:args.limit or None]
    val_images = anns['val']['images'][:args.limit or None]
    # Slicing: every tile keeping a bird plus one clean background tile per source per epoch.
    plan = [views.tiles(im) for im in train_images] if views.uses_tiles else None
    def epoch_samples(e):
        if plan is None: return [(i,None,0) for i in range(len(train_images))]
        samples=[]
        for i,(positive,clean) in enumerate(plan):
            samples+=[(i,t,k) for k,t in enumerate(positive)]
            if clean:
                pick=np.random.RandomState([7,e,int(train_images[i]['id'])]).randint(len(clean))
                samples.append((i,clean[pick],len(positive)))
        return samples
    epoch_len = len(epoch_samples(0))
    source = ROOT/recipe['anchor']
    if not source.exists():
        raise RuntimeError('Freeze the chosen anchor first; do not use a live latest.pt')
    # A best.pt beside a pretraining status is live until that run completes.
    anchor_status = source.parent/'status.json'
    if anchor_status.exists() and json.loads(anchor_status.read_text()).get('status') != 'complete':
        raise RuntimeError('Anchor pretraining is not complete; wait, or copy best.pt to a frozen path')
    spec = dict(version=4, donor_index_sha256=views.donor_hash, donor_count=len(views.donors), model=args.model, epochs=args.epochs, input=list(VIEW),
        parent_checkpoint=rel(source), parent_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        parent_epochs=0, annotation_sha256=hashes, limit=args.limit, recipe=recipe,
        implementation_sha256={f:hashlib.sha256((ROOT/'scripts'/f).read_bytes()).hexdigest() for f in ('train_yolo_data_methods.py','yolo_scale_views.py','yolo_pilot_views.py')},
        train_images=len(train_images), samples_per_epoch=epoch_len, val_images=len(val_images), seed=42,
        initialization='Local frozen EMA anchor; fresh AdamW, EMA update count and branch epoch; not scratch training',
        optimizer='AdamW decay excludes bias/norm; reset Adam moments equally across every arm',
        schedule='Continuation cosine 1e-4 to 1e-5; 100-update warmup from 1e-5',
        batch=BATCH, accumulate=1, loss_objective='Ultralytics batch loss divided by batch size',
        BN_policy=f'YOLO BatchNorm on physical batch {BATCH}; EMA evaluation without recalibration',
        precision='BF16 forward; FP32 loss/assignment and parameters; no GradScaler; eval FP32, TF32 off; deterministic algorithms',
        augmentation=('Independent data method in recipe; native 1280x736 views, boxes kept if >=50% of area remains; '
            'augmentation/copy_paste one bird-centred view per source, slicing all bird tiles + one clean tile per source; '
            'common flip .5 and brightness .8..1.2'),
        evaluation='EMA FP32 on native images padded to a multiple of 32; clean COCO maxDets100 primary; original-label100 and clean300 diagnostic',
        centernet_collision='deterministic smallest-box regression priority; structural collisions remain',
        limitations='original image split; scene independence unproven; slicing costs more updates per epoch; paper-guided not exact author recipe')
    if (out/'experiment.json').exists():
        if json.loads((out/'experiment.json').read_text()) != spec: raise RuntimeError('Experiment mismatch')
    else: atomic_json(out/'experiment.json',spec)
    model = build_model(args.model, size, architecture); ema = ModelEMA(model)
    latest = out/'latest.pt'; continuing = latest.exists()
    # Restore previous only if latest is missing (e.g. interrupted rotation).
    if not continuing and (out/'latest.previous.pt').exists():
        import shutil
        shutil.copy2(out/'latest.previous.pt',latest); continuing=True
    ck = torch.load(latest if continuing else source, map_location='cpu', weights_only=False)
    if continuing:
        if ck['spec'] != spec: raise RuntimeError('Checkpoint protocol mismatch')
    else:
        weights=ck.get('ema',ck['model'])
        if architecture=='p2' and any(k.startswith('model.28.') for k in weights):
            # Anchor is already a YOLOv8-P2 model (Drone P2 pretrain): load it as is.
            print('P2 init: anchor is already a P2 model; loaded without transfer',flush=True)
        elif architecture=='p2':
            transfer=transfer_to_p2(model,weights); weights=model.state_dict()
            print(f'P2 init from anchor: {transfer["copied"]} tensors copied, {transfer["skipped_class_branch"]} class-branch '
                  f'tensors not copyable, {len(transfer["random"])} random',flush=True)
        ck=dict(model=weights,ema=weights,ema_updates=0,epoch=0,offset=0,updates=0,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    model.load_state_dict(ck['model'], strict=True); ema.ema.load_state_dict(ck['ema'], strict=True)
    ema.updates=ck['ema_updates']
    optimizer = grouped_optimizer(model,args.model)
    if continuing: optimizer.load_state_dict(ck['optimizer'])
    epoch, offset = ck['epoch'], ck['offset']
    attempted = ck.get('attempted_updates',ck['updates'])
    successful = ck.get('successful_updates',ck['ema_updates'])
    skipped = ck.get('skipped_updates',attempted-successful)
    branch_updates = ck.get('branch_updates',0) if continuing else 0
    best = ck['best'] if continuing else -1.0
    loss_sum, loss_count = (ck['loss_sum'],ck['loss_count']) if continuing else (0.0,0)
    active_prior = ck.get('active_seconds',0.0) if continuing else 0.0
    peak_prior = ck.get('peak_tensor_allocated_mib_all_sessions',0.0) if continuing else 0.0
    random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng']); torch.set_rng_state(ck['torch_rng'])
    torch.cuda.set_rng_state_all(ck['cuda_rng']); del ck
    started=saved=time.time(); initial_attempts=attempted; processed=0
    alias=out/'status.json'
    print(f'YOLO ablation {recipe["name"]}: resume {epoch}:{offset}, target {args.epochs}',flush=True)

    def status(state,**kw):
        record=dict(status=state,model=args.model,pid=os.getpid(),epoch=min(epoch+1,args.epochs),
            completed_epochs=epoch,epochs=args.epochs,images_in_epoch=offset,epoch_images=epoch_len,
            updates=successful,attempted_updates=attempted,successful_updates=successful,skipped_updates=skipped,
            historical_skips=0,branch_updates=branch_updates,
            elapsed_seconds=time.time()-started,active_seconds=active_prior+time.time()-started,
            processed_this_process=processed,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_tensor_allocated_mib_all_sessions=max(peak_prior,torch.cuda.max_memory_allocated()/2**20),
            tensor_allocated_mib=torch.cuda.memory_allocated()/2**20,tensor_reserved_mib=torch.cuda.memory_reserved()/2**20,
            peak_tensor_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            progress_percent=round(100*(epoch+offset/epoch_len)/args.epochs,3),
            precision=spec['precision'],source_status=rel(out/'status.json'),
            updated_at=time.strftime('%Y-%m-%d %H:%M:%S'),**kw)
        atomic_json(out/'status.json',record)
        if args.publish_status: atomic_json(alias,record)

    def save():
        nonlocal saved
        state=dict(spec=spec,model=model.state_dict(),optimizer=optimizer.state_dict(),scaler={},
            ema=ema.ema.state_dict(),ema_updates=ema.updates,epoch=epoch,offset=offset,
            updates=attempted,attempted_updates=attempted,successful_updates=successful,skipped_updates=skipped,
            branch_updates=branch_updates,best=best,loss_sum=loss_sum,loss_count=loss_count,
            peak_tensor_allocated_mib_all_sessions=max(peak_prior,torch.cuda.max_memory_allocated()/2**20),
            active_seconds=active_prior+time.time()-started,python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        staged=out/'next.pt'; atomic_torch(staged,state)
        if latest.exists(): replace_retry(latest,out/'latest.previous.pt')
        replace_retry(staged,latest); saved=time.time()

    def read(im):
        raw=cv2.imread(str(data/'images'/im['file_name']))
        if raw is None or raw.shape[:2] != (im['height'],im['width']): raise RuntimeError('Invalid image '+im['file_name'])
        if raw.shape[:2]!=(2160,3840): raise RuntimeError('Unexpected aspect ratio')
        return raw

    def make_view(i,tile,k,e):
        im=train_images[i]
        # Largest first, smallest last; deterministic ordering.
        ordered=sorted(by_image['train'].get(im['id'],[]),key=lambda a:(-a['bbox'][2]*a['bbox'][3],a['id']))
        return views(read(im),xyxy(ordered),[42,e,int(im['id']),k],im['id'],tile)

    def submit_reads(samples):
        # Decode and build each view in the pool; per-sample seeds keep results order-independent.
        return [pool.submit(make_view,i,tile,k,epoch) for i,tile,k in samples]

    def train_batch(samples,views_done):
        canvases,labels=[],[]
        for job in views_done:
            canvas,boxes,view_info=job.result()
            view_stats['source_visits']+=1
            view_stats['retained_boxes']+=view_info['retained_boxes']
            view_stats['dropped_boxes']+=view_info['dropped_boxes']
            view_stats['pasted_objects']+=view_info['pasted_count']
            view_stats['photometric_views']+=int(view_info['photometric_applied'])
            view_stats['background_views']+=int(view_info['retained_boxes']==0)
            canvases.append(canvas); labels.append(boxes)
        stack=np.stack(canvases)[:,:,:,::-1].transpose(0,3,1,2)
        img=torch.from_numpy(np.ascontiguousarray(stack)).cuda().float()/255
        box=torch.from_numpy(np.concatenate(labels)).cuda()
        idx=torch.cat([torch.full((len(b),),j,dtype=torch.float32) for j,b in enumerate(labels)]).cuda()
        return dict(img=img,bboxes=box,cls=torch.zeros((len(box),1),device='cuda'),batch_idx=idx)

    def val_tensor(raw):
        h,w=raw.shape[:2]
        canvas=np.full((math.ceil(h/32)*32,math.ceil(w/32)*32,3),114,dtype=np.uint8); canvas[:h,:w]=raw
        return torch.from_numpy(np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1))).cuda().float()[None]/255

    def coco_metrics(preds,path,maxdet):
        with contextlib.redirect_stdout(io.StringIO()):
            gt=COCO(str(path))
            if not preds: return {k:0.0 for k in KEYS}
            ev=COCOeval(gt,gt.loadRes(preds),'bbox');ev.params.imgIds=[im['id'] for im in val_images]
            ev.params.maxDets=[1,10,maxdet];ev.evaluate();ev.accumulate();ev.summarize()
            values=ev.stats.tolist()
            # COCO's default summarize hardcodes AP maxDets=100; extract AP for custom cap.
            if maxdet != 100:
                pr=ev.eval['precision'][:,:,:,0,2]; values[0]=float(pr[pr>-1].mean()) if (pr>-1).any() else -1.
        return dict(zip(KEYS,values))

    def evaluate():
        status('evaluating',validation_images_done=0);torch.cuda.empty_cache()
        preds=[]; em=ema.ema.eval(); cat=anns['val']['categories'][0]['id']
        if args.model=='swin':
            from happyday_model_fixed50 import decode_predictions
            nms=decode_predictions
        else: nms=non_max_suppression
        with torch.inference_mode():
            pending=pool.submit(read,val_images[0]) if val_images else None
            for i,im in enumerate(val_images):
                raw=pending.result()
                pending=pool.submit(read,val_images[i+1]) if i+1<len(val_images) else None
                with torch.autocast('cuda',enabled=False): prediction=em(val_tensor(raw))
                detection=nms(prediction,conf_thres=.001,iou_thres=.6,max_det=300)[0].float().cpu().numpy()
                for x1,y1,x2,y2,score,cls in detection:
                    # Native-resolution input: padding is right/bottom only, so no rescaling.
                    x1,x2=np.clip([x1,x2],0,im['width']);y1,y2=np.clip([y1,y2],0,im['height'])
                    if not np.isfinite([x1,y1,x2,y2,score]).all(): raise RuntimeError('Nonfinite prediction')
                    if x2>x1 and y2>y1: preds.append(dict(image_id=im['id'],category_id=cat,bbox=[float(x1),float(y1),float(x2-x1),float(y2-y1)],score=float(score)))
                if (i+1)%50==0:status('evaluating',validation_images_done=i+1)
        atomic_json(out/f'epoch_{epoch+1}_predictions.json',preds)
        metrics=coco_metrics(preds,annroot/'split_val_coco.json',100)
        metrics['original_labels_100']=coco_metrics(preds,data/'annotations/split_val_coco.json',100)
        dense=coco_metrics(preds,annroot/'split_val_coco.json',300)
        metrics['dense_300']={k.replace('AR100','AR300'):v for k,v in dense.items()}
        metrics.update(train_loss_mean=loss_sum/max(1,loss_count),active_seconds=active_prior+time.time()-started,
            view_statistics_this_process=dict(view_stats),
            precision='FP32 TF32 disabled',annotation_sha256=hashes['val'],successful_updates=successful,
            skipped_updates=skipped,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20)
        atomic_json(out/f'epoch_{epoch+1}_metrics.json',metrics)
        print(json.dumps(dict(epoch=epoch+1,AP=metrics['AP'],AP50=metrics['AP50'],AP75=metrics['AP75'])),flush=True)
        torch.cuda.empty_cache(); return metrics

    view_stats=dict(source_visits=0,retained_boxes=0,dropped_boxes=0,pasted_objects=0,photometric_views=0,background_views=0)
    # JPEG decoding of 3840x2160 sources dominates I/O; threads release the GIL inside OpenCV.
    # The next batch is decoded and its views built while the GPU runs the current one.
    pool=ThreadPoolExecutor(10)
    try:
        # Resume keeps latest.previous as a distinct rollback point.
        if not continuing: save()
        status('starting')
        while epoch < args.epochs:
            if (out/'STOP').exists():save();status('stopped_at_checkpoint');return
            model.train(); optimizer.zero_grad(set_to_none=True)
            samples=epoch_samples(epoch)
            if len(samples)!=epoch_len: raise RuntimeError('Sample plan changed between epochs')
            order=np.random.RandomState(42+epoch).permutation(len(samples))
            batch_at=lambda start:[samples[int(i)] for i in order[start:start+BATCH]]
            prefetch=submit_reads(batch_at(offset)) if offset<len(order) else None
            while offset<len(order):
                group=min(BATCH,len(order)-offset)
                fraction=(epoch+offset/len(order))/args.epochs
                lr=(1e-5+9e-5*(1+math.cos(math.pi*fraction))/2)*(.1+.9*min(1,(branch_updates+1)/100))
                for g in optimizer.param_groups:g['lr']=lr
                current=prefetch
                prefetch=submit_reads(batch_at(offset+group)) if offset+group<len(order) else None
                batch=train_batch(batch_at(offset),current)
                with torch.autocast('cuda',dtype=torch.bfloat16): prediction=model(batch['img'])
                # Assignment and reductions in FP32, outside mixed precision.
                with torch.autocast('cuda',enabled=False):loss,items=model.loss(batch,[v.float() for v in prediction])
                loss=loss/group  # Ultralytics returns the batch loss multiplied by batch size
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss; resume previous safe boundary')
                loss.backward();loss_sum+=float(loss.detach())*group;loss_count+=group
                parts=[[float(v) for v in items.detach()]]
                gradnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
                optimizer.step();ema.update(model);optimizer.zero_grad(set_to_none=True)
                offset+=group;processed+=group;attempted+=1;successful+=1;branch_updates+=1
                status('running',learning_rate=lr,train_loss_mean=loss_sum/loss_count,
                    grad_norm=float(gradnorm),loss_components=np.mean(parts,axis=0).tolist(),last_checkpoint_age_seconds=time.time()-saved)
                stop=(out/'STOP').exists() or (args.stop_after_updates and attempted-initial_attempts>=args.stop_after_updates)
                if stop or time.time()-saved>120:save()
                if stop:status('stopped_at_checkpoint');return
            save();metrics=evaluate()
            if metrics['AP50']>best:
                best=metrics['AP50'];atomic_torch(out/'best.pt',dict(model=ema.ema.state_dict(),spec=spec,epoch=epoch+1,metrics=metrics))
            epoch+=1;offset=0;loss_sum=0.;loss_count=0;save()
        status('complete',best_AP50=best)
    except Exception:
        atomic_json(out/'failure.json',dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),traceback=traceback.format_exc()))
        status('failed');raise

if __name__=='__main__':main()
