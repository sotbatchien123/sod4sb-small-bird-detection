"""Drone2021 YOLOv8-S/M random-init baseline, fixed 10 epochs, resumable.

Recipe v2: physical batch 8 of native-resolution 1280x736 crops (one crop per source image, so an
epoch still visits every image once). Batch 1 made BatchNorm train on per-image statistics and
evaluate on running statistics; v2 removes that mismatch. Validation runs on full native images.
YOLOv8-M uses physical batch 6 (~5.9 GB VRAM). Batch 8 (~7.6 GB reserved, 7.86/8.15 GB in use) ran at
14.3 img/s against 22.7 in isolation, likely from VRAM pressure, so the user chose 6 on 01/11/2026.

--arch p2 (with --size m): YOLOv8m-P2 continued from the finished M pretrain (layers transferred as in
train_yolo_data_methods.transfer_to_p2), 5 epochs (user decision 04/11/2026; cosine schedule over 5) at a lower
learning rate, stopping early as soon as an epoch's AP falls below the best so far (user decision 03/11/2026).
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
from yolo_scale_views import xyxy, intersect
from drone_baseline_data import image_path

CROP = (1280, 736)  # width, height; both multiples of 32
BATCHES = {'s': 8, 'm': 6}  # physical batch per model size, no accumulation
P2_EPOCHS = 5
P2_BATCH = 4  # YOLOv8m-P2: stride-4 head needs ~40-60% more activation memory than M at batch 6
P2_ANCHOR = 'work_dirs/yolo_drone_m_pretrain/best.pt'
KEEP_FRACTION = .5  # a box cut by the crop edge is kept if at least half its area remains
KEYS =['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']

def build_model(size, arch='base'):
    model=DetectionModel(f'yolov8{size}{"-p2" if arch == "p2" else ""}.yaml',nc=1,verbose=False)
    model.args=get_cfg()
    return model.cuda()


def grouped_optimizer(model,name):
    decay=[]; no_decay=[]
    for key,p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim<=1 or key.endswith('.bias') else decay).append(p)
    return torch.optim.AdamW([{'params':decay,'weight_decay':.0005},
                             {'params':no_decay,'weight_decay':0.}],lr=.001)


def crop_start(rng,lo_edge,hi_edge,size,crop):
    """Window start so that [lo_edge, hi_edge] lies inside the crop; any overlap if the box is larger."""
    lo,hi=max(0.,hi_edge-crop),min(lo_edge,size-crop)
    if lo>hi: lo,hi=hi,lo
    hi=int(math.floor(min(max(hi,0),size-crop))); lo=min(int(math.ceil(min(max(lo,0),size-crop))),hi)
    return int(rng.randint(lo,hi+1))


def crop_view(raw,boxes,rng,crop=CROP):
    """Native-resolution crop around one random box (uniform if none), then flip p=0.5.
    boxes are xyxy pixels; returns the crop, kept boxes as normalized xywh, and the window."""
    h,w=raw.shape[:2]; cw,ch=crop
    if len(boxes):
        b=boxes[rng.randint(len(boxes))]
        x0=crop_start(rng,b[0],b[2],w,cw); y0=crop_start(rng,b[1],b[3],h,ch)
    else:
        x0=int(rng.randint(0,w-cw+1)); y0=int(rng.randint(0,h-ch+1))
    window=np.array([x0,y0,x0+cw,y0+ch],dtype=np.float32)
    clipped,area=intersect(boxes,window)
    keep=(area>0)&(area>=KEEP_FRACTION*(boxes[:,2:]-boxes[:,:2]).prod(axis=1))
    kept=clipped[keep]-window[[0,1,0,1]]
    canvas=raw[y0:y0+ch,x0:x0+cw]
    flip=bool(rng.rand()<.5)
    if flip: canvas=canvas[:,::-1]; kept[:,[0,2]]=cw-kept[:,[2,0]]
    norm=np.empty_like(kept)
    norm[:,:2]=(kept[:,:2]+kept[:,2:])/2; norm[:,2:]=kept[:,2:]-kept[:,:2]
    norm/=np.array([cw,ch,cw,ch],dtype=np.float32)
    return canvas,norm,dict(window=window.tolist(),flip=flip,keep=keep)


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
    p.add_argument('--size', choices=('s','m'), default='s')
    p.add_argument('--arch', choices=('base','p2'), default='base')
    p.add_argument('--output', default=None)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--stop-after-updates', type=int, default=0)
    args = p.parse_args()
    args.model='yolo'; args.epochs=10; args.publish_status=False
    P2=args.arch=='p2'
    if P2 and args.size!='m': raise ValueError('P2 pretraining is defined for YOLOv8-M only')
    if P2: args.epochs=P2_EPOCHS
    default_output={'s':'work_dirs/yolo_drone_pretrain','m':'work_dirs/yolo_drone_m_pretrain'}[args.size]
    if P2: default_output='work_dirs/yolo_drone_m_p2_pretrain'
    args.output=args.output or default_output
    BATCH=P2_BATCH if P2 else BATCHES[args.size]
    if args.limit < 0: raise ValueError('Invalid limit')
    if args.limit and args.output == default_output: raise ValueError('Subset tests need separate output')
    out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'work_dirs') or not out.name.startswith('yolo_drone_'):
        raise ValueError('Output must be a dedicated work_dirs/yolo_drone_* directory')
    recipe={'name':Path(default_output).name,'epochs':args.epochs}
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
    data = ROOT/'drone2021'
    annroot = data/'annotations'
    anns, by_image, hashes = {}, {}, {}
    for split in ('train','val'):
        path = annroot/f'split_{split}_coco.json'
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        anns[split] = json.loads(path.read_text(encoding='utf-8'))
        by_image[split] = {}
        for a in anns[split]['annotations']:
            if not a.get('iscrowd',0): by_image[split].setdefault(a['image_id'], []).append(a)
    def selected(split):
        images=anns[split]['images']
        if not args.limit: return images
        positives=[im for im in images if by_image[split].get(im['id'])]
        negatives=[im for im in images if not by_image[split].get(im['id'])]
        chosen=positives[:max(1,args.limit//2)]+negatives[:args.limit//2]
        ids={im['id'] for im in chosen}
        return (chosen+[im for im in images if im['id'] not in ids])[:args.limit]
    train_images=selected('train')
    val_images=selected('val')
    if not args.limit:
        audit=json.loads((ROOT/'reports/yolo_drone_pretrain/data_audit.json').read_text(encoding='utf-8'))
        if not audit['passed'] or audit['annotation_sha256'] != hashes: raise RuntimeError('Dataset audit missing or stale')
    spec=dict(version=2,model=f'YOLOv8-{args.size.upper()}',epochs=10,input=list(CROP),seed=42,
        initialization='random; no external weights; Drone2021 only',annotation_sha256=hashes,
        train_images=len(train_images),val_images=len(val_images),limit=args.limit,
        optimizer='AdamW lr 0.001 to 0.0001 cosine; no bias/norm decay; warmup 100 updates',
        batch=BATCH,accumulate=1,loss_objective='Ultralytics batch loss divided by batch size',
        precision='BF16 forward; FP32 loss and EMA evaluation; deterministic; TF32 off',
        augmentation=('one native-resolution 1280x736 crop per image per epoch: images with birds are cropped to '
            'contain one randomly chosen bird at a random position, images without birds uniformly; boxes kept if '
            '>=50% of area remains; horizontal flip p=0.5; no resize, paste, mosaic or color augmentation'),
        evaluation='original Drone validation at native resolution padded to a multiple of 32; COCO maxDets100; best by AP50-95',
        implementation_sha256={f:hashlib.sha256((ROOT/'scripts'/f).read_bytes()).hexdigest() for f in ('train_yolo_drone.py','drone_baseline_data.py')})
    if P2:
        anchor=ROOT/P2_ANCHOR
        if json.loads((anchor.parent/'status.json').read_text()).get('status')!='complete': raise RuntimeError('M pretrain is not complete')
        spec.update(model='YOLOv8-M-P2',epochs=args.epochs,
            initialization=f'{P2_ANCHOR} (YOLOv8-M Drone pretrain, best AP) transferred to YOLOv8m-P2; new P2 layers and Detect class branch random',
            parent_checkpoint=P2_ANCHOR,parent_sha256=hashlib.sha256(anchor.read_bytes()).hexdigest(),
            optimizer=f'AdamW lr 0.0005 to 0.00005 cosine over {args.epochs} epochs; no bias/norm decay; warmup 100 updates',
            early_stopping='full runs stop (status complete) after the first epoch whose AP is below the best so far',
            implementation_sha256={f:hashlib.sha256((ROOT/'scripts'/f).read_bytes()).hexdigest()
                for f in ('train_yolo_drone.py','drone_baseline_data.py','train_yolo_data_methods.py')})
    if (out/'experiment.json').exists():
        if json.loads((out/'experiment.json').read_text()) != spec: raise RuntimeError('Experiment mismatch')
    else: atomic_json(out/'experiment.json',spec)
    model = build_model(args.size,args.arch); ema = ModelEMA(model)
    latest = out/'latest.pt'; continuing = latest.exists()
    # Restore previous only if latest is missing (e.g. interrupted rotation).
    if not continuing and (out/'latest.previous.pt').exists():
        import shutil
        shutil.copy2(out/'latest.previous.pt',latest); continuing=True
    ck = torch.load(latest, map_location='cpu', weights_only=False) if continuing else {}
    if continuing:
        if ck['spec'] != spec: raise RuntimeError('Checkpoint protocol mismatch')
    else:
        weights=model.state_dict()
        if P2:
            from train_yolo_data_methods import transfer_to_p2
            source=torch.load(ROOT/P2_ANCHOR,map_location='cpu',weights_only=False)['model']
            transfer=transfer_to_p2(model,source); weights=model.state_dict()
            print(f'P2 init from M pretrain: {transfer["copied"]} tensors copied, {len(transfer["random"])} random',flush=True)
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
    print(f'YOLO Drone {recipe["name"]}: resume {epoch}:{offset}, target {args.epochs}',flush=True)

    def status(state,**kw):
        record=dict(status=state,model=args.model,pid=os.getpid(),epoch=min(epoch+1,args.epochs),
            completed_epochs=epoch,epochs=args.epochs,images_in_epoch=offset,epoch_images=len(train_images),
            updates=successful,attempted_updates=attempted,successful_updates=successful,skipped_updates=skipped,
            historical_skips=0,branch_updates=branch_updates,
            elapsed_seconds=time.time()-started,active_seconds=active_prior+time.time()-started,
            processed_this_process=processed,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_tensor_allocated_mib_all_sessions=max(peak_prior,torch.cuda.max_memory_allocated()/2**20),
            tensor_allocated_mib=torch.cuda.memory_allocated()/2**20,tensor_reserved_mib=torch.cuda.memory_reserved()/2**20,
            peak_tensor_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            progress_percent=round(100*(epoch+offset/len(train_images))/args.epochs,3),
            precision=spec['precision'],source_status=str(out/'status.json'),
            updated_at=time.strftime('%Y-%m-%d %H:%M:%S'),**kw)
        atomic_json(out/'status.json',record)
        atomic_json(out/'progress.json',record)
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
        raw=cv2.imread(str(image_path(data,im['file_name'])))
        if raw is None or raw.shape[:2] != (im['height'],im['width']): raise RuntimeError('Invalid image '+im['file_name'])
        if (im['width'],im['height']) not in ((3840,2160),(1920,1080)): raise RuntimeError('Unexpected dimensions')
        return raw

    def train_view(im,e):
        # Runs in a worker thread: read + crop; the per-image RNG makes the result order-independent.
        raw=read(im)
        # Stable ordering of annotations.
        ordered=sorted(by_image['train'].get(im['id'],[]),key=lambda a:(-a['bbox'][2]*a['bbox'][3],a['id']))
        boxes=xyxy(ordered)
        canvas,norm,info=crop_view(raw,boxes,np.random.RandomState(42+e*100000+int(im['id'])))
        if not np.isfinite(norm).all() or not ((norm>=0)&(norm<=1)).all(): raise RuntimeError('Invalid boxes '+im['file_name'])
        return canvas,norm,len(boxes),info['flip']

    def submit_views(ims):
        return [pool.submit(train_view,im,epoch) for im in ims]

    def train_batch(jobs):
        views=[j.result() for j in jobs]
        for _,norm,n,flip in views:
            view_stats['source_visits']+=1; view_stats['object_centred_crops']+=int(n>0)
            view_stats['retained_boxes']+=len(norm); view_stats['dropped_boxes']+=n-len(norm)
            view_stats['flips']+=int(flip)
        stack=np.stack([v[0] for v in views])[:,:,:,::-1].transpose(0,3,1,2)
        img=torch.from_numpy(np.ascontiguousarray(stack)).cuda().float()/255
        box=torch.from_numpy(np.concatenate([v[1] for v in views])).cuda()
        idx=torch.cat([torch.full((len(v[1]),),i,dtype=torch.float32) for i,v in enumerate(views)]).cuda()
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
        nms=non_max_suppression
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
        metrics.update(train_loss_mean=loss_sum/max(1,loss_count),active_seconds=active_prior+time.time()-started,
            view_statistics_this_process=dict(view_stats),
            precision='FP32 TF32 disabled',annotation_sha256=hashes['val'],successful_updates=successful,
            skipped_updates=skipped,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20)
        atomic_json(out/f'epoch_{epoch+1}_metrics.json',metrics)
        print(json.dumps(dict(epoch=epoch+1,AP=metrics['AP'],AP50=metrics['AP50'],AP75=metrics['AP75'])),flush=True)
        torch.cuda.empty_cache(); return metrics

    view_stats=dict(source_visits=0,object_centred_crops=0,retained_boxes=0,dropped_boxes=0,flips=0)
    # JPEG decoding and cropping of 3840x2160 sources dominate; threads release the GIL inside OpenCV,
    # and the next batch is prepared while the GPU trains on the current one.
    pool=ThreadPoolExecutor(10)
    try:
        save(); status('starting')
        while epoch < args.epochs:
            if (out/'STOP').exists():save();status('stopped_at_checkpoint');return
            model.train(); optimizer.zero_grad(set_to_none=True)
            order=np.random.RandomState(42+epoch).permutation(len(train_images))
            batch_at=lambda start:[train_images[int(i)] for i in order[start:start+BATCH]]
            prefetch=submit_views(batch_at(offset)) if offset<len(order) else None
            while offset<len(order):
                group=min(BATCH,len(order)-offset)
                fraction=(epoch+offset/len(order))/args.epochs
                peak,floor=(5e-4,5e-5) if P2 else (1e-3,1e-4)
                lr=(floor+(peak-floor)*(1+math.cos(math.pi*fraction))/2)*(.1+.9*min(1,(branch_updates+1)/100))
                for g in optimizer.param_groups:g['lr']=lr
                current=prefetch
                prefetch=submit_views(batch_at(offset+group)) if offset+group<len(order) else None
                batch=train_batch(current)
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
            declined=metrics['AP']<best
            if metrics['AP']>best:
                best=metrics['AP'];atomic_torch(out/'best.pt',dict(model=ema.ema.state_dict(),spec=spec,epoch=epoch+1,metrics=metrics))
            epoch+=1;offset=0;loss_sum=0.;loss_count=0;save()
            if P2 and declined and not args.limit:
                status('complete',best_AP=best,early_stopped_after_epoch=epoch);return
        status('complete',best_AP=best)
    except Exception:
        atomic_json(out/'failure.json',dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),traceback=traceback.format_exc()))
        status('failed');raise

if __name__=='__main__':main()
