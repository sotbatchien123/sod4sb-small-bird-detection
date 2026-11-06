"""Local GFL baseline: random initialization, Drone2021 then MVA. No downloads."""
import argparse
import functools
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_OUTPUT = None
sys.path.insert(0, str(ROOT/'third_party/UniverseNet'))
os.environ.setdefault('TORCH_HOME', str(ROOT/'.cache/torch'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))

def save_json(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    tmp.replace(path)

def hold_after_drone(out):
    """Persistent user hold: only explicit removal authorizes entering MVA."""
    if not (out/'PAUSE_AFTER_DRONE').exists():
        return False
    save_json(out/'status.json',dict(status='paused_after_drone',stage='drone',
        next_stage='mva',reason='User requested rest before MVA',
        updated_at=time.strftime('%Y-%m-%d %H:%M:%S')))
    print('Drone stage complete. Waiting for explicit permission to start MVA.',flush=True)
    return True

def hold_after_mva_epoch(out, completed_epochs):
    marker=out/'PAUSE_AFTER_MVA_EPOCH'
    if not marker.exists(): return False
    limit=int(marker.read_text(encoding='utf-8-sig').strip())
    if limit<1: raise ValueError('MVA pause epoch must be positive')
    if completed_epochs<limit: return False
    save_json(out/'status.json',dict(status='paused_after_mva_epoch',stage='mva',
        completed_epochs=completed_epochs,pause_after_epoch=limit,
        reason='User requested trial then pause; checkpoint and validation saved',
        updated_at=time.strftime('%Y-%m-%d %H:%M:%S')))
    print(f'MVA completed {completed_epochs} epochs. Waiting for explicit continuation.',flush=True)
    return True

def main():
    global ACTIVE_OUTPUT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['smoke','train'],default='smoke')
    p.add_argument('--short-side',type=int,default=2160)
    p.add_argument('--drone-epochs',type=int,default=12)
    p.add_argument('--mva-epochs',type=int,default=7)
    p.add_argument('--accumulate',type=int,default=8)
    p.add_argument('--output',default='work_dirs/gfl_baseline')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--extend-mva-epochs',action='store_true',help='Allow only an increased MVA epoch limit on resume; preserve all learning state.')
    p.add_argument('--mva-per-image-bn',action='store_true',help='Use per-image BN consistently for MVA train/eval, batch size one; preserve checkpoint buffers unused.')
    p.add_argument('--no-activation-checkpointing',action='store_true')
    p.add_argument('--smoke-images',type=int,default=4)
    p.add_argument('--max-hours',type=float,default=None,help='Stop at a saved optimizer boundary when this process reaches the time budget.')
    p.add_argument('--checkpoint-seconds',type=float,default=120)
    args=p.parse_args()
    process_started=time.monotonic()
    if args.max_hours is not None:
        assert args.max_hours>0
    assert args.accumulate>0 and args.drone_epochs>=0 and args.mva_epochs>0
    assert args.checkpoint_seconds>0
    out=(ROOT/args.output).resolve(); assert ROOT in out.parents
    out.mkdir(parents=True,exist_ok=True)
    import msvcrt
    run_lock=(out/'run.lock').open('a+b')
    run_lock.seek(0); run_lock.write(b'0'); run_lock.flush(); run_lock.seek(0)
    msvcrt.locking(run_lock.fileno(),msvcrt.LK_NBLCK,1)
    ACTIVE_OUTPUT=out
    if args.max_hours is not None:
        save_json(out/'time_budget.json',{'hours':args.max_hours,'started_local':time.strftime('%Y-%m-%d %H:%M:%S'),
            'behavior':'Save and stop at optimizer boundary; incomplete epoch is not reported as completed. Evaluation is deferred if budget is exhausted.'})
    if args.mode=='train' and (out/'latest.pth').exists() and not args.resume:
        raise RuntimeError('Existing run: use --resume or a different output directory')
    import numpy as np
    import torch
    from amp_recovery import recover_gradients
    from training_memory import capture_rng, replay_recipe, memory_snapshot
    from gpu_compat import install
    install()
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate
    from mmcv.runner import wrap_fp16_model
    from mmdet.datasets import build_dataset, build_dataloader
    from mmdet.models import build_detector
    from mmdet.apis import single_gpu_test
    from torch.utils.data import DataLoader, Subset

    torch.set_num_threads(6)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark=False
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    cfg=Config.fromfile(str(ROOT/'configs/gfl_local_cpu.py'))
    cfg.model.backbone.init_cfg=None
    cfg.model.backbone.frozen_stages=-1  # Randomly initialized layers must learn.
    cfg.model.backbone.norm_eval=False
    cfg.model.backbone.with_cp=not args.no_activation_checkpointing
    cfg.load_from=None; cfg.resume_from=None; cfg.fp16=None
    scale=(round(args.short_side*16/9),args.short_side)
    for t in cfg.data.train.pipeline:
        if t.type=='Resize': t.img_scale=scale
    for split in ('val','test'):
        cfg.data[split].pipeline[1].img_scale=scale
        cfg.data[split].pipeline[1].flip=False
    cfg.dump(str(out/'model_config.py'))
    roots={'drone':ROOT/'drone2021','mva':ROOT/'mva2023_sod4bird_train'}
    sources={}
    for name,root in roots.items():
        for split in ('train','val'):
            f=root/'annotations'/f'split_{split}_coco.json'
            sources[name+'_'+split]=hashlib.sha256(f.read_bytes()).hexdigest()
    spec={'model':'GFL ResNet50 FPN','initialization':'random; no external weights',
          'stages':[{'name':'drone','epochs':args.drone_epochs,'lr':0.005},
                    {'name':'mva','epochs':args.mva_epochs,'lr':0.001 if args.drone_epochs else 0.005}],
          'image_scale':scale,'batch_size':1,'accumulate':args.accumulate,
          'seed':42,'source_annotation_sha256':sources,'filter_empty_gt':False,
          'activation_checkpointing':not args.no_activation_checkpointing,
          'augmentation':'horizontal flip 0.5 only; no slicing, mosaic or TTA',
          'evaluation':'MVA original 879-image validation after each MVA epoch; COCO AP/AP50/AP75',
          'checkpoint_selection':'best MVA validation AP50; also report last epoch',
          'scope':'Local from-scratch baseline, not a reproduction of author COCO-pretrained scores'}
    if args.drone_epochs==6:
        spec['stages'][0]['lr_milestones_zero_based']=[4,5]
    if args.mva_epochs==200:
        spec['stages'][1]['lr_milestones_zero_based']=[140,180]
    if args.mva_epochs==50:
        spec['stages'][1]['lr_milestones_zero_based']=[35,45]
    if args.mva_per_image_bn:
        spec['stages'][1]['normalization']='per-image BatchNorm; batch_size=1; running buffers unused; train/eval identical'
    cfg.local_bn_policy_by_stage={s['name']:s.get('normalization','standard BatchNorm') for s in spec['stages']}
    cfg.dump(str(out/'model_config.py'))
    if args.resume:
        previous_spec=json.loads((out/'experiment.json').read_text(encoding='utf-8'))
        requested_spec=json.loads(json.dumps(spec))
        if args.extend_mva_epochs:
            comparison=json.loads(json.dumps(previous_spec))
            old_epochs=comparison['stages'][1]['epochs']
            assert args.mva_epochs>old_epochs, 'Extension must increase the epoch limit'
            comparison['stages'][1]['epochs']=args.mva_epochs
            assert comparison==requested_spec, 'Only the MVA epoch limit may change'
            save_json(out/f'experiment_before_extension_{old_epochs}epochs.json',previous_spec)
        else:
            assert previous_spec==requested_spec, 'Resume settings changed'
    else:
        save_json(out/'experiment.json',spec)
    model=build_detector(cfg.model)
    model.init_weights()
    model.CLASSES=('bird',)
    model.cuda()
    wrap_fp16_model(model)
    parallel=MMDataParallel(model,device_ids=[0])
    scaler=torch.amp.GradScaler('cuda',init_scale=128)
    state=None
    recovered_from_previous=False
    if args.resume:
        try:
            state=torch.load(out/'latest.pth',map_location='cpu',weights_only=True)
        except Exception as exc:
            state=torch.load(out/'latest.previous.pth',map_location='cpu',weights_only=True)
            recovered_from_previous=True
            save_json(out/'checkpoint_recovery.json',{'source':'latest.previous.pth','latest_error':str(exc),'time':time.strftime('%Y-%m-%d %H:%M:%S')})
        assert json.loads(json.dumps(state['meta']['experiment']))==previous_spec, 'Checkpoint and experiment configuration differ'
        model.load_state_dict(state['state_dict'],strict=True)
        scaler.load_state_dict(state['scaler'])
        if args.extend_mva_epochs:
            save_json(out/'extension.json',{'from_mva_epochs':old_epochs,'to_mva_epochs':args.mva_epochs,
                'checkpoint_stage':state['stage'],'completed_epochs':state['epoch'],'offset':state['offset'],
                'optimizer_updates_preserved':state['updates'],'optimizer_and_rng_restored':True})
            save_json(out/'experiment.json',spec)
    started=time.time(); total_images=0; best_ap=-1.; smoke_losses=[]
    val_cfg=cfg.data.val.copy()
    val_cfg.update(ann_file=str(roots['mva']/'annotations/split_val_coco.json'),
                   img_prefix=str(roots['mva']/'images')+'/',test_mode=True,classes=('bird',))
    validation=None
    if args.mode=='train':
        validation=build_dataset(val_cfg)
        assert len(validation)==879

    last_checkpoint_time=time.monotonic()
    def checkpoint(stage, epoch, offset, optimizer, updates):
        nonlocal last_checkpoint_time, recovered_from_previous
        value={'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
               'scaler':scaler.state_dict(),'stage':stage,'epoch':epoch,'offset':offset,
               'updates':updates,'best_ap50':best_ap,'python_rng':random.getstate(),
               'numpy_rng':(np.random.get_state()[0],np.random.get_state()[1].tolist(),
                            *np.random.get_state()[2:]),
               'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
               'meta':{'CLASSES':('bird',),'experiment':spec}}
        with (out/'latest.tmp').open('wb') as checkpoint_file:
            torch.save(value,checkpoint_file)
            checkpoint_file.flush(); os.fsync(checkpoint_file.fileno())
        if (out/'latest.pth').exists() and not recovered_from_previous:
            backup_tmp=out/'latest.previous.tmp'
            if backup_tmp.exists(): backup_tmp.unlink()
            os.link(out/'latest.pth',backup_tmp)
            backup_tmp.replace(out/'latest.previous.pth')
        (out/'latest.tmp').replace(out/'latest.pth')
        recovered_from_previous=False
        last_checkpoint_time=time.monotonic()
        save_json(out/'checkpoint_status.json',{'stage':stages[stage]['name'],'epoch_index':epoch,
            'images_in_epoch':offset,'optimizer_updates':updates,'saved_local':time.strftime('%Y-%m-%d %H:%M:%S')})

    stages=spec['stages']
    if args.mode=='smoke': stages=[dict(name='mva',epochs=1,lr=0.001)]
    for stage_idx,stage in enumerate(stages):
        name=stage['name']
        if stage['epochs']==0: continue
        if state and stage_idx<state['stage']: continue
        if name=='mva' and hold_after_drone(out): return
        if name=='mva' and args.mva_per_image_bn:
            from per_image_bn import enable_per_image_bn
            print('MVA per-image BN enabled:',enable_per_image_bn(model),'layers',flush=True)
        train_cfg=cfg.data.train.copy()
        train_cfg.update(ann_file=str(roots[name]/'annotations/split_train_coco.json'),
                         img_prefix=str(roots[name]/'images')+'/',classes=('bird',),filter_empty_gt=False)
        dataset=build_dataset(train_cfg)
        expected=51119 if name=='drone' else 8880
        assert len(dataset)==expected
        optimizer=torch.optim.SGD(model.parameters(),lr=stage['lr'],momentum=0.9,weight_decay=0.0001)
        epoch_start=0; offset_start=0; updates=0
        if state and stage_idx==state['stage']:
            optimizer.load_state_dict(state['optimizer'])
            epoch_start=state['epoch']; offset_start=state['offset']; updates=state['updates']; best_ap=state['best_ap50']
            random.setstate(state['python_rng'])
            r=state['numpy_rng']; np.random.set_state((r[0],np.array(r[1],dtype=np.uint32),*r[2:]))
            torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state(state['cuda_rng'])
            if args.extend_mva_epochs:
                checkpoint(stage_idx,epoch_start,offset_start,optimizer,updates)
        # Optimizer tensors and RNG have been restored; release the CPU copy.
        state=None
        gc.collect()
        for epoch in range(epoch_start,stage['epochs']):
            if name=='mva' and hold_after_mva_epoch(out,epoch): return
            offset=offset_start if epoch==epoch_start else 0
            status={'status':'running','stage':name,'epoch':epoch+1,'epochs':stage['epochs'],
                    'images_in_epoch':offset,'epoch_images':len(dataset),'optimizer_updates':updates,
                    'normalization':stage.get('normalization','standard BatchNorm'),
                    'updated_at':time.strftime('%Y-%m-%d %H:%M:%S')}
            save_json(out/'status.json',status)
            if (out/'STOP').exists():
                checkpoint(stage_idx,epoch,offset,optimizer,updates)
                status['status']='stopped_at_checkpoint';save_json(out/'status.json',status); return
            indices=np.random.RandomState(42+stage_idx*1000+epoch).permutation(len(dataset)).tolist()
            if args.mode=='smoke':
                positives=[i for i in range(len(dataset)) if len(dataset.get_ann_info(i)['bboxes'])]
                negatives=[i for i in range(len(dataset)) if not len(dataset.get_ann_info(i)['bboxes'])]
                indices=[positives[i//2] if i%2==0 else negatives[i//2] for i in range(args.smoke_images)]
            size=len(indices)
            model.train(); optimizer.zero_grad(set_to_none=True)
            group_count=0
            group_batches=[]
            epoch_start_time=time.time(); initial_offset=offset
            for dataset_index in indices[offset:]:
                recipe=(dataset_index,capture_rng())
                batch=collate([dataset[dataset_index]],samples_per_gpu=1)
                if group_count==0:
                    group_buffers={k:v.detach().clone() for k,v in model.named_buffers()}
                    group_rng=(torch.get_rng_state(),torch.cuda.get_rng_state())
                    group_batches=[]
                group_batches.append(recipe)
                group_start=offset-(offset % args.accumulate)
                divisor=min(args.accumulate,size-group_start)
                losses=parallel(return_loss=True,**batch)
                loss,log=model._parse_losses(losses)
                if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
                scaler.scale(loss/divisor).backward()
                offset+=1; total_images+=1; group_count+=1
                smoke_losses.append(float(loss.detach())) if args.mode=='smoke' else None
                del batch,losses,loss
                if group_count==divisor:
                    def replay_index(index):
                        retry_batch=collate([dataset[index]],samples_per_gpu=1)
                        retry_losses=parallel(return_loss=True,**retry_batch)
                        retry_loss,_=model._parse_losses(retry_losses)
                        if not torch.isfinite(retry_loss): raise RuntimeError('Non-finite loss during AMP retry')
                        scaler.scale(retry_loss/divisor).backward()
                    def replay(recipe):
                        replay_recipe(recipe,replay_index)
                    def record_retry(attempt,old_scale,new_scale):
                        event=dict(stage=name,epoch=epoch+1,images_in_epoch=offset,
                            optimizer_updates=updates,attempt=attempt,old_scale=old_scale,
                            new_scale=new_scale,time=time.strftime('%Y-%m-%d %H:%M:%S'))
                        with (out/'amp_retries.jsonl').open('a',encoding='utf-8') as f:
                            f.write(json.dumps(event)+'\n')
                        print('AMP_RETRY '+json.dumps(event),flush=True)
                    grad=recover_gradients(model,optimizer,scaler,group_batches,replay,
                        group_buffers,group_rng,record_retry)
                    warmup=min(1.,(updates+1)/500.)
                    milestones=stage.get('lr_milestones_zero_based',[8,11] if name=='drone' else [6])
                    decay=(0.1**sum(epoch>=s for s in milestones))
                    for group in optimizer.param_groups: group['lr']=stage['lr']*warmup*decay
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                    group_count=0; group_batches=[]; updates+=1
                    elapsed=time.time()-started
                    status={'status':'running','stage':name,'epoch':epoch+1,'epochs':stage['epochs'],
                            'images_in_epoch':offset,'epoch_images':size,'optimizer_updates':updates,
                            'normalization':stage.get('normalization','standard BatchNorm'),
                            'last_losses':log,'learning_rate':optimizer.param_groups[0]['lr'],
                            'elapsed_seconds_this_process':elapsed,'seconds_per_image':elapsed/total_images,
                            'training_seconds_per_image':(time.time()-epoch_start_time)/(offset-initial_offset),
                            'gradient_norm_before_clip':float(grad),
                            'peak_gpu_mib':torch.cuda.max_memory_allocated()/1024**2,
                            'host_memory':memory_snapshot(),
                            'updated_at':time.strftime('%Y-%m-%d %H:%M:%S')}
                    save_json(out/'status.json',status)
                    if updates%10==0 or updates==1: print(json.dumps(status),flush=True)
                    budget_expired=args.max_hours is not None and time.monotonic()-process_started>=args.max_hours*3600
                    if args.mode=='train' and (updates==1 or updates%50==0 or time.monotonic()-last_checkpoint_time>=args.checkpoint_seconds or (out/'STOP').exists() or budget_expired):
                        checkpoint(stage_idx,epoch,offset,optimizer,updates)
                        gc.collect()
                        if budget_expired:
                            status.update(status='time_budget_reached',last_completed_epoch=epoch,evaluation_of_partial_epoch_pending=True)
                            save_json(out/'status.json',status); return
                        if (out/'STOP').exists():
                            status['status']='stopped_at_checkpoint';save_json(out/'status.json',status); return
            if args.mode=='smoke':
                checkpoint(stage_idx,epoch,offset,optimizer,updates)
                # Exercise serialization plus actual inference after positive/negative training.
                saved=torch.load(out/'latest.pth',map_location='cpu',weights_only=True)
                model.load_state_dict(saved['state_dict'],strict=True)
                small_annotation=json.loads(Path(val_cfg.ann_file).read_text())
                small_annotation['images']=small_annotation['images'][:2]
                sample_ids={i['id'] for i in small_annotation['images']}
                small_annotation['annotations']=[a for a in small_annotation['annotations'] if a['image_id'] in sample_ids]
                save_json(out/'smoke_validation.coco.json',small_annotation)
                small_cfg=val_cfg.copy(); small_cfg.ann_file=str(out/'smoke_validation.coco.json')
                small_val=build_dataset(small_cfg)
                tiny=DataLoader(small_val,batch_size=1,num_workers=0,
                    collate_fn=functools.partial(collate,samples_per_gpu=1))
                model.eval()
                results=[]
                with torch.no_grad():
                    for b in tiny: results.extend(parallel(return_loss=False,rescale=True,**b))
                metrics=small_val.evaluate(results,metric='bbox',proposal_nums=(1,10,100))
                assert 'bbox_mAP' in metrics, 'COCO evaluation did not return metrics'
                save_json(out/'smoke_metrics_not_benchmark.json',metrics)
                smoke=dict(status='passed',train_images=args.smoke_images,losses=smoke_losses,
                    optimizer_updates=updates,inference_images=2,checkpoint_reload=True,
                    seconds_per_training_image=status['training_seconds_per_image'],peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2)
                save_json(out/'smoke.json',smoke)
                save_json(out/'status.json',smoke)
                print('SMOKE PASSED',flush=True); return
            checkpoint(stage_idx,epoch,offset,optimizer,updates)
            if name=='mva' or (name=='drone' and epoch+1==stage['epochs']):
                model.eval()
                eval_dataset=validation
                if name=='drone':
                    drone_val_cfg=val_cfg.copy()
                    drone_val_cfg.update(ann_file=str(roots['drone']/'annotations/split_val_coco.json'),img_prefix=str(roots['drone']/'images')+'/')
                    eval_dataset=build_dataset(drone_val_cfg)
                    assert len(eval_dataset)==13572
                status.update(status='evaluating',validation_images=len(eval_dataset),validation_images_done=0)
                save_json(out/'status.json',status)
                val_loader=build_dataloader(eval_dataset,samples_per_gpu=1,workers_per_gpu=0,dist=False,shuffle=False)
                predictions=[]
                with torch.no_grad():
                    for val_batch in val_loader:
                        if (out/'STOP').exists():
                            checkpoint(stage_idx,epoch,offset,optimizer,updates)
                            status.update(status='stopped_at_checkpoint',evaluation_pending=True)
                            save_json(out/'status.json',status); return
                        predictions.extend(parallel(return_loss=False,rescale=True,**val_batch))
                        if len(predictions)%50==0:
                            status.update(validation_images_done=len(predictions),updated_at=time.strftime('%Y-%m-%d %H:%M:%S'))
                            save_json(out/'status.json',status)
                prefix=str(out/f'{name}_epoch_{epoch+1}')
                eval_dataset.format_results(predictions,jsonfile_prefix=prefix)
                metrics=eval_dataset.evaluate(predictions,metric='bbox',proposal_nums=(1,10,100))
                save_json(out/f'{name}_epoch_{epoch+1}_metrics.json',metrics)
                (out/'results.md').write_text(
                    '# GFL baseline\n\n'
                    f'Stage: {name}; epoch: {epoch+1}. Train: {len(dataset)} images; validation: {len(eval_dataset)} images. Input: {scale[0]} x {scale[1]}.\n\n'
                    'Random initialization; no downloaded model weights. See experiment.json for the training stages.\n\n'
                    f'AP: {100*metrics["bbox_mAP"]:.2f}%; AP50: {100*metrics["bbox_mAP_50"]:.2f}%; AP75: {100*metrics["bbox_mAP_75"]:.2f}%.\n\n'
                    'This is a local baseline. A one-epoch result does not establish convergence or reproduce the paper.\n',encoding='utf-8')
                if name=='mva' and metrics['bbox_mAP_50']>best_ap:
                    best_ap=metrics['bbox_mAP_50']
                    torch.save({'state_dict':model.state_dict(),'meta':{'CLASSES':('bird',),'epoch':epoch+1,'metrics':metrics,'experiment':spec,'normalization':stage.get('normalization','standard BatchNorm')}},out/'best_mva.pth')
                del predictions
            checkpoint(stage_idx,epoch+1,0,optimizer,updates)
            if name=='mva' and hold_after_mva_epoch(out,epoch+1): return
            if args.max_hours is not None and time.monotonic()-process_started>=args.max_hours*3600:
                save_json(out/'status.json',dict(status='time_budget_reached',stage=name,last_completed_epoch=epoch+1,partial_epoch=False,best_mva_AP50=best_ap)); return
        if name=='drone':
            torch.save({'state_dict':model.state_dict(),'meta':{'CLASSES':('bird',),'experiment':spec}},out/'drone_pretrain.pth')
            if hold_after_drone(out): return
        state=None
    save_json(out/'status.json',dict(status='complete',best_mva_AP50=best_ap,elapsed_seconds_this_process=time.time()-started))

if __name__=='__main__':
    try: main()
    except Exception:
        traceback.print_exc()
        if ACTIVE_OUTPUT is not None:
            save_json(ACTIVE_OUTPUT/'failure.json',{'status':'failed','error':traceback.format_exc(),'time':time.strftime('%Y-%m-%d %H:%M:%S')})
            save_json(ACTIVE_OUTPUT/'status.json',{'status':'failed','details':'failure.json','resume_checkpoint':'latest.pth if present'})
        sys.exit(1)
