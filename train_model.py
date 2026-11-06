"""Train the two detectors of the final system, step by step. Each step calls the script in scripts/ with the settings
used in the paper (YOLO recipes in configs/yolo_improvements/, GFL settings recorded in configs/gfl_r50_fpn_ep38/).

    python train_model.py yolo-pretrain              # 1. YOLOv8-M on Drone2021, 10 epochs (MVA runs start from epoch 7)
    python train_model.py yolo --method slicing      # 2. YOLOv8-M on MVA, 30 epochs (also: baseline, augmentation, copy_paste)
    python train_model.py gfl                        # 3. GFL R50-FPN: Drone2021 6 epochs, then MVA 50 epochs (paper: epoch 38)
    python train_model.py all --dry-run              # print the commands only

All runs are resumable: rerun the same command after an interruption. One 8 GB GPU is enough.
"""
import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METHODS = ('baseline', 'augmentation', 'copy_paste', 'slicing')


def commands(step, method):
    py = sys.executable
    yolo_pretrain = [py, 'scripts/train_yolo_drone.py', '--size', 'm']
    yolo = [py, 'scripts/train_yolo_data_methods.py', '--model', 'yolo', '--recipe', f'configs/yolo_improvements/{method}_m_e30.json']
    gfl = [py, 'scripts/train_gfl_baseline.py', '--mode', 'train', '--short-side', '1440', '--drone-epochs', '6',
           '--mva-epochs', '50', '--mva-per-image-bn', '--no-activation-checkpointing', '--output', 'work_dirs/gfl_r50_fpn']
    return {'yolo-pretrain': [yolo_pretrain], 'yolo': [yolo], 'gfl': [gfl], 'all': [yolo_pretrain, yolo, gfl]}[step]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('yolo-pretrain', 'yolo', 'gfl', 'all'))
    p.add_argument('--method', choices=METHODS, default='slicing', help='MVA data strategy for the yolo step')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    for cmd in commands(args.step, args.method):
        print('>', ' '.join(Path(c).name if c == sys.executable else c for c in cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)
