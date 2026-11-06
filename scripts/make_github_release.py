"""Copy the files needed to reproduce the final pipeline into github_release/ (nothing is moved or deleted).

Excluded on purpose: images and competition annotations (licence of the organisers; scripts rebuild the audited
labels), checkpoints and predictions (hundreds of MB; Releases instead), vendored libraries (.yolo-deps,
third_party/UniverseNet: install / clone at the pinned versions), personal notes, other course assignments.

python scripts/make_github_release.py
"""
import glob, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'github_release'

SCRIPTS = ['amp_recovery', 'atomic_status_io', 'build_donor_bank', 'drone_baseline_data', 'ensemble_wbf', 'error_analysis',
           'eval_sahi_tta', 'export_tta_parts', 'gfl_predict', 'gpu_compat', 'make_report_figures', 'make_submission',
           'merge_tune', 'sahi_fixed_metrics', 'per_image_bn', 'prepare_fixed50', 'run_qos', 'train_gfl_baseline', 'train_yolo_data_methods',
           'train_yolo_drone', 'training_memory', 'win_qos', 'yolo_pilot_views', 'yolo_scale_views', 'make_github_release']
FILES = [f'scripts/{s}.py' for s in SCRIPTS] + [
    'scripts/run_test_predictions.ps1', 'scripts/run_submission_after_test.ps1',
    'configs/gfl_local_cpu.py',
    *[f'configs/yolo_improvements/{c}.json' for c in ('baseline_m_e30', 'augmentation_m_e30', 'copy_paste_m_e30', 'slicing_m_e30')],
    'data/mva_audited_v1/manifest.json',
    'requirements-5070.txt',
]
# results: (source glob, destination folder)
RESULTS = [
    ('reports/sahi_tta/*.json', 'results/inference'),
    ('reports/ensemble/*.json', 'results/ensemble'),
    ('reports/error_analysis/*.json', 'results/error_analysis'),
    ('reports/error_analysis/slicing_m_e30_tta_*.png', 'results/error_analysis'),
    ('reports/error_analysis/pretrain_only_m_full_*.png', 'results/error_analysis'),
    ('reports/figures/*.png', 'results/figures'),
    ('reports/gfl_audited/gfl_rescored_audited.json', 'results/gfl'),
    ('reports/gfl_predict/gfl_ep38_val_limit20.json', 'results/gfl'),
    ('reports/submission/final_ensemble_test_info.json', 'results/submission'),
    ('reports/FINAL_SUMMARY.md', 'results'),
]
TRAIN_LOGS = {  # run folder -> files that document how it was trained and its best epoch
    'work_dirs/yolo_drone_m_pretrain': ['experiment.json', 'epoch_7_metrics.json'],
    'work_dirs/yolo_improvements/baseline_m_e30': ['experiment.json', 'epoch_30_metrics.json'],
    'work_dirs/yolo_improvements/augmentation_m_e30': ['experiment.json', 'epoch_30_metrics.json'],
    'work_dirs/yolo_improvements/copy_paste_m_e30': ['experiment.json', 'epoch_30_metrics.json'],
    'work_dirs/yolo_improvements/slicing_m_e30': ['experiment.json', 'epoch_30_metrics.json'],
    'work_dirs/gfl_drone12_mva7_1440': ['experiment.json', 'mva_epoch_38_metrics.json', 'model_config.py'],
}
SKIP = ('_predictions.json', '_raw.npz', '_test.json', '_limit')


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst); return dst.stat().st_size


def main():
    total, n, missing = 0, 0, []
    for f in FILES:
        if (ROOT/f).exists(): total += copy(ROOT/f, OUT/f); n += 1
        else: missing.append(f)
    for pattern, dest in RESULTS:
        for src in map(Path, glob.glob(str(ROOT/pattern))):
            if any(s in src.name for s in SKIP) and 'final_ensemble_test_info' not in src.name and 'gfl_ep38_val_limit20' not in src.name: continue
            total += copy(src, OUT/dest/src.name); n += 1
    for run, names in TRAIN_LOGS.items():
        for name in names:
            src = ROOT/run/name
            if src.exists(): total += copy(src, OUT/'results/training'/Path(run).name/name); n += 1
            else: missing.append(f'{run}/{name}')
    print(f'{n} files, {total/1e6:.1f} MB -> {OUT}')
    if missing: print('missing:', missing)


if __name__ == '__main__':
    main()
