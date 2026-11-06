"""Original GFL + horizontal-flip TTA, adapted only for local paths and CPU."""
_base_ = '../third_party/UniverseNet/configs/datasets/mva2023_sod4bird/gfl_r50_fpn_fp16_1x2x8_7e_mva2023_sod4bird_2160_test_whole_config.py'

data_root = '{{ fileDirname }}/../mva2023_sod4bird_train/'
classes = ('bird',)
model = dict(backbone=dict(init_cfg=None))
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        classes=classes,
        ann_file=data_root + 'annotations/local_train_mini.json',
        img_prefix=data_root + 'images/'),
    val=dict(
        classes=classes,
        ann_file=data_root + 'annotations/local_val_mini.json',
        img_prefix=data_root + 'images/'),
    test=dict(
        classes=classes,
        ann_file=data_root + 'annotations/local_val_mini.json',
        img_prefix=data_root + 'images/'))

device = 'cpu'
gpu_ids = []
fp16 = None
load_from = None
opencv_num_threads = 0
mp_start_method = 'spawn'
dist_params = dict(backend='gloo')
