"""Narrow MMCV 1.6 compatibility layer for the author's GFL on modern PyTorch.

The GFL/ResNet/FPN path uses plain PyTorch layers plus axis-aligned NMS. Use
torchvision's compiled CPU/CUDA NMS instead of MMCV's obsolete binary extension.
Other MMCV extension operators explicitly fail; this is not a general MMCV port.
No upstream source files or installed packages are modified.
"""
import os
from collections import Counter
from types import SimpleNamespace

USAGE = Counter()
_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import torch
    import torchvision
    import mmcv
    from mmcv.utils import ext_loader
    import mmcv.parallel._functions as parallel_functions

    if mmcv.__version__ != '1.6.0':
        raise RuntimeError('This compatibility layer is scoped to MMCV 1.6.0.')
    # Locally the RTX 5070 needs PyTorch >= 2.7 (CUDA 12.8); Kaggle's T4 runs any PyTorch 2.x.
    minimum = (2, 0) if os.environ.get('SOD4SB_KAGGLE') == '1' else (2, 7)
    if tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) < minimum:
        raise RuntimeError(f'Use PyTorch >= {minimum[0]}.{minimum[1]} for this GPU runner.')

    original_load = ext_loader.load_ext

    def axis_aligned_nms(boxes, scores, iou_threshold, offset=0):
        if offset not in (0, 1):
            raise ValueError('NMS offset must be 0 or 1')
        USAGE['nms_' + boxes.device.type] += 1
        work_boxes = boxes.float()
        if offset:
            work_boxes = work_boxes.clone()
            work_boxes[:, 2:] += 1
        return torchvision.ops.nms(work_boxes, scores.float(), float(iou_threshold))

    def unsupported(name):
        def fail(*args, **kwargs):
            raise NotImplementedError(
                f'MMCV operation {name} is outside the supported GFL/ResNet/FPN path. '
                'This GPU environment does not implement other detectors or Elsa InternImage.')
        return fail

    def load_ext(name, functions):
        if name != '_ext':
            return original_load(name, functions)
        return SimpleNamespace(**{
            function: axis_aligned_nms if function == 'nms' else unsupported(function)
            for function in functions
        })

    # PyTorch 2.x changed this private helper from an integer to torch.device.
    modern_get_stream = torch.nn.parallel._functions._get_stream

    def get_stream(device):
        if isinstance(device, int):
            device = torch.device('cuda', device)
        return modern_get_stream(device)

    ext_loader.load_ext = load_ext
    parallel_functions._get_stream = get_stream
    _INSTALLED = True


def require_gfl_config(config):
    model = config.model
    if (model.type != 'GFL' or model.backbone.type != 'ResNet'
            or model.neck.type != 'FPN' or model.bbox_head.type != 'GFLHead'
            or model.test_cfg.nms.type != 'nms'
            or model.backbone.get('dcn') is not None):
        raise RuntimeError('The GPU compatibility layer only supports the original GFL/ResNet/FPN model.')
