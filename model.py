"""Final SOD4SB detector in one file: YOLOv8-M (slicing) on the full frame and its mirror, GFL R50-FPN,
one weighted boxes fusion (IoU 0.55, weights 1:1:1). Mirrors scripts/eval_sahi_tta.py (mode tta),
scripts/gfl_predict.py and scripts/make_submission.py, so results match the reported numbers.

    from model import Detector
    det = Detector('weights/yolov8m_slicing_e30.pt', 'weights/gfl_r50_fpn_ep38.pth')
    boxes = det(image_path)          # (n, 5) array: x1, y1, x2, y2, score in pixels of the 4K frame
"""
import math, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'.yolo-deps'), str(ROOT/'third_party/UniverseNet'), str(ROOT/'scripts')]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'.cache/yolo')); os.environ.setdefault('YOLO_OFFLINE', 'true')

import cv2
import numpy as np
import torch

CONF, NMS_IOU, MAX_DET = .001, .6, 300      # YOLO post-processing (as in training-time validation)
WBF_IOU = .55                               # final fusion


# ----------------------------------------------------------------------------- YOLOv8-M
def load_yolo(path, device='cuda'):
    from ultralytics.nn.tasks import DetectionModel
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = DetectionModel('yolov8m.yaml', nc=1, verbose=False)
    model.load_state_dict(ck['model'], strict=True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    return model.to(device).eval()


def _tensor(img):
    h, w = img.shape[:2]
    canvas = np.full((math.ceil(h/32)*32, math.ceil(w/32)*32, 3), 114, dtype=np.uint8); canvas[:h, :w] = img
    return torch.from_numpy(np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1))).float().div(255)


@torch.inference_mode()
def predict_yolo(model, img, flip=True):
    """img: HxWx3 BGR uint8 (full frame, no resize). Returns [plain] or [plain, flipped-back] (n,5) arrays."""
    from ultralytics.utils.ops import non_max_suppression
    dev = next(model.parameters()).device
    out = []
    views = [img] + ([np.ascontiguousarray(img[:, ::-1])] if flip else [])
    for k, v in enumerate(views):
        d = non_max_suppression(model(_tensor(v)[None].to(dev)), conf_thres=CONF, iou_thres=NMS_IOU, max_det=MAX_DET)[0]
        d = d[:, :5].float().cpu().numpy()
        if k == 1: d[:, [0, 2]] = img.shape[1] - d[:, [2, 0]]
        out.append(d.astype(np.float32))
    return out


# ----------------------------------------------------------------------------- GFL R50-FPN
def load_gfl(checkpoint, config=ROOT/'configs/gfl_r50_fpn_ep38/model_config.py', device='cuda:0'):
    from gpu_compat import install
    install()
    from mmcv import Config
    from mmcv.runner import wrap_fp16_model
    from mmdet.models import build_detector
    from per_image_bn import enable_per_image_bn
    cfg = Config.fromfile(str(config)); cfg.model.backbone.init_cfg = None
    model = build_detector(cfg.model); model.CLASSES = ('bird',); model.cfg = cfg
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['state_dict'], strict=True)
    model.to(device); wrap_fp16_model(model); enable_per_image_bn(model)   # as trained: FP16, per-image BN
    return model.eval()


@torch.no_grad()
def predict_gfl(model, image_path):
    """Test pipeline of the config (2560x1440, no flip), boxes rescaled to the original frame."""
    from mmdet.apis import inference_detector
    return inference_detector(model, str(image_path))[0].astype(np.float32)


# ----------------------------------------------------------------------------- fusion
def fuse(sources, weights=None, iou=WBF_IOU):
    """Weighted boxes fusion of a list of (n,5) arrays (one per model or view)."""
    from ensemble_wbf import wbf
    return wbf(sources, weights or [1.]*len(sources), iou)


class Detector:
    """Final configuration: WBF of [GFL, YOLO plain, YOLO flipped] with equal weights (val AP50 84.5%)."""
    def __init__(self, yolo_weights, gfl_weights, device='cuda'):
        self.yolo = load_yolo(yolo_weights, device)
        self.gfl = load_gfl(gfl_weights, device='cuda:0' if device == 'cuda' else device)

    def __call__(self, image_path):
        img = cv2.imread(str(image_path))
        plain, flipped = predict_yolo(self.yolo, img, flip=True)
        b = fuse([predict_gfl(self.gfl, image_path), plain, flipped])
        if len(b):
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, img.shape[1]); b[:, [1, 3]] = b[:, [1, 3]].clip(0, img.shape[0])
        return b


def draw(img, boxes, gt=None, thr=.3, thickness=1, font_scale=.35):
    """Draw predictions with score >= thr (red) and optional ground-truth xyxy boxes (green).
    Defaults suit enlarged crops; use thickness=3, font_scale=1.2 for a whole 4K frame."""
    out = img.copy()
    for x1, y1, x2, y2 in (gt if gt is not None else []):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), thickness)
    for x1, y1, x2, y2, s in boxes[boxes[:, 4] >= thr]:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness)
        cv2.putText(out, f'{s:.2f}', (int(x1), int(y1)-3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1, cv2.LINE_AA)
    return out
