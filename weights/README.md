# Weights

The checkpoints are larger than GitHub's 100 MB file limit, so they are attached to the
[GitHub Release](../../releases) instead of being committed. Download them into this folder:

| File | Model | Size | Val AP50 | SHA-256 |
|---|---|---|---|---|
| `yolov8m_slicing_e30.pt` | YOLOv8-M, Drone2021 pretrain → MVA with slicing, epoch 30 | 103.7 MB | 79.3% (81.0% with flip TTA) | `58f2d7cc914063f9dc76abdec7b82a38cdd2392d26d86a4abf8a5857a48bd798` |
| `gfl_r50_fpn_ep38.pth` | GFL ResNet-50 FPN, Drone2021 → MVA, epoch 38 | 129.4 MB | 74.9% | `788a416af92c5f4cc3d2da793fe40d696264ad53ad0ddf2b539e545a84b1b951` |
| `yolov8m_drone2021_ep7.pt` *(optional)* | YOLOv8-M pretrained on Drone2021 only, epoch 7 (start point of every MVA run) | 103.7 MB | – | `25b9ca2cea39a8dfbab1e183f10ae8e4f8a5163fa46158191e4e7b68a8095488` |

Both models together (flip TTA + GFL, one weighted boxes fusion) give **84.5% AP50** on the audited validation split.

Check a download:

```bash
sha256sum weights/*.pt weights/*.pth
```

Formats: the YOLO files hold `{'model': state_dict, 'epoch', 'metrics', 'spec'}` for `ultralytics.nn.tasks.DetectionModel('yolov8m.yaml', nc=1)`;
the GFL file holds `{'state_dict', 'meta'}` for `configs/gfl_r50_fpn_ep38/model_config.py` (mmdet 2.25, per-image BatchNorm, FP16).
