# Ghi nhận bên thứ ba (Third-party notices)

File [LICENSE](LICENSE) (AGPL-3.0) áp dụng cho mã nguồn, tài liệu và trọng số do dự án tạo ra. Dự án dùng AGPL-3.0 vì
mã huấn luyện và suy luận YOLO gọi trực tiếp thư viện Ultralytics (AGPL-3.0). Các thành phần dưới đây thuộc bên thứ
ba, giữ nguyên điều khoản của họ và **không được đưa vào repo**, trừ ngoại lệ ghi rõ ở mục Dữ liệu.

## Mã nguồn và thư viện

| Thành phần | Nguồn | License | Cách dùng trong dự án |
|---|---|---|---|
| Ultralytics YOLOv8 (8.3.40) | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 | Cài vào `.yolo-deps/` bằng `pip install --target`. Dùng định nghĩa mô hình (`DetectionModel`, `yolov8m.yaml`), EMA và NMS; vòng lặp huấn luyện do dự án viết. Không phân phối lại. |
| UniverseNet (commit `3652b18`) | [shinya7y/UniverseNet](https://github.com/shinya7y/UniverseNet) | Apache-2.0 | Clone vào `third_party/UniverseNet/`. Cung cấp GFL R50-FPN và MMDetection 2.x. Không phân phối lại. |
| MMDetection 2.x (trong UniverseNet) | [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) | Apache-2.0 | Dựng và chạy GFL. |
| MMCV 1.6.0 (bản lite, không có CUDA op) | [open-mmlab/mmcv](https://github.com/open-mmlab/mmcv) | Apache-2.0 | `scripts/gpu_compat.py` thay các CUDA op còn thiếu bằng NMS của torchvision. |
| PyTorch, torchvision | [pytorch/pytorch](https://github.com/pytorch/pytorch) | BSD-3-Clause | Huấn luyện, suy luận, NMS. |
| pycocotools | [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi) | BSD-2-Clause | Tính AP50, AP75, AR100 (`evaluate.py`). |
| NumPy, pandas | numpy.org, pandas.pydata.org | BSD-3-Clause | Xử lý số liệu, notebook. |
| OpenCV (opencv-python) | [opencv/opencv](https://github.com/opencv/opencv) | Apache-2.0 | Đọc ảnh 4K, vẽ box. |
| matplotlib, Pillow, PyYAML, addict, yapf, terminaltables | — | Giấy phép riêng của từng thư viện (đều cho phép dùng tự do) | Hình, cấu hình, phụ thuộc của MMCV. |

Phương pháp **weighted boxes fusion** (Solovyev et al., 2021) và **SAHI** (Akyon et al., 2022) được dự án tự cài lại
(`scripts/ensemble_wbf.py`, `scripts/merge_tune.py`, `scripts/eval_sahi_tta.py`); không dùng mã của hai thư viện gốc.

## Trọng số mô hình

| Trọng số | Nguồn | Ghi chú |
|---|---|---|
| `yolov8m_drone2021_ep7.pt`, `yolov8m_slicing_e30.pt`, `gfl_r50_fpn_ep38.pth` | Dự án tự huấn luyện, khởi tạo ngẫu nhiên | Đăng ở GitHub Release v1.0, theo LICENSE của repo. **Không dùng trọng số ImageNet hay COCO.** |
| Checkpoint GFL do ban tổ chức SOD4SB phát hành (khởi tạo từ COCO) | Baseline của challenge (UniverseNet) | Chỉ dùng để **kiểm tra code đánh giá** (Mục V-C của bài báo). Không phân phối lại. |

## Dữ liệu

| Dữ liệu | Nguồn | Điều khoản |
|---|---|---|
| MVA 2023 SOD4SB (ảnh 4K và nhãn) | Ban tổ chức MVA 2023 Challenge on Small Object Detection for Birds | Copyright (c) 2023 MVA organization and Toyota Technological Institute; điều khoản dựa trên MIT, dùng thương mại phải liên hệ mva2023-te@mva-org.jp. Toàn văn: [`results/verification/MVA_SOD4SB_LICENSE.txt`](results/verification/MVA_SOD4SB_LICENSE.txt). |
| Drone2021 (Fujii et al.) | Tải theo hướng dẫn của SOD4SB | Theo điều khoản của nhà cung cấp. Chỉ dùng để tiền huấn luyện. |

**Ngoại lệ duy nhất:** [`results/verification/mva_val_audited_coco.json`](results/verification/mva_val_audited_coco.json)
là nhãn validation của MVA 2023 SOD4SB (879 ảnh), chỉ bỏ 1 box trùng (2.864 box). File được đưa vào repo để notebook
kiểm chứng chạy được mà không cần tải dữ liệu, kèm thông báo bản quyền gốc như license của MVA yêu cầu. File không
chứa ảnh. `results/verification/val_predictions.npz` là dự đoán do mô hình của dự án sinh ra, theo LICENSE của repo.

Ngoài hai file trên, dự án không phân phối lại ảnh hay nhãn nào. Cách tải dữ liệu xem [`data/README`](data/README).

## Mẫu bài báo

| Mẫu | License | Ghi chú |
|---|---|---|
| IEEEtran, elsarticle | LPPL | Có sẵn trong bản phân phối LaTeX (Tectonic, TeX Live). |
| Springer Nature `sn-jnl` (12/2024) | LPPL | Không đưa vào repo (`.gitignore`); tải từ trang LaTeX author support của Springer Nature. |

## Trích dẫn

- S. Fujii, K. Akita, N. Ukita. *Distant Bird Detection for Safe Drone Flight and Its Dataset* (Drone2021). MVA 2021.
- Y. Kondo et al. *MVA2023 Small Object Detection Challenge for Spotting Birds: Dataset, Methods, and Results.* MVA 2023.
- X. Li et al. *Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection.* NeurIPS 2020.
- G. Jocher, A. Chaurasia, J. Qiu. *Ultralytics YOLOv8.* 2023.
- Y. Shinya. *USB: Universal-Scale Object Detection Benchmark* (UniverseNet). BMVC 2022.
- R. Solovyev, W. Wang, T. Gabruseva. *Weighted Boxes Fusion: Ensembling Boxes from Different Object Detection Models.* Image and Vision Computing, 2021.
- F. C. Akyon, S. O. Altinuc, A. Temizel. *Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection.* ICIP 2022.
