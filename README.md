# Phát hiện chim nhỏ trong ảnh 4K (MVA 2023 SOD4SB): slicing, tiled inference và gộp box của hai detector

Đồ án nghiên cứu **nên giữ độ phân giải ở đâu** khi phát hiện chim chỉ khoảng 16 pixel trong ảnh 3840×2160: lúc huấn
luyện, lúc suy luận, hay khi gộp dự đoán của nhiều detector. Hai detector được huấn luyện **không dùng trọng số tải sẵn**
(không ImageNet, không COCO); mọi lựa chọn làm trên tập validation, tập test công khai chỉ chạy một lần để nộp.

**Nhóm thực hiện (Đại học FPT Đà Nẵng):** Vũ Quốc Đạt · Nguyễn Gia Phúc · Ngô Ngọc Trường Sơn —
**Giảng viên hướng dẫn:** Phan Thị Thu Hồng

**Bài báo:** bản IEEE [paper/main.pdf](paper/main.pdf) · bản Springer Nature [paper/main_springer.pdf](paper/main_springer.pdf) ·
bản Elsevier [paper/main_elsevier.pdf](paper/main_elsevier.pdf) · mã nguồn LaTeX [paper/main.tex](paper/main.tex)

**Notebook kiểm chứng (chạy được ngay, chỉ cần CPU):** [notebooks/results_and_verification.ipynb](notebooks/results_and_verification.ipynb)

**Trọng số đã huấn luyện:** [GitHub Release `v1.0`](https://github.com/sotbatchien123/sod4sb-small-bird-detection/releases/tag/v1.0)

---

## Mục lục

1. [Kết quả chính](#1-kết-quả-chính)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Cài đặt môi trường](#3-cài-đặt-môi-trường)
4. [Chuẩn bị dữ liệu](#4-chuẩn-bị-dữ-liệu)
5. [Tái lập từng thí nghiệm](#5-tái-lập-từng-thí-nghiệm)
6. [Kiểm chứng kết quả, notebook và trọng số](#6-kiểm-chứng-kết-quả-notebook-và-trọng-số)
7. [Giao thức đánh giá](#7-giao-thức-đánh-giá)
8. [Phiên bản mã nguồn và lưu ý tái lập](#8-phiên-bản-mã-nguồn-và-lưu-ý-tái-lập)
9. [Biên dịch bài báo](#9-biên-dịch-bài-báo)
10. [License và trích dẫn](#10-license-và-trích-dẫn)

---

## 1. Kết quả chính

Tập validation MVA 2023 SOD4SB đã rà soát (879 ảnh, 2.864 chim), COCO, tối đa 100 box mỗi ảnh. Thời gian đo trên
một GPU RTX 5070 Laptop 8 GB, cho mỗi ảnh 4K.

| Bước | AP50 | AP75 | AR100 | ms/ảnh |
|---|---|---|---|---|
| GFL R50-FPN (epoch 38) | 74.9 | 36.3 | 46.6 | 294 |
| YOLOv8-M baseline (Drone2021 → MVA) | 70.2 | 25.5 | 43.9 | 338 |
| + slicing khi huấn luyện | 79.3 | 38.8 | 50.0 | 337 |
| + flip TTA, gộp bằng WBF | 81.0 | 40.3 | 51.1 | 709 |
| **+ GFL, một lần WBF (IoU 0,55, trọng số 1:1:1)** | **84.5** | **45.8** | **53.4** | ≈1003 |

![Kết quả](Result.png)

- **Slicing là thay đổi hiệu quả nhất** (+9,1 AP50): mỗi epoch dùng mọi ô 1280×736 có chim, giữ nguyên độ phân giải
  gốc. Tăng màu không giúp (−0,2), copy-paste làm giảm (−3,3). Bảng IV của bài báo.
- **SAHI làm giảm độ chính xác** (79,3 → 67,5 AP50) vì ảnh đã được xử lý ở độ phân giải gốc: phóng to ô không thêm
  thông tin, đổi kích thước chim so với lúc huấn luyện, và sinh box bị cắt ở mép ô. Sửa ba lỗi này lấy lại 10,2 điểm
  nhưng vẫn không vượt suy luận thường. Bảng V.
- **YOLO và GFL mạnh ở các cỡ chim khác nhau:** YOLO tốt hơn ở chim 8–16 px (80,2 so với 67,4), GFL tốt hơn ở chim
  32–64 px (72,4 so với 60,6). Ensemble giữ phần tốt hơn của cả hai. Bảng VII.
- **Kiểm tra chéo split A / split B:** chọn luật gộp trên một nửa validation, chấm trên nửa kia: 84,1% và 84,8%
  (flip TTA một mình: 80,7% và 81,4%). Mục V-B.
- **So với checkpoint GFL do ban tổ chức phát hành** (khởi tạo COCO, ảnh 3840, flip): 88,9 / 44,2 / 54,2. Pipeline của
  nhóm thấp hơn 4,4 điểm AP50 nhưng **cao hơn 1,6 điểm AP75**, dù không dùng trọng số tải sẵn. Bảng VIII.

---

## 2. Cấu trúc thư mục

```
sod4sb-small-bird-detection/
├── model.py                  # pipeline cuối: YOLOv8-M (ảnh gốc + ảnh lật) + GFL R50-FPN + WBF (class Detector)
├── main.py                   # chạy pipeline cuối trên một thư mục ảnh, ghi dự đoán COCO và chấm điểm
├── train_model.py            # huấn luyện theo từng bước: yolo-pretrain | yolo --method ... | gfl | all --dry-run
├── evaluate.py               # AP50 / AP75 / AR100 (COCO, maxDets 100), --by-size
├── load_data.py              # đọc nhãn và ảnh
├── demo.ipynb                # demo trên ảnh val/test (cần GPU, ảnh và trọng số; output đã lưu sẵn)
├── notebooks/
│   └── results_and_verification.ipynb   # tính lại mọi số của bài báo từ dự đoán đã lưu, chỉ cần CPU
├── configs/
│   ├── yolo_improvements/    # công thức 4 nhánh YOLO: baseline, augmentation, copy_paste, slicing
│   └── gfl_r50_fpn_ep38/     # cấu hình và thông số huấn luyện GFL
├── scripts/                  # mọi thí nghiệm của bài báo (danh sách bên dưới)
├── results/
│   ├── training/             # metric theo epoch, experiment.json của từng lượt huấn luyện
│   ├── inference/            # flip TTA, SAHI, các luật gộp (Bảng V)
│   ├── ensemble/             # các luật gộp YOLO + GFL, kiểm tra split A/B (Bảng VI)
│   ├── error_analysis/       # theo cỡ chim, điểm F1 tốt nhất, ảnh minh hoạ lỗi (Bảng VII, Hình 5)
│   ├── submission/           # tóm tắt file nộp tập test công khai
│   └── verification/         # dự đoán val đã lưu + nhãn val đã rà soát, cho notebook kiểm chứng
├── tests/                    # unit test cho evaluate.py và hai luật WBF
├── data/                     # README cách tải dữ liệu, manifest nhãn đã rà soát
├── weights/                  # chỗ đặt trọng số tải từ Release (README có SHA-256)
├── paper/                    # bài báo: IEEE (main.tex), script tạo bản Elsevier và Springer, hình
├── THIRD_PARTY_NOTICES.md    # thư viện, dữ liệu và mẫu bài báo của bên thứ ba
└── LICENSE                   # AGPL-3.0
```

### Danh sách script

| Script | Việc làm | Dùng cho |
|---|---|---|
| `scripts/prepare_fixed50.py` | Tạo nhãn đã rà soát (bỏ 1 box trùng ở val) | Mọi bảng |
| `scripts/train_yolo_drone.py` | YOLOv8-M trên Drone2021, 10 epoch, khởi tạo ngẫu nhiên | Bước 1 |
| `scripts/train_yolo_data_methods.py` | YOLOv8-M trên MVA, 30 epoch, một chiến lược dữ liệu mỗi lượt | Bảng III, IV |
| `scripts/train_gfl_baseline.py` | GFL R50-FPN: Drone2021 6 epoch rồi MVA 50 epoch, BN theo từng ảnh | Bảng III |
| `scripts/eval_sahi_tta.py` | Suy luận YOLO: `full`, `tta`, `sahi`, `sahi_tta`; tập `val` hoặc `test` | Bảng V |
| `scripts/export_tta_parts.py` | Tách lượt TTA thành dự đoán ảnh gốc / ảnh lật / WBF | Bảng V, VI |
| `scripts/merge_tune.py` | So sánh các luật gộp trên box đã lưu (CPU), kèm kiểm tra split A/B | Bảng V |
| `scripts/sahi_fixed_metrics.py` | AP50/AP75/AR100 của các luật SAHI đã sửa | Bảng V |
| `scripts/gfl_predict.py` | Suy luận GFL giống hệt lúc validation của trainer | Bảng III, VIII |
| `scripts/ensemble_wbf.py` | Gộp YOLO + GFL (NMS, WBF với nhiều IoU/trọng số), kiểm tra split A/B | Bảng VI |
| `scripts/error_analysis.py` | AP50 theo cỡ chim, điểm F1 tốt nhất, ảnh minh hoạ TP/FP/FN | Bảng VII, Hình 3, 5 |
| `scripts/make_submission.py` | File nộp tập test công khai (WBF 1:1:1) | Nộp bài |
| `scripts/export_verification_data.py` | Đóng gói dự đoán val cho notebook kiểm chứng | Mục 6.1 |
| `scripts/make_verification_notebook.py` | Tạo notebook kiểm chứng | Mục 6.1 |
| `scripts/make_report_figures.py`, `scripts/make_repo_figures.py` | Hình của bài báo và của README | Hình 2, 4 |
| `scripts/per_image_bn.py`, `scripts/gpu_compat.py`, `scripts/amp_recovery.py` | BN theo từng ảnh (batch 1), MMCV trên GPU Blackwell, thử lại khi tràn FP16 | GFL |
| `scripts/paths.py` | Mọi đường dẫn ghi vào file kết quả đều tương đối với gốc repo | Tất cả |

---

## 3. Cài đặt môi trường

Python 3.9, PyTorch 2.7.1 + CUDA 12.8, một GPU 8 GB (đã chạy trên RTX 5070 Laptop). Chỉ để chạy notebook kiểm chứng
và unit test thì **không cần GPU**: cài `requirements.txt` và PyTorch bản CPU là đủ.

```bash
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS: source .venv/bin/activate
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install --target .yolo-deps ultralytics==8.3.40
git clone https://github.com/shinya7y/UniverseNet third_party/UniverseNet
git -C third_party/UniverseNet checkout 3652b18c7ce68122dae7a32670624727d50e0914
```

`requirements-5070.txt` ghi đúng phiên bản của môi trường đã sinh ra kết quả.

### Khắc phục sự cố khi cài đặt

1. **Cài PyTorch trước `requirements.txt`.** Nếu cài ngược lại, pip có thể kéo một bản PyTorch khác không có CUDA 12.8.
2. **GPU đời mới (Blackwell, RTX 50xx) và MMCV 1.6:** không có bản MMCV 1.6 dựng sẵn cho CUDA 12.8. Dùng bản lite
   (`MMCV_WITH_OPS=0 pip install mmcv==1.6.0`); `scripts/gpu_compat.py` thay các CUDA op còn thiếu bằng NMS của
   torchvision. Đã kiểm tra trên 6 trường hợp NMS 1.200 box: kết quả giống hệt bản CPU gốc.
3. **`ModuleNotFoundError: ultralytics`:** Ultralytics được cài vào `.yolo-deps/`, không vào site-packages. `model.py` tự
   thêm thư mục này vào `sys.path`; nếu chạy notebook từ nơi khác, thêm `.yolo-deps` và `third_party/UniverseNet`
   vào `PYTHONPATH`.
4. **Windows: DataLoader của GFL bị treo** khi dùng nhiều worker. Cấu hình GFL đặt `workers_per_gpu=0`.
5. **Windows: huấn luyện chậm một nửa khi chạy nền** (luồng đọc ảnh bị dồn sang lõi tiết kiệm điện). Chạy qua
   `python -u scripts/run_qos.py scripts/<script>.py ...` để tắt EcoQoS cho tiến trình.

---

## 4. Chuẩn bị dữ liệu

Ảnh **không** được phân phối lại; tải từ ban tổ chức MVA 2023 SOD4SB và đặt theo [data/README](data/README):

```
mva2023_sod4bird_train/{images, annotations}      # 8.880 ảnh train + 879 ảnh val
mva2023_sod4bird_pub_test/{images, annotations}   # 9.699 ảnh test công khai, không có nhãn
drone2021/{images, annotations}                   # chỉ cần cho bước tiền huấn luyện
```

```bash
python scripts/prepare_fixed50.py      # tạo data/mva_audited_v1/annotations/split_{train,val}_coco.json
```

Thay đổi duy nhất so với nhãn gốc: bỏ **1 box trùng hoàn toàn** ở tập val (2.864 thay vì 2.865 box; điểm thay đổi
dưới 0,1). Checksum của nhãn gốc và nhãn đã rà soát nằm trong [data/mva_audited_v1/manifest.json](data/mva_audited_v1/manifest.json).

---

## 5. Tái lập từng thí nghiệm

Mọi lệnh chạy từ thư mục gốc repo. Kết quả được ghi vào `work_dirs/` (checkpoint) và `reports/` (dự đoán, số liệu);
bản số liệu dùng trong bài báo đã được chép vào `results/`.

### 5.1 Tiền huấn luyện YOLOv8-M trên Drone2021

```bash
python train_model.py yolo-pretrain        # 10 epoch; checkpoint epoch 7 (AP50 Drone2021 44,6%) khởi tạo mọi lượt MVA
```

### 5.2 Bốn chiến lược dữ liệu cho YOLOv8-M trên MVA (Bảng III, IV)

```bash
python train_model.py yolo --method baseline
python train_model.py yolo --method augmentation
python train_model.py yolo --method copy_paste
python train_model.py yolo --method slicing
```

Cùng khởi tạo, cùng lịch học, cùng batch 6, 30 epoch; chỉ khác cách cắt ô. Kết quả báo cáo dùng **epoch cuối (30)**.
Công thức chi tiết ở `configs/yolo_improvements/*.json`.

### 5.3 GFL R50-FPN (Bảng III)

```bash
python train_model.py gfl                  # Drone2021 6 epoch, rồi MVA 50 epoch, ảnh 2560x1440, BN theo từng ảnh
python -u scripts/gfl_predict.py --split val --run work_dirs/gfl_r50_fpn --checkpoint best_mva.pth
```

Trainer lưu `best_mva.pth` là epoch có AP50 validation cao nhất (ở lượt của bài báo: epoch 38); epoch cuối đạt 74,2%
(xem mục 8). Trainer cũng ghi dự đoán val sau mỗi epoch (`mva_epoch_<k>.bbox.json`).

### 5.4 Suy luận: flip TTA và SAHI (Bảng V)

```bash
python -u scripts/eval_sahi_tta.py --run slicing_m_e30 --mode full
python -u scripts/eval_sahi_tta.py --run slicing_m_e30 --mode tta
python -u scripts/eval_sahi_tta.py --run slicing_m_e30 --mode sahi --zoom 1.5
python -u scripts/eval_sahi_tta.py --run slicing_m_e30 --mode sahi --zoom 1
python -u scripts/eval_sahi_tta.py --run slicing_m_e30 --mode sahi_tta --zoom 1.5
python scripts/merge_tune.py reports/sahi_tta/slicing_m_e30_tta_raw.npz            # luật gộp, kiểm tra split A/B
python scripts/merge_tune.py reports/sahi_tta/slicing_m_e30_sahi_z1.5_raw.npz      # các cách sửa SAHI
python scripts/sahi_fixed_metrics.py                                               # AP75/AR100 của SAHI đã sửa
python scripts/export_tta_parts.py reports/sahi_tta/slicing_m_e30_tta_raw.npz      # dự đoán ảnh gốc / ảnh lật
```

### 5.5 Gộp YOLO và GFL (Bảng VI, Mục V-B)

```bash
python scripts/ensemble_wbf.py --name gfl+plain+flip \
    work_dirs/gfl_r50_fpn/mva_epoch_38.bbox.json \
    reports/sahi_tta/slicing_m_e30_tta_plain_predictions.json \
    reports/sahi_tta/slicing_m_e30_tta_flip_predictions.json
```

Script so sánh NMS và WBF với IoU 0,5/0,55/0,6 và trọng số GFL 1/1,5/2/3, đồng thời chọn luật trên split A rồi chấm
trên split B (và ngược lại). Kết quả: `results/ensemble/gfl+plain+flip.json`.

### 5.6 Phân tích theo cỡ chim và lỗi (Bảng VII, Hình 3, 4, 5)

```bash
python scripts/error_analysis.py reports/sahi_tta/baseline_m_e30_full_predictions.json reports/sahi_tta/slicing_m_e30_full_predictions.json
python evaluate.py <file_dự_đoán>.json --by-size
python scripts/make_report_figures.py --en
```

### 5.7 So sánh với checkpoint GFL của ban tổ chức (Bảng VIII)

Checkpoint baseline của challenge (khởi tạo COCO, ảnh 3840×2160, flip TTA) được chạy bằng chính code đánh giá của nhóm
trên 879 ảnh val, cho 88,9 / 44,2 / 54,2 (AP50 / AP75 / AR100). Lượt chạy này chỉ dùng để kiểm tra code đánh giá; checkpoint
đó không có trong repo.

### 5.8 File nộp tập test công khai

```bash
python -u scripts/eval_sahi_tta.py --run slicing_m_e30 --mode tta --split test --chunk 500
python -u scripts/gfl_predict.py --split test
python scripts/export_tta_parts.py reports/sahi_tta/slicing_m_e30_tta_test_raw.npz
python scripts/make_submission.py          # 9.699 ảnh, 1.782.137 box; tóm tắt ở results/submission/
```

---

## 6. Kiểm chứng kết quả, notebook và trọng số

### 6.1 Notebook kiểm chứng (không cần GPU, ảnh hay trọng số)

[notebooks/results_and_verification.ipynb](notebooks/results_and_verification.ipynb) đã chạy sẵn, nên xem được output
ngay trên GitHub. Notebook đọc dự đoán val đã lưu (`results/verification/val_predictions.npz`, 7,4 MB) và nhãn val đã
rà soát, rồi:

1. Chấm từng nguồn: YOLO baseline, YOLO slicing, hai góc nhìn của lượt flip TTA, GFL.
2. Gộp lại flip TTA (luật `wbf` của `merge_tune.py`) và ensemble cuối (luật của `ensemble_wbf.py`).
3. So với Bảng III và Bảng VII: **mọi ô khớp, sai lệch 0,0**.
4. Chấm luật gộp cuối trên split A và split B.

Chạy lại (khoảng 3 phút trên CPU):

```bash
pip install -r requirements.txt jupyter
jupyter nbconvert --to notebook --execute --inplace notebooks/results_and_verification.ipynb
```

### 6.2 Demo trên ảnh

[demo.ipynb](demo.ipynb) nạp hai detector, chạy pipeline cuối trên một ảnh val và một ảnh test công khai, vẽ box, và
chấm 20 ảnh val bằng `main.py`. Cần GPU, ảnh MVA và trọng số; output đã được lưu sẵn trong notebook.

```python
from model import Detector
det = Detector('weights/yolov8m_slicing_e30.pt', 'weights/gfl_r50_fpn_ep38.pth')
boxes = det('frame.jpg')      # (n, 5): x1, y1, x2, y2, score, theo pixel của ảnh 4K
```

### 6.3 Trọng số đã huấn luyện

Các file vượt giới hạn 100 MB của GitHub nên nằm ở [Release `v1.0`](https://github.com/sotbatchien123/sod4sb-small-bird-detection/releases/tag/v1.0).
Tải vào [`weights/`](weights) và kiểm tra bằng SHA-256 trong [weights/README.md](weights/README.md).

| File | Mô hình | AP50 val |
|---|---|---|
| `yolov8m_slicing_e30.pt` | YOLOv8-M, Drone2021 → MVA với slicing, epoch 30 | 79,3% (81,0% với flip TTA) |
| `gfl_r50_fpn_ep38.pth` | GFL ResNet-50 FPN, Drone2021 → MVA, epoch 38 | 74,9% |
| `yolov8m_drone2021_ep7.pt` *(tuỳ chọn)* | YOLOv8-M chỉ tiền huấn luyện Drone2021, điểm xuất phát của mọi lượt MVA | – |

### 6.4 Unit test

```bash
python -m unittest discover tests -v
```

10 test tổng hợp (không phải thí nghiệm): `evaluate.py` cho 1,0 với dự đoán hoàn hảo, phân biệt IoU 0,5 và 0,75, tính
đúng AR100 khi bỏ sót và theo nhóm kích thước; hai luật WBF giữ điểm khi các nguồn đồng ý, giữ 1/3 điểm khi chỉ một
trong ba nguồn thấy box, tính toạ độ trung bình có trọng số và không gộp hai con chim khác nhau.

---

## 7. Giao thức đánh giá

1. **Chỉ số:** AP50 (chỉ số chính của challenge), AP75 (độ khít của box) và AR100 (tỉ lệ chim được tìm thấy, trung bình
   trên IoU 0,50–0,95). COCO, tối đa 100 box mỗi ảnh. Không dùng AP trung bình 0,50–0,95 hay AP_small.
2. **Checkpoint:** YOLO trên MVA luôn dùng **epoch cuối (30)**, không chọn theo validation. YOLO Drone2021 dùng epoch 7
   (AP50 cao nhất trên val **Drone2021**, không phải MVA). GFL dùng epoch 38, chọn theo AP50 val MVA (xem mục 8).
3. **Chọn luật gộp:** mọi luật gộp và trọng số được kiểm tra chéo trên split A (vị trí lẻ) và split B (vị trí chẵn) của
   tập val: chọn trên một nửa, chấm trên nửa kia.
4. **Tập test công khai:** không có nhãn; chỉ chạy một lần với cấu hình đã chọn trên val để tạo file nộp.

---

## 8. Phiên bản mã nguồn và lưu ý tái lập

- **Tag `v1.0`** đánh dấu phiên bản đã phát hành cùng trọng số.
- **Đường dẫn trong file kết quả** luôn tương đối với gốc repo và dùng dấu `/` (`scripts/paths.py`), nên không lộ thông
  tin máy và đọc được trên mọi hệ điều hành.
- **Checkpoint GFL được chọn trên val MVA:** epoch 38 đạt 74,9% AP50, epoch cuối 74,2%. Vì vậy số của GFL và ensemble
  lạc quan khoảng 0,7 điểm; bài báo ghi rõ điều này. Kiểm tra split A/B không bao gồm lựa chọn này.
- **GPU không hoàn toàn tất định:** lượt suy luận `full` và ảnh gốc trong lượt `tta` của cùng mô hình slicing lệch nhau
  vài chục box (53.827 và 53.854) nhưng cho cùng AP50/AP75/AR100. Notebook kiểm chứng dùng đúng file của từng bảng.
- **Huấn luyện tiếp được sau khi bị ngắt:** mọi trainer lưu checkpoint nguyên tử (mô hình, optimizer, scaler, trạng thái
  ngẫu nhiên, vị trí trong epoch) và kiểm tra hash của mã trước khi chạy tiếp; đã kiểm chứng chạy tiếp cho kết quả
  giống hệt chạy liền.

---

## 9. Biên dịch bài báo

[paper/main.tex](paper/main.tex) (lớp `IEEEtran`) là bản gốc; bản Elsevier và Springer Nature được tạo từ nó, nên chỉ
cần sửa một chỗ.

```bash
cd paper
python to_elsevier.py && python to_springer.py
tectonic -X compile main.tex                 # hoặc: xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex
tectonic -X compile main_elsevier.tex
tectonic -X compile main_springer.tex        # cần sn-jnl.cls và sn-mathphys-num.bst
```

- Tên tiếng Việt cần **XeLaTeX** (Tectonic dùng XeTeX) với font TeX Gyre Termes / Latin Modern; pdfLaTeX dùng gói `vntex`.
- `sn-jnl.cls` và `sn-mathphys-num.bst` (LPPL) không có trong repo: tải gói mẫu LaTeX của Springer Nature và chép hai
  file vào `paper/`.
- **Overleaf:** tải lên `main.tex`, `refs.bib` và thư mục `figures/`, chọn *Menu → Compiler → XeLaTeX*.

---

## 10. License và trích dẫn

Mã nguồn, tài liệu và trọng số của nhóm dùng [AGPL-3.0](LICENSE), vì code dùng Ultralytics YOLOv8 (AGPL-3.0).
UniverseNet, MMDetection và MMCV dùng Apache-2.0. Dữ liệu MVA 2023 SOD4SB và Drone2021 giữ điều khoản của ban tổ chức.
Chi tiết: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

```
@misc{sod4sb2026,
  title  = {Where Should Resolution Be Preserved? Slicing, Tiled Inference and Heterogeneous Box Fusion for Small Bird Detection},
  author = {Vũ, Quốc Đạt and Nguyễn, Gia Phúc and Ngô, Ngọc Trường Sơn and Phan, Thị Thu Hồng},
  year   = {2026},
  note   = {IEEE-format manuscript}
}
```
