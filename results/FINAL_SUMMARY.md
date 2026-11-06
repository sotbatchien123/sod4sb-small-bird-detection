# SOD4SB — tổng hợp kết quả cuối (04/11/2026)

Tất cả số liệu: tập **val audit** (879 ảnh 4K, 2.864 box), COCO maxDets 100, trừ khi ghi khác.
Tập test công khai (9.699 ảnh, không nhãn) chỉ dùng một lần để xuất file nộp với cấu hình đã chốt trên val.

## 1. Pipeline cuối

```
Drone2021 ─pretrain 10 ep (dùng ep 7)─► YOLOv8-M ─fine-tune MVA 30 ep + Slicing (crop 1280×736, 1:1)─► YOLO Slicing M
Drone2021 ─pretrain 6 ep─► GFL R50-FPN ─fine-tune MVA (2560×1440, BN theo ảnh), dùng ep 38────────────► GFL

Ảnh 4K gốc ─┬─► YOLO (ảnh thường) ──┐
            ├─► YOLO (ảnh lật, lật box lại) ─┼─► WBF một tầng (IoU 0,55, trọng số 1:1:1) ─► kết quả
            └─► GFL ────────────────┘
```

| Bước | AP50 | Tăng AP50 | AP75 | AR100 | ms/ảnh (RTX 5070 Laptop) |
|---|---|---|---|---|---|
| Baseline M (pretrain Drone → MVA) | 70,2% | — | 25,5% | 43,9% | 338 |
| + Slicing lúc train | 79,3% | +9,1 | 38,8% | 50,0% | 337 |
| + TTA lật ngang, ghép WBF | 81,0% | +1,7 | 40,3% | 51,1% | 709 |
| + Ensemble GFL (WBF 3 nguồn) | **84,5%** | +3,5 | **45,8%** | **53,4%** | ~709 + 294 |

Cấu hình cuối: AP50 **84,5%**, AP75 45,8%, AR100 53,4%.
Chỉ số dùng trong báo cáo: AP50, AP75, AR100. Mọi YOLO đều lấy epoch cuối (30), không chọn epoch trên val MVA. Riêng GFL dùng ep 38/50, được chọn theo AP50 trên val MVA (epoch cuối 74,2% thay vì 74,9%), nên số của GFL và ensemble hơi lạc quan khoảng 0,7 điểm.
Kiểm tra chống "chọn may": chọn luật trên nửa ảnh lẻ → nửa chẵn 84,8%; chọn trên nửa chẵn → nửa lẻ 84,1%
(YOLO TTA đơn lẻ trên cùng các nửa: 81,4% / 80,7%).

Hình: `reports/figures/01_pipeline_progression.png`, `reports/figures/02_ap50_by_size.png`.

## 2. Ablation lúc train (YOLO, epoch 30)

| Model | AP50 | AP75 | AR100 | AP50 so với Baseline |
|---|---|---|---|---|
| YOLOv8-M Baseline | 70,2% | 25,5% | 43,9% | — |
| YOLOv8-M Augmentation màu | 70,0% | 25,5% | 43,8% | −0,2 |
| YOLOv8-M Copy-Paste | 66,9% | 23,2% | 42,2% | −3,3 |
| **YOLOv8-M Slicing** | **79,3%** | **38,8%** | **50,0%** | **+9,1** |
| YOLOv8-M chỉ pretrain (không MVA) | 0,1% | 0,0% | 2,5% | — |
| GFL R50-FPN (ep 38) | 74,9% | 36,3% | 46,6% | (đối chứng) |

Chỉ pretrain đạt 0,1% không phải vì không thấy chim: box dự đoán to hơn chuẩn MVA (trung vị √diện tích
40 px so với 16 px của nhãn MVA; nhãn Drone2021 25–31 px), IoU < 0,5 nên thành FP. Fine-tune trên MVA
chủ yếu dạy lại quy ước khoanh box.

## 3. Ablation lúc dự đoán (Slicing M)

| Cách | NMS 0,6 | Cách ghép tốt nhất | ms/ảnh |
|---|---|---|---|
| Thường | 79,3% | — | 337 |
| TTA lật ngang | 79,9% | WBF 81,0% | 709 |
| SAHI ×1,5 (36 tile) | 67,5% | bỏ box mép tile + score tile ×0,5: 77,7% | 1748 |
| SAHI ×1,5 + TTA | 66,0% | 78,2% | 3621 |
| SAHI ×1,0 (16 tile) | 78,5% | 79,3% | 963 |

SAHI không giúp vì pipeline đã dự đoán ở độ phân giải gốc (không thu nhỏ ảnh 4K); tile phóng to chỉ là
nội suy, lệch thang đo so với lúc train, sinh box giả ở tile trống và box cắt đôi ở mép tile.

## 4. AP50 theo kích thước chim

| Model | 8–16 px (1.456) | 16–32 px (1.070) | 32–64 px (316) | Best F1 | FP / FN |
|---|---|---|---|---|---|
| Baseline M | 64,3% | 80,3% | 58,9% | 71,8% | 495 / 984 |
| Slicing M | 78,4% | 86,1% | 58,4% | 81,8% | 224 / 728 |
| Slicing M + TTA (WBF) | 80,2% | 87,6% | 60,6% | 82,9% | 198 / 698 |
| GFL | 67,4% | 87,7% | **72,4%** | 77,4% | 216 / 920 |
| **Ensemble** | **83,5%** | **89,4%** | **72,4%** | **84,5%** | 218 / **611** |

YOLO Slicing mạnh ở chim nhỏ (8–16 px), GFL mạnh ở chim lớn (32+ px): hai mô hình bù cho nhau,
ensemble lấy được phần tốt của cả hai. Ảnh minh hoạ FN/FP/TP: `reports/error_analysis/`.

## 5. File

| Nội dung | Đường dẫn |
|---|---|
| Kết quả từng cách dự đoán | `reports/sahi_tta/*.json` |
| Thử cách ghép box | `reports/sahi_tta/*_merge_tune.json` |
| Ensemble | `reports/ensemble/*.json` |
| Phân tích lỗi | `reports/error_analysis/` |
| File nộp tập test | `reports/submission/final_ensemble_test_predictions.json` |
| Script | `eval_sahi_tta.py`, `merge_tune.py`, `ensemble_wbf.py`, `export_tta_parts.py`, `gfl_predict.py`, `make_submission.py`, `error_analysis.py`, `make_report_figures.py` |
