# 📊 TimesMamba - Giải Thích Chi Tiết và Demo Thực Tế

## 🎯 TimesMamba Là Gì?

TimesMamba là một **model dự báo chuỗi thời gian đa biến** (multivariate time series forecasting). Nói đơn giản:

- **Input**: Dữ liệu trong quá khứ (96 time steps)
- **Output**: Dự báo tương lai (96 time steps tiếp theo)
- **Dữ liệu**: Có thể nhiều biến cùng lúc (nhiệt độ + độ ẩm + áp suất, hoặc nhiều cổ phiếu)

## 📈 Kết Quả Benchmark (Hình trong figures/)

Bảng so sánh cho thấy TimesMamba **tốt hơn** iTransformer và PatchTST:

### ETTh1 Dataset (Electricity Transforming Temperature):
- **TimesMamba**: MSE=0.375, MAE=0.397 
- **iTransformer**: MSE=0.386, MAE=0.405
- **PatchTST**: MSE=0.414, MAE=0.419
→ **TimesMamba thắng** cả 2 metrics!

### Electricity Dataset:  
- **TimesMamba**: MSE=0.141, MAE=0.237
- **iTransformer**: MSE=0.148, MAE=0.240  
- **PatchTST**: MSE=0.195, MAE=0.285
→ **TimesMamba vượt trội** rõ rệt!

### Traffic Dataset:
- **TimesMamba**: MSE=0.376, MAE=0.257
- **iTransformer**: MSE=0.395, MAE=0.268
- **PatchTST**: MSE=0.544, MAE=0.359
→ **TimesMamba tốt nhất**!

**Kết luận**: TimesMamba có **độ chính xác cao nhất** trong dự báo chuỗi thời gian!

## 🏗️ Cấu Trúc Model (Giải Thích folder model/ và layers/)

### 1. **TimesMamba.py** - Model chính:
```python
class Model:
    def __init__(self):
        # RevIN: Normalize dữ liệu đầu vào
        self.revin_layer = RevIN()
        
        # Embedding: Chuyển sequence thành vector
        self.enc_embedding = SeriesEmbedding() 
        
        # Mamba: Core architecture (thay thế Transformer)
        self.mamba = MambaForSeriesForecasting()
        
        # Projector: Tạo output prediction
        self.projector = Linear()
```

### 2. **layers/Embed.py** - Series Embedding:
- Chuyển đổi raw time series thành dense vectors
- Giống word embedding nhưng cho time series

### 3. **layers/RevIN.py** - Reversible Instance Normalization:
- Normalize input để training ổn định
- Có thể "reverse" để khôi phục scale gốc
- Giúp model học được pattern thực tế

### 4. **model/mambacore.py** - Mamba Architecture:
- **Mamba blocks**: Thay thế self-attention của Transformer
- **Hiệu quả hơn**: Linear complexity thay vì quadratic 
- **Ít VRAM hơn**: Không cần attention matrices

## 📁 Dataset Folder Trống - Làm Sao?

Dataset folder trống vì datasets quá lớn. Bạn cần tải từ:

### ETTh1/ETTh2/ETTm1/ETTm2:
```bash
# Download from: https://github.com/thuml/iTransformer
# Structure mong muốn:
dataset/
├── ETT-small/
│   ├── ETTh1.csv  # Hourly electricity data
│   ├── ETTh2.csv
│   ├── ETTm1.csv  # Monthly data  
│   └── ETTm2.csv
```

### Electricity & Traffic:
```bash  
dataset/
├── electricity/
│   └── electricity.csv  # 321 clients, hourly consumption
└── traffic/
    └── traffic.csv       # 862 sensors, hourly occupancy
```

## 🚀 Demo Thực Tế với Synthetic Data

Tôi sẽ tạo demo với dữ liệu giả lập có ý nghĩa để bạn hiểu rõ mục đích:

### Scenario: Dự báo điện năng tiêu thụ
- **Input**: 96 giờ qua (4 ngày) của 3 khu vực
- **Output**: Dự báo 24 giờ tiếp theo  
- **Variables**: Điện dân dụng, công nghiệp, thương mại

### Scenario: Dự báo giá cổ phiếu
- **Input**: 96 ngày qua của 5 cổ phiếu
- **Output**: Dự báo 7 ngày tiếp theo
- **Variables**: AAPL, GOOGL, MSFT, TSLA, NVDA

## 💡 Ý Nghĩa Test Files Tôi Tạo

### 1. `test_timesmamba.py`:
- **Mục đích**: Kiểm tra model có import/chạy được không  
- **Dữ liệu**: Random tensor (vô nghĩa)
- **Kết quả**: Chỉ test technical functionality

### 2. `test_text_experiment.py`:
- **Mục đích**: Chứng minh TimesMamba KHÔNG phù hợp với text
- **Lesson**: Dùng đúng tool cho đúng task

### 3. `proper_text_solution.py`:
- **Mục đích**: Giải pháp đúng cho text processing
- **Lesson**: Simple solution = best solution

## 🎯 Test Thực Tế Sẽ Tạo

Tôi sẽ tạo demo với:
1. **Synthetic time series data** có pattern thực tế
2. **Meaningful prediction task** 
3. **Visualization** kết quả dự báo
4. **Comparison** với baseline methods
5. **Explanation** tại sao kết quả có nghĩa

Bạn muốn tôi tạo demo nào?
- 🏭 **Industrial**: Dự báo sản lượng nhà máy
- 📈 **Financial**: Dự báo giá cổ phiếu  
- 🌡️ **Weather**: Dự báo nhiệt độ đa điểm
- ⚡ **Energy**: Dự báo tiêu thụ điện
- 🚗 **Traffic**: Dự báo lưu lượng giao thông

## 🏃‍♂️ Cách chạy (commands) — tạo output và kiểm tra

Các file dữ liệu mẫu đã được thêm vào `dataset/ETT-small`:

- `ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv`

Chạy demo trực tiếp (sẽ in metrics và lưu ảnh forecast):

PowerShell / cmd:
```
conda activate timesmamba
cd D:\KLTN\TimesMamba
python demo_electricity_forecast.py
```

Kết quả mong đợi:
- Console: thông tin dataset, input/target shapes, metrics (MSE/MAE/MAPE)
- File: `electricity_forecast_demo.png` trong thư mục gốc project (hình dự báo)

Chạy script so sánh (wrong vs right usage):
```
python usage_comparison.py
```
Kết quả: in ra ví dụ dùng sai (text) và ví dụ dùng đúng (time series), giúp hiểu rõ khác biệt.

Nếu bạn muốn kiểm tra loader đọc file `ETTh1.csv`:
1) Tạo script kiểm tra nhỏ (ví dụ `scripts/test_loader.py`) hoặc dùng Python REPL:
```
python - <<EOF
from data_provider.data_factory import data_provider
class Args: pass
args = Args()
args.root_path = 'dataset/ETT-small'
args.data = 'ETTh1'
args.data_path = 'ETTh1.csv'
args.seq_len = 96
args.label_len = 48
args.pred_len = 24
args.features = 'S'
args.target = 'OT'
args.batch_size = 2
args.num_workers = 0
args.embed = 'timeF'
args.freq = 'h'
ds, loader = data_provider(args, 'train')
print('Dataset length', len(ds))
for i, batch in enumerate(loader):
    print('Batch', i, [b.shape for b in batch])
    break
EOF
```

Ghi chú: `data_provider` ghép `args.root_path` + `args.data_path` để đọc CSV (xem `data_provider/data_factory.py`).

---
**Tóm tắt**: đã thêm file mẫu vào `dataset/ETT-small`; dùng `python demo_electricity_forecast.py` để sinh output (metrics + `electricity_forecast_demo.png`).