# 📦 Hướng Dẫn Cài Đặt TimesMamba - Chi Tiết Từ A-Z

## 🎯 Tổng Quan
TimesMamba là một model dự báo chuỗi thời gian đa biến sử dụng kiến trúc Mamba thay thế cho Transformer. Hướng dẫn này sẽ giúp bạn cài đặt hoàn chỉnh project trên Windows.

## 💻 Yêu Cầu Hệ Thống

### Tối Thiểu:
- **OS**: Windows 10/11 (64-bit)
- **RAM**: 8GB trở lên
- **Storage**: 10GB free space
- **Python**: Sẽ cài qua Miniconda

### Khuyến Nghị (cho GPU):
- **GPU**: NVIDIA GPU với CUDA support
- **VRAM**: 4GB+ 
- **NVIDIA Driver**: Version 470+ 
- **CUDA**: 11.8+ hoặc 12.1+

---

## 🚀 Hướng Dẫn Cài Đặt

### Bước 1: Cài Đặt Miniconda

#### 1.1. Tải Miniconda
```powershell
# Mở PowerShell và chạy:
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\Downloads\miniconda"
cd "$env:USERPROFILE\Downloads\miniconda"

# Tải Miniconda
Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile "Miniconda3-latest-Windows-x86_64.exe"
```

#### 1.2. Cài Đặt Miniconda
```powershell
# Chạy installer
.\Miniconda3-latest-Windows-x86_64.exe
```
**Lưu ý**: Trong quá trình cài đặt, **KHÔNG** chọn "Add to PATH" để tránh xung đột.

#### 1.3. Khởi Tạo Conda cho PowerShell
```powershell
# Sau khi cài xong, khởi tạo conda
& 'C:\Users\[YOUR_USERNAME]\Miniconda3\Scripts\conda.exe' init powershell

# Đóng và mở lại PowerShell, hoặc chạy:
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" init powershell
```

#### 1.4. Chấp Nhận Terms of Service
```powershell
# Chạy từng lệnh sau:
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

#### 1.5. Kiểm Tra Conda
```powershell
conda --version
# Kết quả mong đợi: conda 26.1.1 (hoặc version mới hơn)
```

---

### Bước 2: Tạo Môi Trường Python

#### 2.1. Tạo Conda Environment
```powershell
# Tạo environment với Python 3.11 (theo yêu cầu của project)
conda create -n timesmamba python=3.11 -y

# Kích hoạt environment
conda activate timesmamba
```

#### 2.2. Kiểm Tra Environment
```powershell
python --version
# Kết quả mong đợi: Python 3.11.x
```

---

### Bước 3: Cài Đặt PyTorch

#### Option A: GPU với CUDA (Khuyến nghị nếu có GPU)
```powershell
# Kiểm tra NVIDIA driver trước
nvidia-smi

# Nếu driver >= 470, cài PyTorch CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Option B: CPU Only (Cho máy không có GPU hoặc driver cũ)
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 3.1. Kiểm Tra PyTorch
```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

---

### Bước 4: Cài Đặt Dependencies Cơ Bản

```powershell
# Cài đặt các thư viện cần thiết
pip install "numpy<2" packaging pandas scikit-learn matplotlib einops

# Kiểm tra cài đặt
python -c "import numpy, pandas, matplotlib, einops; print('✅ Core packages installed')"
```

---

### Bước 5: Xử Lý Mamba-SSM (Quan Trọng!)

TimesMamba cần `mamba-ssm` package, nhưng package này khó cài trên Windows. Có 2 cách xử lý:

#### Option A: Sử dụng Mock (Khuyến nghị cho testing)
```powershell
# Tạo file mock (copy code từ mamba_ssm_mock.py đã tạo)
# File này sẽ cho phép project chạy được mà không cần mamba-ssm thật
```

#### Option B: Cài Mamba-SSM Thật (Cần GPU + CUDA)
```powershell
# Chỉ thực hiện nếu đã có CUDA và GPU driver mới
# Cài CUDA Toolkit nếu cần:
conda install -c nvidia cuda-toolkit=12.1 -y

# Kiểm tra nvcc
nvcc --version

# Cài mamba-ssm (có thể mất lâu và cần compiler)
pip install mamba-ssm==1.1.0 causal-conv1d==1.1.0
```

---

### Bước 6: Clone và Setup Project

```powershell
# Chuyển đến thư mục project (nếu chưa có)
cd D:\KLTN\TimesMamba

# Tạo mock mamba-ssm (nếu dùng Option A)
# Copy nội dung mamba_ssm_mock.py đã tạo
```

---

### Bước 7: Test Installation

```powershell
# Test cơ bản
python test_timesmamba.py

# Kết quả mong đợi:
# ✅ All imports successful!
# ✅ Model created successfully
# ✅ Forward pass successful!
```

---

## 🔧 Các Lệnh Sử Dụng

### Kích Hoạt Environment (Luôn chạy trước khi làm việc)
```powershell
conda activate timesmamba
```

### Chạy Training

#### ETTh1 Dataset (Channel Independence)
```powershell
python run.py \
  --is_training 1 \
  --model TimesMamba \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --channel_independence \
  --e_layers 2 \
  --d_model 128 \
  --d_ff 256 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --train_epochs 10
```

#### ECL Dataset (Channel Mixing)
```powershell
python run.py \
  --is_training 1 \
  --model TimesMamba \
  --data custom \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --use_mark \
  --e_layers 2 \
  --d_model 128 \
  --batch_size 32 \
  --train_epochs 10
```

#### Traffic Dataset (With Temporal Features)
```powershell
python run.py \
  --is_training 1 \
  --model TimesMamba \
  --data custom \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --use_mark \
  --e_layers 2 \
  --d_model 128 \
  --batch_size 16 \
  --train_epochs 10
```

### Chạy với Mock Mamba-SSM
```powershell
# Nếu sử dụng mock, luôn load mock trước:
python -c "exec(open('mamba_ssm_mock.py').read()); exec(open('run.py').read())"
```

---

## 🎛️ Tham Số Quan Trọng

### Model Configuration:
- `--d_model`: Dimension của model (default: 128)
- `--d_ff`: Dimension của feedforward (default: 256) 
- `--e_layers`: Số layer encoder (default: 2)
- `--dropout`: Dropout rate (default: 0.1)

### Mamba Specific:
- `--ssm_expand`: Mamba expansion factor (0 = disable Mamba)
- `--r_ff`: FFN expansion ratio (0 = disable FFN)

### Data Preprocessing:
- `--channel_independence`: Channel independence mode
- `--use_mark`: Use temporal features
- `--no_norm`: Disable RevIN normalization
- `--revin_affine`: Enable learnable affine in RevIN

### Training:
- `--seq_len`: Input sequence length (default: 96)
- `--pred_len`: Prediction length (default: 96) 
- `--batch_size`: Batch size (default: 32)
- `--train_epochs`: Training epochs (default: 10)
- `--learning_rate`: Learning rate (default: 0.001)

---

## 📁 Cấu Trúc Dataset

### Tải Datasets
Theo hướng dẫn trong [iTransformer repo](https://github.com/thuml/iTransformer):

```
dataset/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── electricity/
│   └── electricity.csv
├── traffic/
│   └── traffic.csv
├── weather/
│   └── weather.csv
└── exchange_rate/
    └── exchange_rate.csv
```

---

## 🐛 Troubleshooting

### Lỗi Thường Gặp:

#### 1. Conda không nhận diện được
```powershell
# Giải pháp: Khởi tạo lại conda
& 'C:\Users\[USERNAME]\Miniconda3\Scripts\conda.exe' init powershell
# Đóng và mở lại PowerShell
```

#### 2. PyTorch CUDA không hoạt động
```powershell
# Kiểm tra driver NVIDIA
nvidia-smi

# Nếu driver cũ, cài lại PyTorch CPU:
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 3. Mamba-SSM không cài được
```powershell
# Sử dụng mock thay thế (khuyến nghị):
python -c "exec(open('mamba_ssm_mock.py').read()); [YOUR_COMMAND]"
```

#### 4. Out of Memory
```powershell
# Giảm batch size:
python run.py --batch_size 16  # hoặc 8

# Hoặc giảm model size:
python run.py --d_model 64 --d_ff 128
```

#### 5. Import Error
```powershell
# Cài lại dependencies:
pip install --force-reinstall numpy pandas scikit-learn matplotlib einops
```

### Kiểm Tra Hệ Thống:
```powershell
# Kiểm tra environment
conda info --envs
conda list

# Kiểm tra Python packages
pip list | findstr -E "(torch|numpy|pandas)"

# Test import cơ bản
python -c "import torch, numpy, pandas, einops; print('All OK')"
```

---

## 📊 Performance Tips

### Tối Ưu Cho CPU:
- Sử dụng `batch_size` nhỏ (16-32)
- Giảm `d_model` và `d_ff` nếu cần
- Set `--ssm_expand 0` để disable Mamba (chỉ dùng FFN)

### Tối Ưu Cho GPU:
- Tăng `batch_size` (64-128)
- Sử dụng mixed precision training
- Cài đặt mamba-ssm thật cho performance tốt nhất

---

## 📚 Tài Liệu Tham Khảo

- [TimesMamba Paper](https://github.com/thuml/Time-Series-Library)  
- [Mamba Architecture](https://github.com/state-spaces/mamba)
- [iTransformer Dataset](https://github.com/thuml/iTransformer)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)

---

## 🎉 Hoàn Thành!

Sau khi hoàn thành các bước trên, bạn sẽ có:

✅ Môi trường TimesMamba hoàn chỉnh  
✅ PyTorch với/không CUDA support  
✅ Tất cả dependencies cần thiết  
✅ Mock mamba-ssm để testing  
✅ Scripts và configs sẵn sàng  

**Commands để bắt đầu:**
```powershell
conda activate timesmamba
python test_timesmamba.py
python run.py --model TimesMamba --data ETTh1 --seq_len 96 --pred_len 96
```

**Chúc bạn thành công với KLTN TimesMamba! 🚀**