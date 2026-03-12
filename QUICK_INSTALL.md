# 🚀 TimesMamba - Lệnh Cài Đặt Nhanh

## ⚡ Cài Đặt Nhanh (Copy & Paste)

### Bước 1: Cài Miniconda
```powershell
# Tải Miniconda
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\Downloads\miniconda"
cd "$env:USERPROFILE\Downloads\miniconda"
Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile "Miniconda3-latest-Windows-x86_64.exe"
.\Miniconda3-latest-Windows-x86_64.exe

# Khởi tạo conda (sau khi cài xong)
& 'C:\Users\manno\Miniconda3\Scripts\conda.exe' init powershell
# Đóng và mở lại PowerShell
```

### Bước 2: Chấp nhận ToS
```powershell
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r  
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

### Bước 3: Tạo Environment
```powershell
conda create -n timesmamba python=3.11 -y
conda activate timesmamba
```

### Bước 4: Cài PyTorch
```powershell
# CPU only (khuyến nghị cho Windows):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# HOẶC GPU với CUDA (nếu có GPU + driver mới):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Bước 5: Cài Dependencies
```powershell
pip install "numpy<2" packaging pandas scikit-learn matplotlib einops
```

### Bước 6: Test
```powershell
cd D:\KLTN\TimesMamba
python test_timesmamba.py
```

---

## 🎯 Lệnh Chạy Nhanh

### Kích hoạt môi trường (luôn chạy trước)
```powershell
conda activate timesmamba
```

### Chạy training
```powershell
# ETTh1
python run.py --model TimesMamba --data ETTh1 --seq_len 96 --pred_len 96 --channel_independence

# ECL  
python run.py --model TimesMamba --data custom --root_path ./dataset/electricity/ --data_path electricity.csv --seq_len 96 --pred_len 96 --use_mark

# Traffic
python run.py --model TimesMamba --data custom --root_path ./dataset/traffic/ --data_path traffic.csv --seq_len 96 --pred_len 96 --use_mark
```

---

## 🔧 Troubleshooting Nhanh

```powershell
# Nếu conda không hoạt động:
& 'C:\Users\manno\Miniconda3\Scripts\conda.exe' init powershell

# Nếu thiếu package:
pip install --force-reinstall numpy pandas scikit-learn matplotlib einops

# Kiểm tra setup:
python -c "import torch, numpy, pandas; print('OK')"

# Test project:
python -c "exec(open('mamba_ssm_mock.py').read()); print('Mock loaded')"
```

📖 **Hướng dẫn chi tiết**: Xem [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)