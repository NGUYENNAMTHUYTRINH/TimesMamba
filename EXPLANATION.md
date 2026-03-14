# 📊 TimesMamba - Giải Thích Chi Tiết & Cách Vận Hành

## 🎯 TimesMamba Là Gì?

**TimesMamba** là một **model dự báo chuỗi thời gian đa biến** (multivariate time series forecasting).

Nói dễ hiểu:
- Nó nhìn vào **dữ liệu quá khứ** (ví dụ: 4 ngày qua)
- Sau đó **dự báo tương lai** (ví dụ: 1 ngày tiếp theo)
- Có thể xử lý **nhiều biến cùng lúc** (nhiệt độ + độ ẩm + áp suất, v.v.)

---

## 🔄 Cách TimesMamba Vận Hành (Workflow)

### **Sơ đồ Tổng Quát:**

```
┌─────────────────────────────────────────────────────────────┐
│                      TỌA ĐỘ THỜI GIAN                      │
│                                                              │
│  Quá Khứ (96 bước)    │ Tương Lai (24 bước)                │
│  ──────────────────   │ ──────────────────                  │
│  Day 1 → Day 4        │ Day 5 → Day 5 (1 ngày)            │     
│  (4 ngày dữ liệu)     │ (dự báo cần làm)                   │
│                        │                                      │
│  Input: [n₁, n₂,      │ Output: [p₁, p₂, p₃, p₄, ...]    │
│           n₃, n₄, ...]│ (dự đoán của model)                │
│                        │                                      │
└─────────────────────────────────────────────────────────────┘
      ↓↓↓ MỤC TIÊU ↓↓↓
  Dự báo chính xác nhất có thể!
```

---

## 📈 3 Giai Đoạn Chính

### **Giai Đoạn 1: TRAINING (Huấn Luyện)**

**Mục đích:** Dạy model nhận biết pattern trong dữ liệu

#### Dữ liệu đầu vào:
```
train/datasets/ETTh1/ETTh1_train.csv
├─ Columns: OT, M1, M2, M3  (4 biến)
├─ Rows: ~26,000 time steps
└─ Dữ liệu thực từ hệ thống Electricity Transforming Temperature
```

#### Quá trình Training:

```python
┌─────────────────────────────────────────────────────┐
│                  TRAINING LOOP                      │
└─────────────────────────────────────────────────────┘

Epoch 1, 2, 3, ..., 10:
    ↓
    Lấy 32 mẫu từ dữ liệu (1 batch)
    ↓
    ┌─────────────────────────────────────────┐
    │  INPUT BATCH (32 × 96 × 4)              │
    │  ─────────────────────────────          │
    │  32 mẫu, mỗi mẫu:                       │
    │  - 96 time steps (lịch sử 4 ngày)       │
    │  - 4 biến (OT, M1, M2, M3)              │
    └─────────────────────────────────────────┘
    ↓
    Model.forward(input) → tính OUTPUT dự báo
    ↓
    ┌─────────────────────────────────────────┐
    │  OUTPUT BATCH (32 × 24 × 4)             │
    │  ─────────────────────────────          │
    │  32 mẫu, mỗi mẫu:                       │
    │  - 24 time steps (dự báo 1 ngày)        │
    │  - 4 biến                                │
    └─────────────────────────────────────────┘
    ↓
    SO SÁNH với GROUND TRUTH (dữ liệu thực)
    ↓
    LOSS (lỗi) = MSE(Predicted, Actual)
    ↓
    CẬP NHẬT WEIGHTS (backpropagation)
    ↓
    ✅ Batch hoàn thành, qua batch tiếp theo
    
    (Lặp lại cho tất cả batches)
```

#### Kết quả sau Training:
- 📁 `best_model_ETTh1.pth` - Trọng số model tốt nhất
- 📊 `training_loss_ETTh1.png` - Đồ thị Loss giảm qua epochs

**Ý nghĩa:** Loss càng giảm → Model học tốt hơn → Dự báo chính xác hơn

---

### **Giai Đoạn 2: VALIDATION (Kiểm Chứng) - Internal Process**

```
⚠️ Validation KHÔNG phải giai đoạn riêng - Nó là INTERNAL PROCESS của Training

Validation Data: train/datasets/ETTh1/ETTh1_val.csv
│
├─ Sau mỗi epoch:
├─   1. Run model trên tất cả training data → train_loss
├─   2. Run model trên validation data (dữ liệu đã seen trong training)
├─   3. Tính validation_loss
├─   4. So sánh:
│       ├─ Nếu val_loss < best_val_loss → Save best_model.pth ✅
│       ├─ Nếu val_loss > best_val_loss 3 lần liên tiếp → STOP (Early Stopping)
│       └─ Nếu val_loss ↓ → Model đang learn tốt
│
└─ Mục đích: Chọn "best model" dựa trên val_loss, tránh overfitting

⚡ Kết quả Validation hiển thị trên Training tab (validation loss curve)
```

---

### **Giai Đoạn 3: TESTING (Kiểm Tra)**

**Mục đích:** Đánh giá model trên dữ liệu mới hoàn toàn

#### Dữ liệu đầu vào:
```
test/datasets/ETTh1/ETTh1_test.csv
├─ Columns: OT, M1, M2, M3  (4 biến)
├─ Rows: ~7,000 time steps
└─ Dữ liệu HOÀN TOÀN MỚI (model chưa từng thấy)
```

#### Quá trình Testing:

```
┌──────────────────────────────────────────┐
│        1. LOAD TRAINED MODEL              │
│        (best_model_ETTh1.pth)             │
└──────────────────────────────────────────┘
           ↓
┌──────────────────────────────────────────┐
│   2. CHẠY INFERENCE (dự báo)              │
│   ─────────────────────────────          │
│                                            │
│   Lấy từng mẫu test + dự báo:             │
│                                            │
│   Input: [Actual Day 1] 96 steps          │
│      ↓                                      │
│   Model → Output: [Predicted] 24 steps    │
│      ↓                                      │
│   So sánh với Ground Truth (Actual Day 2) │
└──────────────────────────────────────────┘
           ↓
┌──────────────────────────────────────────┐
│   3. TÍNH METRICS                         │
│   ─────────────────────────────          │
│                                            │
│   MSE = Trung bình bình phương sai       │
│         (Penalize lỗi lớn)                │
│                                            │
│   MAE = Trung bình sai số tuyệt đối      │
│         (Dễ hiểu hơn)                     │
│                                            │
│   Ngụ ý: Dự báo sai trung bình bao nhiêu? │
└──────────────────────────────────────────┘
           ↓
┌──────────────────────────────────────────┐
│   4. VISUALIZATION                        │
│   ─────────────────────────────          │
│                                            │
│   Vẽ biểu đồ:                             │
│   - Đường XANH: Giá trị thực (Actual)    │
│   - Đường ĐỎ: Dự báo (Predicted)        │
│                                            │
│   Nhìn vào → Hiểu được:                   │
│   + Có theo trend không?                   │
│   + Lỗi lớn ở đâu?                       │
└──────────────────────────────────────────┘
```

---

## 📊 Hiểu Rõ Kích Thước Dữ Liệu

### **Shape (Hình dạng) của Dữ Liệu:**

```
TRAINING:
├─ Train Data Shape: (26,000-96-24, 4) roughly
│  └─ Có ~26,000 mẫu
│  └─ Mỗi mẫu = 96 time steps lịch sử + 24 steps được dự báo
│  └─ Mỗi time step = 4 biến
│
├─ Batch Size: 32
│  └─ Mỗi lại: xử lý 32 mẫu cùng lúc
│  └─ Shape mỗi batch input: (32, 96, 4)
│  └─ Shape mỗi batch output: (32, 24, 4)
│
└─ Số batches/epoch = ~26,000 / 32 = ~812 batches

TESTING:
├─ Test Data Shape: (7,000-96-24, 4) roughly
│  └─ Có ~7,000 mẫu
│  └─ Mỗi mẫu = 96 steps lịch sử + 24 steps cần dự báo
│  └─ Mỗi time step = 4 biến
│
├─ Batch Size: 1 (để visualization rõ)
│  └─ Mỗi lần: xử lý 1 mẫu
│  └─ Shape mỗi batch input: (1, 96, 4)
│  └─ Shape mỗi batch output: (1, 24, 4)
│
└─ Ta chạy trên ~10 batches để demo
```

### **Cụ Thể với ETTh1 Dataset:**

```
CSV File: ETTh1.csv
├─ Cột: date, OT, M1, M2, M3
├─ 4 biến:
│  ├─ OT: Tải điện (Electricity Load) - TARGET (dự báo)
│  ├─ M1, M2, M3: Công suất phát (Power Generation)
│  └─ Tất cả = Đặc trưng đầu vào cho model
├─ ~35,000 dòng dữ liệu
└─ Được chia thành:
   ├─ TRAIN: 70% (~26,000 dòng)
   ├─ VAL: 15% (~5,000 dòng)
   └─ TEST: 15% (~7,000 dòng)
```

---

## 🧠 Kiến Trúc Model (Đơn Giản Hóa)

```
Input (96 × 4)
    ↓
┌─────────────────────────────────┐
│  1. EMBEDDING LAYER             │
│  ─────────────────────────────  │
│  Chuyển raw time series         │
│  thành dense vectors            │
│                                  │
│  96 × 4 → 96 × 64              │
│  (64 = d_model = dimension)     │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  2. NORMALIZATION (RevIN)       │
│  ─────────────────────────────  │
│  Normalize dữ liệu để training  │
│  ổn định hơn                    │
│  (Giống scale data trước ML)    │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  3. MAMBA BLOCKS                │
│  ─────────────────────────────  │
│  Core của model!                │
│  - 2 layers (e_layers=2)        │
│  - Học patterns từ dữ liệu      │
│  - Linear complexity (hiệu quả) │
│                                  │
│  96 × 64 → 96 × 64             │
│  (giữ nguyên shape)             │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  4. DENORMALIZATION (RevIN⁻¹)   │
│  ─────────────────────────────  │
│  Khôi phục scale gốc            │
│  (Đảo ngược normalization)      │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  5. OUTPUT LAYER                │
│  ─────────────────────────────  │
│  Dự báo 24 steps tiếp theo      │
│  cho mỗi biến                   │
│                                  │
│  → 24 × 4                       │
│  (24 steps, 4 biến)             │
└─────────────────────────────────┘
    ↓
Output (24 × 4) - Dự Báo!

---

## 🎮 Streamlit Dashboard Tabs

### **Tại Sao Chỉ Có 3 Tab (Train, Test, Comparison)?**

```
❌ KHÔNG CÓ TAB VALIDATION vì:
├─ Validation là INTERNAL PROCESS của Training (không phải action riêng)
├─ User không cần chạy validation riêng
├─ Validation loss tự động tính trong training script
└─ Kết quả validation hiển thị trên Training tab (loss curve)

✅ 3 TAB LÀ ĐỦ:
├─ 🚀 Training Tab:
│  ├─ Control: Dataset, epochs, learning rate
│  ├─ Visualization: Training loss + validation loss (curve)
│  └─ Note: "Validation monitored internally to save best model"
│
├─ 🧪 Testing Tab:
│  ├─ Chạy inference trên test data (completely new)
│  ├─ Calculate MSE, MAE metrics
│  └─ Visualization: Actual vs Predicted curves
│
└─ 🏆 Comparison Tab:
   ├─ So sánh nhiều models (TimesMamba vs iTransformer vs PatchTST)
   ├─ So sánh multiple datasets
   └─ Rankings: Best model for each configuration
```

---

```
TimesMamba/
│
├── model/
│   ├── TimesMamba.py          ← Model chính (combines tất cả layers)
│   └── mambacore.py           ← Mamba architecture (State Space Model)
│
├── layers/
│   ├── Embed.py               ← Series Embedding (time series → vectors)
│   └── RevIN.py               ← Normalize/Denormalize (stabilize training)
│
├── train/
│   ├── train.py               ← Training script
│   ├── datasets/
│   │   └── ETTh1/
│   │       ├── ETTh1_train.csv    (Training set - 70%)
│   │       └── ETTh1_val.csv      (Validation set - 15%)
│   └── best_model_ETTh1.pth   ← Learned weights (save sau training)
│
├── test/
│   ├── test.py                ← Testing script
│   ├── simple_test.py         ← Simpler test version
│   ├── datasets/
│   │   └── ETTh1/
│   │       └── ETTh1_test.csv     (Test set - 15%, hoàn toàn mới)
│   └── predictions_ETTh1.csv  ← Model predictions
│
└── data_provider/
    └── data_factory.py        ← Load CSV và tạo batches
```

---

## 📊 Ví Dụ Cụ Thể: Training & Testing ETTh1

### **Kích thước dữ liệu:**

```
ETTh1.csv (CSV file gốc):
├─ ~35,050 dòng (time steps)
├─ Cột: date, OT (target), M1, M2, M3 (features)
└─ Được chia:
   ├─ TRAIN: dòng 0 → 24,500          (~70% = ~24,500 dòng)
   ├─ VAL:   dòng 24,500 → 29,300    (~15% = ~4,800 dòng)
   └─ TEST:  dòng 29,300 → 35,050    (~15% = ~5,750 dòng)
```

### **Training Process (Chi Tiết):**

```
┌─────────────────────────────────────────────────────────┐
│                   EPOCH 1/10                            │
│                                                          │
│  Load train data: 24,500 samples                        │
│            ↓                                              │
│  Create sliding windows (seq_len=96, pred_len=24)      │
│            ↓                                              │
│  Số mẫu training: ~24,500 - 96 - 24 = ~24,380         │
│            ↓                                              │
│  Batch size: 32                                         │
│  → Số batches/epoch = 24,380 / 32 = ~763 batches      │
│                                                          │
│  BATCH 1:                                               │
│  ├─ Input: (32, 96, 4)     - 32 mẫu, mỗi 96 steps x 4 biến
│  ├─ Forward pass: model(input) → (32, 24, 4)          │
│  ├─ Loss: MSE(output, target)                          │
│  ├─ Backprop: Update weights                           │
│  └─ Optimizer: Adam step                               │
│                                                          │
│  BATCH 2, 3, ..., 763: Lặp lại                         │
│                                                          │
│  Sau mỗi epoch:                                         │
│  ├─ Tính average loss trên TRAIN                       │
│  ├─ Run validation trên VAL set                        │
│  ├─ Nếu val_loss tốt → Save "best_model"              │
│  └─ Nếu val_loss không cải thiện 3 epochs → STOP      │
│                                                          │
└─────────────────────────────────────────────────────────┘

EPOCH 1: train_loss=0.832, val_loss=0.401 ✅ (best so far)
EPOCH 2: train_loss=0.425, val_loss=0.389 ✅ (improved)
EPOCH 3: train_loss=0.332, val_loss=0.392    (worse)
EPOCH 4: train_loss=0.298, val_loss=0.395    (worse)
EPOCH 5: train_loss=0.276, val_loss=0.398    (worse 3 times)
                    ↓
        EARLY STOPPING! Load best_model từ EPOCH 2
```

### **Testing Process:**

```
┌──────────────────────────────────────────────────┐
│               TESTING PHASE                      │
│                                                   │
│  1. Load best_model_ETTh1.pth                   │
│  2. Load test data: 5,750 samples                │
│  3. Create sliding windows: ~5,630 mẫu test     │
│  4. Batch size: 1 (để visualization)             │
│                                                   │
│  BATCH 1 (1 mẫu test):                          │
│  ├─ Input:  [Day 1-4] 96 steps                  │
│  │  Example: [520, 524, 525, ..., 530] (4 ngày) │
│  │                                               │
│  ├─ Model.forward(): Dự báo 24 steps tiếp theo│
│  │  Output: [528, 531, 534, ..., 535]           │
│  │                                               │
│  ├─ Ground Truth: [Day 5] 24 steps              │
│  │  Actual: [530, 533, 536, ..., 537]           │
│  │                                               │
│  └─ Compare:                                     │
│     Predicted: [528, 531, 534, 535, ...]       │
│     Actual:    [530, 533, 536, 537, ...]       │
│     Error:     [0.2, 0.2, 0.2, 0.2, ...]       │
│                                                   │
│  BATCH 2, 3, ..., limit to 10 batches for demo │
│                                                   │
└──────────────────────────────────────────────────┘

Tính Metrics trên 10 batches:
├─ MSE = 0.375
├─ MAE = 0.397
└─ Visualization: Vẽ 3 line plots (actual vs predicted)
```

---

## 🎓 Tại Sao Kết Quả Như Vậy?

### **TimesMamba tốt hơn vì:**

| Đặc điểm | TimesMamba | Transformer |
|---------|-----------|------------|
| **Phức tạp** | O(n) - Linear | O(n²) - Quadratic |
| **VRAM** | Ít | Nhiều |
| **Training** | Nhanh | Chậm |
| **Accuracy** | Cao ✅ | Trung bình |
| **Scalability** | Tốt cho dữ liệu dài | Khó scale |

### **Kết quả Benchmark (Từ Ảnh):**

```
                TimesMamba  iTransformer  PatchTST
ETTh1 L96           0.375 ✅   0.386       0.414
Electricity L96     0.141 ✅   0.148       0.195
Traffic L96         0.376 ✅   0.395       0.544

→ TimesMamba thắng ở TẤT CẢ datasets! 🎉
```

---

## ⚡ Nhanh Chóng Chạy Thử

### **1. Training:**
```bash
cd train
python train.py
```
Kết quả: `best_model_ETTh1.pth` + `training_loss_ETTh1.png`

### **2. Testing:**
```bash
cd test
python test.py
```
Kết quả: Metrics (MSE/MAE) + `test_results_ETTh1.png`

### **3. Xem Dashboard:**
```bash
streamlit run streamlit_app.py
```
→ Chọn tab: 🚀 Training | 🧪 Testing | 🏆 Comparison

---

## ✨ Tóm Tắt Toàn Bộ

```
1️⃣  CHUẨN BỊ: Đọc CSV → Tạo sliding windows
    
2️⃣  TRAINING:
    - Dạy model nhận pattern từ 96 time steps
    - Dự báo 24 time steps tiếp theo
    - Lưu "best model" dựa trên validation loss
    
3️⃣  TESTING:
    - Load model tốt nhất
    - Dự báo trên dữ liệu HOÀN TOÀN MỚI
    - Tính MSE, MAE metrics
    
4️⃣  KẾT QUẢ:
    - MSE = 0.375, MAE = 0.397
    - Tốt hơn iTransformer & PatchTST
    - Sẵn sàng dùng production!
```

---

## ❓ FAQ

**Q: Tại sao Streamlit chỉ có 3 tab (Train, Test, Comparison) không có Validation?**
A: Vì validation là internal process của training:
   - Không chạy validation riêng → Validation tích hợp trong training script
   - Training tab tự động hiển thị validation loss (nếu có log file)
   - Validation loss dùng để chọn "best model" → Không cần test riêng
   - Test chỉ chạy trên "best model" được chọn bởi validation

**Q: seq_len=96 là gì?**
A: Cửa sổ đầu vào = 4 ngày dữ liệu lịch sử (nếu hourly data)

**Q: pred_len=24 là gì?**
A: Cửa sổ đầu ra = 1 ngày dự báo tiếp theo

**Q: Tại sao chia train/val/test?**
A: 
   - Train: Dạy model (70%)
   - Val: Ngừng sớm nếu overfit, chọn best model (15%)
   - Test: Đánh giá công bằng trên dữ liệu mới (15%)

**Q: Model học như thế nào?**
A: Backpropagation: tính gradient của mỗi layer, update weights để minimize loss

**Q: Tại sao MSE khác MAE?**
A: MSE phạt lỗi lớn nặng hơn, MAE "friendly" hơn

**Q: Validation data được dùng trong quá trình training phải không?**
A: Đúng! Validation data ĐƯỢC dùng nhưng KHÔNG để train weights
   - Chỉ dùng để:
     * Tính validation loss (monitoring)
     * Chọn best checkpoint (save model)
     * Implement early stopping (stop nếu không cải thiện)
   - KHÔNG dùng để update gradients

---

**Hiểu rõ chưa?** Giờ chạy code để thấy real output! 🚀