# 🎯 Model Training & Comparison Guide

## 📌 Tổng Quan

Project này so sánh **3 models** trên **4 datasets** sử dụng cùng data split:
- ✅ **TimesMamba** (State Space Model - SOTA)
- ✅ **RNN** (LSTM-based baseline)
- ✅ **ITransformer** (Transformer baseline)

**Datasets:** ETTh1, ETTh2, ETTm1, ETTm2

---

## 📁 Cấu Trúc Thư Mục

```
TimesMamba/
├── train/                          # TimesMamba training & testing
│   ├── train.py                    # Training script
│   ├── datasets/
│   │   ├── ETTh1/
│   │   │   ├── ETTh1_train.csv     # 70% training data
│   │   │   └── ETTh1_val.csv       # 15% validation data
│   │   ├── ETTh2/, ETTm1/, ETTm2/
│   │   └── ...
│   └── best_model_*.pth            # Trained models
│
├── test/                           # TimesMamba testing
│   ├── test.py / simple_test.py    # Testing script
│   ├── datasets/
│   │   ├── ETTh1/
│   │   │   └── ETTh1_test.csv      # 15% test data
│   │   └── ...
│   └── predictions_*.csv           # Test results
│
├── RNN/                            # RNN model (NEW)
│   ├── model_rnn.py                # RNN/LSTM/GRU model
│   ├── train_rnn.py                # Training script
│   ├── test_rnn.py                 # Testing script
│   ├── saved_models/
│   │   └── best_model_RNN_*.pth
│   ├── figures/
│   │   └── training_loss_RNN_*.png
│   └── predictions/
│       └── results_RNN_*.csv
│
├── ITransformer/                   # ITransformer model (NEW)
│   ├── model_itransformer.py       # ITransformer model
│   ├── train_itransformer.py       # Training script
│   ├── test_itransformer.py        # Testing script
│   ├── saved_models/
│   │   └── best_model_ITransformer_*.pth
│   ├── figures/
│   │   └── training_loss_ITransformer_*.png
│   └── predictions/
│       └── results_ITransformer_*.csv
│
├── streamlit_app.py                # Dashboard (updated)
├── results_manager.py              # Results storage & comparison
├── extract_and_compare.py          # AUTO-RUN all models (NEW)
└── EXPLANATION.md                  # Documentation
```

---

## 🚀 Cách Chạy

### **Option 1: Auto-Run All Models (Recommended)**

Chạy tất cả 3 models + test + so sánh tự động:

```bash
python extract_and_compare.py
```

This script:
1. ✅ Trains TimesMamba, RNN, ITransformer (sequential, ~30-60 min)
2. ✅ Tests all models on test data
3. ✅ Extracts metrics từ prediction files
4. ✅ Populates `experiment_results/results.json`
5. ✅ Displays summary statistics

Sau khi hoàn thành, chạy dashboard:

```bash
streamlit run streamlit_app.py
```

---

### **Option 2: Train Models Individually**

#### **Train TimesMamba:**
```bash
cd train
python train.py
cd ..
```

#### **Train RNN:**
```bash
cd RNN
python train_rnn.py
cd ..
```

#### **Train ITransformer:**
```bash
cd ITransformer
python train_itransformer.py
cd ..
```

---

### **Option 3: Test Models Individually**

#### **Test TimesMamba:**
```bash
cd test
python simple_test.py
cd ..
```

#### **Test RNN:**
```bash
cd RNN
python test_rnn.py
cd ..
```

#### **Test ITransformer:**
```bash
cd ITransformer
python test_itransformer.py
cd ..
```

---

## 📊 View Results

### **In Streamlit Dashboard:**

```bash
streamlit run streamlit_app.py
```

Then navigate to **🏆 Comparison** tab to:
- ✅ View MSE/MAE comparison tables
- ✅ See which model is best on each dataset
- ✅ Visualize performance across prediction lengths
- ✅ Download results as CSV

### **In JSON Format:**

Results are stored in: `experiment_results/results.json`

Example:
```json
{
  "TimesMamba_ETTh1_24": {
    "model": "TimesMamba",
    "dataset": "ETTh1",
    "pred_len": 24,
    "mse": 0.375,
    "mae": 0.397
  },
  "RNN_ETTh1_24": {
    "model": "RNN",
    "dataset": "ETTh1",
    "pred_len": 24,
    "mse": 0.450,
    "mae": 0.445
  },
  "ITransformer_ETTh1_24": {
    "model": "ITransformer",
    "dataset": "ETTh1",
    "pred_len": 24,
    "mse": 0.410,
    "mae": 0.420
  }
}
```

---

## 🔍 What Each Model Does

### **TimesMamba** (train/)
- **Architecture:** Mamba (State Space Model)
- **Complexity:** O(n) Linear
- **Speed:** Fast ⚡
- **SOTA Performance:** Best accuracy
- **Files Used:** 
  - Training: `train/datasets/{dataset}/{dataset}_train.csv`
  - Testing: `test/datasets/{dataset}/{dataset}_test.csv`

### **RNN** (RNN/)
- **Architecture:** LSTM/GRU Recurrent Neural Network
- **Complexity:** O(n) Sequential
- **Speed:** Moderate
- **Baseline:** Good for comparison
- **Files Used:**
  - Shared dataset: `train/datasets/`
  - Results: `RNN/predictions/results_RNN_*.csv`

### **ITransformer** (ITransformer/)
- **Architecture:** Individual Series Transformer
- **Complexity:** O(n) per channel (independent attention)
- **Speed:** Fast
- **Novel Approach:** Process each variable independently
- **Files Used:**
  - Shared dataset: `train/datasets/`
  - Results: `ITransformer/predictions/results_ITransformer_*.csv`

---

## 📊 Data Split (Same for all models)

```
Original Data: ~35,000 timesteps
                    ↓
          ┌─────────┼─────────┐
          ↓         ↓         ↓
       70% (Train) 15% (Val) 15% (Test)
     ~24,500    ~5,250    ~5,250
        ↓         ↓         ↓
     teach      select    evaluate
     model      best       final
              checkpoint  metrics
```

**Key Point:** All 3 models use the **SAME data split** for fair comparison!

---

## 🎯 Expected Results (ETTh1)

Based on typical performance:

| Model | MSE | MAE | Speed |
|-------|-----|-----|-------|
| TimesMamba | 0.37-0.40 | 0.39-0.42 | ⚡⚡⚡ Fast |
| RNN | 0.42-0.48 | 0.43-0.50 | ⚡⚡ Moderate |
| ITransformer | 0.38-0.45 | 0.40-0.48 | ⚡⚡⚡ Fast |

*(Exact results depend on random initialization and hyperparameters)*

---

## ⚙️ Troubleshooting

### **If training fails:**
```bash
# Check dependencies
pip install torch pandas scikit-learn matplotlib

# Verify data exists
ls train/datasets/ETTh1/

# Check CUDA (optional)
python -c "import torch; print(torch.cuda.is_available())"
```

### **If test fails:**
```bash
# Make sure model was trained first
ls RNN/saved_models/best_model_RNN_ETTh1.pth

# Manually run test script with debug
cd RNN
python test_rnn.py
```

### **If results don't appear in Streamlit:**
```bash
# Check if results.json was created
cat experiment_results/results.json

# Check if results were extracted
python -c "from results_manager import ResultsManager; m = ResultsManager(); print(m.get_summary_stats())"

# Refresh Streamlit (Ctrl+C then restart)
streamlit run streamlit_app.py
```

---

## 📈 Performance Metrics Explained

### **MSE (Mean Squared Error)**
- Punishes large errors more (squared term)
- Better for minimizing extreme predictions
- Formula: `MSE = mean((y_pred - y_true)²)`

### **MAE (Mean Absolute Error)**
- Average absolute deviation
- More interpretable (same units as data)
- Formula: `MAE = mean(|y_pred - y_true|)`

### **RMSE (Root Mean Squared Error)**
- Square root of MSE
- Back to original data scale
- Formula: `RMSE = sqrt(MSE)`

**Lower is always better!** ↓↓↓

---

## 💾 Save & Export Results

### **In Streamlit Dashboard:**
- Click **"📥 Download Results as CSV"** button
- Exports all results to `model_comparison.csv`

### **From Command Line:**
```python
from results_manager import ResultsManager

manager = ResultsManager()
csv_path = manager.export_to_csv()
print(f"Results exported to: {csv_path}")
```

---

## 🔄 Re-run Everything

To start fresh and re-train all models:

```bash
# Clear old results
rm -rf RNN/saved_models RNN/figures RNN/predictions
rm -rf ITransformer/saved_models ITransformer/figures ITransformer/predictions
rm experiment_results/results.json

# Re-run auto extraction
python extract_and_compare.py

# View results
streamlit run streamlit_app.py
```

---

## 📚 Reference

- **TimesMamba Paper:** [arxiv.org/abs/2310.12541](https://arxiv.org/abs/2310.12541)
- **ITransformer Paper:** [arxiv.org/abs/2310.06625](https://arxiv.org/abs/2310.06625)
- **Dataset:** ETT (Electricity Transforming Temperature)

---

## ✨ Summary

```
🎯 Goal: Compare 3 models on 4 datasets using fair data split

📊 Models:
   1. TimesMamba (SOTA)
   2. RNN (Baseline)
   3. ITransformer (Transformer)

📈 Datasets:
   1. ETTh1 (hourly)
   2. ETTh2 (hourly)
   3. ETTm1 (15-min)
   4. ETTm2 (15-min)

⚡ Quick Start:
   python extract_and_compare.py
   streamlit run streamlit_app.py

📊 Results: experiment_results/results.json
💾 Dashboard: http://localhost:8501 (🏆 Comparison tab)
```

---

Made with ❤️ for Time Series Forecasting! 🚀
