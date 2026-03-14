# 🚀 QUICK START - Train & Compare 3 Models

## ⚡ 30-Second Quick Start

```bash
# Run everything automatically (train all 3 models + test + compare)
python extract_and_compare.py

# View results in browser
streamlit run streamlit_app.py
```

Then go to **🏆 Comparison** tab to see which model is best!

---

## 📋 What Happens Automatically

```
1️⃣  TRAIN PHASE (~15-20 min):
   ├─ TimesMamba (5 min)  → best_model_TimesMamba.pth
   ├─ RNN (5 min)         → best_model_RNN.pth
   └─ ITransformer (5 min) → best_model_ITransformer.pth

2️⃣  TEST PHASE (~5-10 min):
   ├─ TimesMamba test  → MSE=?, MAE=?
   ├─ RNN test        → MSE=?, MAE=?
   └─ ITransformer test → MSE=?, MAE=?

3️⃣  AUTO-POPULATE RESULTS:
   └─ experiment_results/results.json ✅

4️⃣  VIEW IN STREAMLIT:
   ├─ 📊 Comparison tables
   ├─ 🥇 Rankings (best model)
   ├─ 📈 Visualizations
   └─ 💾 Export as CSV
```

---

## 3️⃣ Models Explained in 10 Seconds Each

### **🔵 TimesMamba** (SOTA Performance)
- Uses State Space Model (Mamba architecture)
- Fastest & most accurate
- O(n) complexity
- Expected: MSE 0.37-0.40 🥇

### **🟢 RNN** (Classic Baseline)
- LSTM/GRU based
- Good for comparison
- O(n) sequential computation
- Expected: MSE 0.42-0.48

### **🟡 ITransformer** (Novel Approach)
- Individual attention per channel
- No channel mixing
- Fast Transformer variant
- Expected: MSE 0.38-0.45

---

## 💾 Results Storage

All results saved to: `experiment_results/results.json`

```json
{
  "TimesMamba_ETTh1_24": {
    "mse": 0.375,
    "mae": 0.397
  },
  "RNN_ETTh1_24": {
    "mse": 0.450,
    "mae": 0.445
  },
  "ITransformer_ETTh1_24": {
    "mse": 0.410,
    "mae": 0.420
  }
}
```

---

## 🎯 Next Steps

### After Training Completes:

1. **View Dashboard** (localhost:8501):
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Go to 🏆 Comparison Tab:**
   - See MSE/MAE comparison tables
   - Check rankings (which model is best)
   - Download results as CSV

3. **Understanding Results:**
   - Lower MSE = better performance
   - TimesMamba usually wins (SOTA)
   - Compare across datasets (ETTh1, ETTh2, etc.)

---

## ❓ FAQ - Quick Answers

**Q: How long does this take?**
A: ~30-60 minutes (depends on GPU vs CPU)

**Q: Do I need GPU?**
A: No, CPU works but slower

**Q: What if training fails?**
A: Check `MODEL_TRAINING_GUIDE.md` for troubleshooting

**Q: Can I train just one model?**
A: Yes, see `MODEL_TRAINING_GUIDE.md` Option 2

**Q: Where are models saved?**
A: 
   - TimesMamba: `train/best_model_*.pth`
   - RNN: `RNN/saved_models/best_model_RNN_*.pth`
   - ITransformer: `ITransformer/saved_models/best_model_ITransformer_*.pth`

**Q: Can I see training curves?**
A: Yes! Check `*/figures/training_loss_*.png`

---

## 🔄 File Structure (What Gets Created)

```
TimesMamba/ (root)
├── train/
│   ├── datasets/      (CSV data - SHARED)
│   └── best_model_*.pth
│
├── test/
│   ├── datasets/      (test CSV - SHARED)
│   └── predictions_*.csv
│
├── RNN/               (NEW)
│   ├── saved_models/
│   │   └── best_model_RNN_*.pth
│   ├── figures/
│   │   └── training_loss_RNN_*.png
│   └── predictions/
│       └── results_RNN_*.csv
│
├── ITransformer/      (NEW)
│   ├── saved_models/
│   │   └── best_model_ITransformer_*.pth
│   ├── figures/
│   │   └── training_loss_ITransformer_*.png
│   └── predictions/
│       └── results_ITransformer_*.csv
│
├── experiment_results/
│   └── results.json   ← comparison data
│
└── streamlit_app.py
```

---

## 👉 Ready? Just Run This:

```bash
python extract_and_compare.py
```

The script will:
1. Train 3 models sequentially
2. Test each model
3. Extract metrics automatically
4. Show you the summary
5. Tell you to run: `streamlit run streamlit_app.py`

---

## 📚 Learn More

- Full guide: `MODEL_TRAINING_GUIDE.md`
- Detailed explanation: `EXPLANATION.md`
- Models overview: `EXPLANATION.md` (新 section: "3 Models được so sánh")

---

## 🎉 That's It!

```
1. python extract_and_compare.py  ← Trains & compares
2. streamlit run streamlit_app.py  ← View results
3. Click 🏆 Comparison tab        ← See which model wins!
```

**Made with ❤️ for Time Series Forecasting! 🚀**
