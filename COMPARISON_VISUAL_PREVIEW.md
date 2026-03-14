# 📊 Model Comparison Dashboard - Visual Preview

## What You'll See in Streamlit

After running `streamlit run streamlit_app.py` and clicking the **🏆 Comparison** tab:

### Section 1: Experiment Summary
```
┌─────────────────────────────────────────────────────────┐
│         📊 Experiment Summary                            │
├─────────────┬─────────────┬─────────────┬──────────────┤
│ Total      │ Models      │ Datasets    │ Avg MSE      │
│ Experiments│            │            │              │
├─────────────┼─────────────┼─────────────┼──────────────┤
│     36      │      3      │      3      │   0.3874     │
└─────────────┴─────────────┴─────────────┴──────────────┘
```

### Section 2: MSE Comparison Table (Green = Best)

```
📈 Model Comparison by Dataset & Prediction Length
[MSE Comparison]  [MAE Comparison]

                          TimesMamba  iTransformer  PatchTST
ETTh1_L96                    0.375 ✅   0.386       0.414
ETTh1_L192                   0.425 ✅   0.441       0.460
ETTh1_L336                   0.468 ✅   0.487       0.501
ETTh1_L720                   0.479 ✅   0.503       0.500
Electricity_L96              0.141 ✅   0.148       0.195
Electricity_L192             0.156 ✅   0.162       0.199
Electricity_L336             0.172 ✅   0.178       0.215
Electricity_L720             0.204 ✅   0.225       0.256
Traffic_L96                  0.376 ✅   0.395       0.544
Traffic_L192                 0.386 ✅   0.417       0.540
Traffic_L336                 0.412 ✅   0.433       0.551
Traffic_L720                 0.458 ✅   0.467       0.586
```

**TimesMamba wins all comparisons!** ✅

### Section 3: Detailed Comparison View

```
🔍 Detailed Comparison View

Select Models to Compare:
 ☑ TimesMamba
 ☑ iTransformer  
 ☑ PatchTST

Select Datasets:
 ☑ ETTh1
 ☑ Electricity
 ☑ Traffic

[Shows filtered comparison table + line charts showing MSE vs Prediction Length]
```

### Section 4: Best Model Rankings

```
🥇 Best Model Rankings

Best Models by MSE              Best Models by MAE
┌──────────────────────┐       ┌──────────────────────┐
│Dataset   Pred Len    │       │Dataset   Pred Len    │
│          Best Model  │       │          Best Model  │
├──────────────────────┤       ├──────────────────────┤
│ETTh1     96          │       │ETTh1     96          │
│          TimesMamba  │       │          TimesMamba  │
│          0.375       │       │          0.397       │
│                      │       │                      │
│ETTh1     192         │       │ETTh1     192         │
│          TimesMamba  │       │          TimesMamba  │
│          0.425       │       │          0.424       │
│...                   │       │...                   │
└──────────────────────┘       └──────────────────────┘
```

### Section 5: Results Management

```
⚙️ Manage Results

[Add New Experimental Result ▼]
  Model Name:        [_________________]
  Dataset Name:      [_________________]
  Prediction Length: [96________]
  MSE:               [0.000000__]
  MAE:               [0.000000__]
  Notes:             [_________________]
  
  [➕ Add Result] [🗑️ Clear All]
```

## Key Features You Get

### ✅ **1. Automatic Best Value Highlighting**
- Green cells mark the lowest (best) MSE/MAE in each row
- Instantly see which model performs best

### ✅ **2. Multi-Metric Display**
- Toggle between MSE and MAE tabs
- Compare different evaluation metrics

### ✅ **3. Flexible Filtering**
- Select specific models to compare
- Choose specific datasets
- Results update dynamically

### ✅ **4. Visual Charts**
- Line plots showing MSE vs Prediction Length
- Trends across different horizons
- Multiple datasets side-by-side

### ✅ **5. Automatic Rankings**
- Best model identified for each dataset/pred_len
- Ranked by both MSE and MAE
- Shows exact metric values

### ✅ **6. Data Export**
- Download results as CSV
- Use in Excel or other Analysis tools
- Timestamp tracking for all results

## Example: How to Use It

### Scenario 1: Compare All Models
1. Go to **🏆 Comparison** tab
2. View default MSE Comparison table
3. **Result**: Instantly see TimesMamba beats all others in green ✅

### Scenario 2: Compare Selection
1. Select only **TimesMamba** and **iTransformer**
2. Select only **ETTh1** dataset
3. See side-by-side performance on ETTh1 data

### Scenario 3: Add Your Results
1. Train your model on a new dataset
2. Get MSE, MAE scores
3. Click **Add Result** in management section
4. Fill in form and click ➕
5. **Refresh browser (F5)**
6. New results appear in tables!

## Data Behind the Scenes

The `experiment_results/results.json` file stores:

```json
{
  "TimesMamba_ETTh1_96": {
    "model": "TimesMamba",
    "dataset": "ETTh1",
    "pred_len": 96,
    "mse": 0.375,
    "mae": 0.397,
    "notes": "Best performance",
    "timestamp": "2026-03-14T13:54:00.111715"
  }
}
```

Each experiment has:
- 🏷️ **Unique ID**: model_dataset_predlen
- 📊 **Metrics**: MSE, MAE, RMSE (optional), MAPE (optional)
- 📝 **Context**: Notes, timestamp
- 🔄 **Trackable**: When was it run

## Next Steps

### 1️⃣ **Run Example** (Already Done ✅)
```bash
python add_results.py example
```

### 2️⃣ **View in Streamlit**
```bash
streamlit run streamlit_app.py
# Click 🏆 Comparison tab
```

### 3️⃣ **Add Your Own Results**
- Train TimesMamba, iTransformer, PatchTST on your datasets
- Record MSE, MAE scores
- Add to system via interactive UI or script
- Watch comparison table update!

### 4️⃣ **Export & Analyze**
- Download CSV from Streamlit
- Create presentations/reports
- Share results with team

## Which Table Matches Your Image?

**Your original image showed:**
```
        TimesMamba  iTransformer  PatchTST
ETTh1
  96      0.375      0.386       0.414
  192     0.425      0.441       0.460
  336     0.468      0.487       0.501
  720     0.479      0.503       0.500
Electricity
  96      0.141      0.148       0.195
  192     0.156      0.162       0.199
  336     0.172      0.178       0.215
  720     0.204      0.225       0.256
Traffic
  96      0.376      0.395       0.544
  192     0.386      0.417       0.540
  336     0.412      0.433       0.551
  720     0.458      0.467       0.586
```

**You now have it as:**
```
                          TimesMamba  iTransformer  PatchTST
ETTh1_L96                    0.375       0.386      0.414
ETTh1_L192                   0.425       0.441      0.460
...
```

✅ **Exact same data, just reorganized for easier lookup!**

---

Ready to go? 🚀 Start with:
```bash
streamlit run streamlit_app.py
```
