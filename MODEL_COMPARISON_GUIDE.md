# 🏆 Model Comparison Feature Guide

## Overview

This guide shows you how to use the **new model comparison feature** to compare TimesMamba against other models (iTransformer, PatchTST, etc.) with results organized by dataset and prediction length.

## Quick Start

### Step 1: Load Example Data

To see the comparison feature in action with sample data:

```bash
python add_results.py example
```

This will load example comparison results like the image you provided.

### Step 2: Start Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Then navigate to the **🏆 Comparison** tab to see the model comparison tables.

## Features

### 📊 Comparison Tab Features

1. **Experiment Summary**
   - Total number of experiments
   - Number of models compared
   - Number of datasets
   - Average metrics

2. **MSE/MAE Comparison Tables**
   - Side-by-side comparison of all models
   - Green highlight shows the best (lowest) value
   - Rows = Datasets + Prediction Lengths
   - Columns = Models

3. **Detailed Comparison View**
   - Filter by specific models
   - Filter by specific datasets
   - Performance visualization with charts
   - Shows trends across prediction lengths

4. **Best Model Rankings**
   - Automatically identifies best model for each dataset/prediction length
   - Ranked by MSE and MAE metrics

5. **Results Management**
   - Add new results manually
   - Clear all results when needed
   - Export to CSV

## Adding Your Own Results

### Option 1: Interactive Mode

```bash
python add_results.py interactive
```

You'll be prompted to enter:
- Model name
- Dataset name
- Prediction length
- MSE and MAE scores
- Optional RMSE, MAPE, and notes

### Option 2: Python Script

```python
from results_manager import ResultsManager

manager = ResultsManager()

# Add a single result
manager.add_result(
    model_name="TimesMamba",
    dataset="ETTh1",
    pred_len=96,
    mse=0.375,
    mae=0.397,
    notes="Trained with learning_rate=0.001"
)

# Add multiple results
results = [
    {"model": "TimesMamba", "dataset": "ETTh1", "pred_len": 96, "mse": 0.375, "mae": 0.397},
    {"model": "iTransformer", "dataset": "ETTh1", "pred_len": 96, "mse": 0.386, "mae": 0.405},
    {"model": "PatchTST", "dataset": "ETTh1", "pred_len": 96, "mse": 0.414, "mae": 0.419},
]

from add_results import bulk_import_from_dict
bulk_import_from_dict(results)
```

### Option 3: Using Streamlit UI

In the Streamlit dashboard:
1. Go to **🏆 Comparison** tab
2. Expand **⚙️ Manage Results**
3. Click **Add New Experimental Result**
4. Fill in the form and click **➕ Add Result**

## Understanding the Table Format

The comparison table shows results like this:

```
                          TimesMamba  iTransformer  PatchTST
ETTh1_L96                     0.375       0.386      0.414
ETTh1_L192                    0.425       0.441      0.460
ETTh1_L336                    0.468       0.487      0.501
ETTh1_L720                    0.479       0.503      0.500
...
```

Where:
- **Rows**: Dataset name + prediction Length (L96, L192, L336, L720)
- **Columns**: Model names
- **Values**: MSE/MAE scores (lower is better)
- **Green**: Best (lowest) value in each row

## Export & Analysis

### Export Results to CSV

Results are automatically exported to `experiment_results/results.csv`

### View in Excel/Pandas

```python
import pandas as pd

df = pd.read_csv('experiment_results/results.csv')
df.to_excel('comparison.xlsx', index=False)
```

### Get Summary Statistics

```python
from results_manager import ResultsManager

manager = ResultsManager()
summary = manager.get_summary_stats()

print(f"Total experiments: {summary['total_experiments']}")
print(f"Models compared: {summary['models']}")
print(f"Average MSE: {summary['mse_stats']['mean']:.4f}")
```

## Tips for Best Results

1. **Consistent Dataset Names**: Use the same dataset names across experiments
   - Good: "ETTh1", "Electricity", "Traffic"
   - Bad: "ETTh1", "ETT_h1", "ett_h1" (these create separate entries)

2. **Standard Prediction Lengths**: Use common values like 96, 192, 336, 720

3. **Add Context**: Use the "notes" field to record:
   - Training hyperparameters
   - Hardware used (CPU/GPU)
   - Training duration
   - Any modifications made

4. **Multiple Runs**: If you run the same model multiple times, the manager will store all runs with unique timestamps

## File Structure

```
TimesMamba/
├── results_manager.py          # Main results management module
├── add_results.py              # Quick add script
├── MODEL_COMPARISON_GUIDE.md   # This file
├── streamlit_app.py            # Updated with comparison tab
└── experiment_results/
    ├── results.json            # All stored results
    └── results.csv             # Exported CSV
```

## API Reference

### ResultsManager Class

```python
from results_manager import ResultsManager

manager = ResultsManager(results_dir="experiment_results")

# Add a result
manager.add_result(
    model_name: str,     # Required
    dataset: str,        # Required
    pred_len: int,       # Required
    mse: float,          # Required
    mae: float,          # Required
    rmse: float = None,  # Optional
    mape: float = None,  # Optional
    notes: str = "",     # Optional
    timestamp: str = None # Auto if None
)

# Get comparison data
df = manager.get_model_comparison(
    models=['TimesMamba', 'iTransformer'],
    datasets=['ETTh1'],
    pred_lens=[96, 192]
)

# Get pivot table
pivot = manager.get_pivot_table(
    metric='mse',  # or 'mae', 'rmse', 'mape'
    models=['TimesMamba', 'iTransformer'],
    datasets=['ETTh1']
)

# Get best model for a dataset
best = manager.get_best_model(
    dataset='ETTh1',
    pred_len=96,
    metric='mse'
)  # Returns (model_name, score)

# Export to CSV
csv_path = manager.export_to_csv(filename='my_results.csv')

# Get summary
summary = manager.get_summary_stats()

# Clear all results (if needed)
manager.clear_results()
```

## Troubleshooting

**Q: Why doesn't my added result appear in the table?**
A: Results are saved to disk. Refresh your Streamlit browser tab (F5) to see new results.

**Q: Can I modify or delete a single result?**
A: Currently, results can only be fully cleared. For selective deletion, edit the `experiment_results/results.json` file directly.

**Q: How do I import results from a CSV file?**
A: You can manually add them using the interactive mode or write a Python script:

```python
import pandas as pd
from add_results import bulk_import_from_dict

df = pd.read_csv('my_results.csv')
results_list = df.to_dict('records')
bulk_import_from_dict(results_list)
```

## Next Steps

1. ✅ Load example data: `python add_results.py example`
2. ✅ Start dashboard: `streamlit run streamlit_app.py`
3. ✅ View comparison in **🏆 Comparison** tab
4. ✅ Add your own results from model training/testing
5. ✅ Analyze and compare model performance

Happy comparing! 🎯
