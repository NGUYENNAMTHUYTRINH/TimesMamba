# 🔗 Integrate Test Results into Comparison

This guide shows how to automatically add results from your TimesMamba, iTransformer, and PatchTST test runs into the comparison system.

## Method 1: Parse Test Results Automatically

### From Your Current test.py Output

If your test script outputs metrics, you can extract them:

```python
# In your test script (after testing)
import json
from results_manager import ResultsManager

# After getting MSE and MAE from testing
mse = 0.375
mae = 0.397
dataset = "ETTh1"
pred_len = 96

# Add to comparison
manager = ResultsManager()
manager.add_result(
    model_name="TimesMamba",
    dataset=dataset,
    pred_len=pred_len,
    mse=mse,
    mae=mae,
    notes="Test run on CPU"
)

print(f"✅ Result added: {dataset} L{pred_len} - MSE: {mse:.4f}")
```

### Update Your test/test.py

Add this at the end of your test function:

```python
def test_model(args, model_path=None):
    """Test TimesMamba model"""
    # ... existing test code ...
    
    # Calculate metrics
    mse = mean_squared_error(targets.flatten(), predictions.flatten())
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    
    print(f"[RESULTS] Test Results for {args.data}:")
    print(f"   MSE: {mse:.6f}")
    print(f"   MAE: {mae:.6f}")
    
    # 🆕 ADD THIS: Save to comparison
    from results_manager import ResultsManager
    manager = ResultsManager()
    manager.add_result(
        model_name="TimesMamba",  # Change for other models
        dataset=args.data,
        pred_len=args.pred_len,
        mse=mse,
        mae=mae,
        notes=f"Test on {args.data} with pred_len={args.pred_len}"
    )
    print(f"✅ Result saved to comparison database")
    
    # ... rest of function ...
```

## Method 2: Batch Import Results

### From CSV File

If you have results in a CSV file:

```python
import pandas as pd
from add_results import bulk_import_from_dict

# Read CSV
df = pd.read_csv('my_test_results.csv')

# Expected columns: model, dataset, pred_len, mse, mae
# Optional: rmse, mape, notes

results_list = df.to_dict('records')
bulk_import_from_dict(results_list)

print("✅ All results imported!")
```

### CSV Format Expected:

```csv
model,dataset,pred_len,mse,mae,notes
TimesMamba,ETTh1,96,0.375,0.397,Initial test
TimesMamba,ETTh1,192,0.425,0.424,With augmentation
iTransformer,ETTh1,96,0.386,0.405,Baseline
PatchTST,ETTh1,96,0.414,0.419,Standard config
```

## Method 3: Training Loop Integration

### Add Tracking During Training

```python
# In your train/train.py

from results_manager import ResultsManager

# Initialize results manager
results_manager = ResultsManager()

# In your training/validation loop
for epoch in range(epochs):
    # ... training code ...
    
    # Validate periodically
    if epoch % 10 == 0:
        val_mse = validate(model, val_loader)
        val_mae = compute_mae(model, val_loader)
        
        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")
        
        # 🆕 Track in comparison
        results_manager.add_result(
            model_name="TimesMamba",
            dataset=args.data,
            pred_len=args.pred_len,
            mse=val_mse,
            mae=val_mae,
            notes=f"Epoch {epoch} checkpoint"
        )
        
        print(f"Epoch {epoch}: MSE={val_mse:.4f}, Status: saved to comparison")
```

## Method 4: Bulk Add Multiple Models

### Compare 3 Models on 4 Datasets

```python
from results_manager import ResultsManager

manager = ResultsManager()

# Define test configurations
configs = [
    # Model, Dataset, PredLen, MSE, MAE
    ("TimesMamba", "ETTh1", 96, 0.375, 0.397),
    ("TimesMamba", "ETTh1", 192, 0.425, 0.424),
    ("TimesMamba", "ETTh2", 96, 0.445, 0.410),
    ("TimesMamba", "ETTm1", 96, 0.320, 0.380),
    
    ("iTransformer", "ETTh1", 96, 0.386, 0.405),
    ("iTransformer", "ETTh1", 192, 0.441, 0.436),
    ("iTransformer", "ETTh2", 96, 0.460, 0.425),
    ("iTransformer", "ETTm1", 96, 0.335, 0.395),
    
    ("PatchTST", "ETTh1", 96, 0.414, 0.419),
    ("PatchTST", "ETTh1", 192, 0.460, 0.445),
    ("PatchTST", "ETTh2", 96, 0.490, 0.450),
    ("PatchTST", "ETTm1", 96, 0.355, 0.410),
]

# Add all results
added = 0
for model, dataset, pred_len, mse, mae in configs:
    manager.add_result(
        model_name=model,
        dataset=dataset,
        pred_len=pred_len,
        mse=mse,
        mae=mae,
        notes=f"Tested {model} on {dataset}"
    )
    added += 1
    print(f"✅ Added: {model} on {dataset} (L{pred_len})")

print(f"\n🎉 {added} results added to comparison database!")

# Show summary
pivot = manager.get_pivot_table(metric='mse')
print("\nMSE Comparison:")
print(pivot)
```

## Method 5: Parse Existing Log Files

### Extract from Training Logs

```python
import re
from results_manager import ResultsManager

def parse_training_log(log_file):
    """Parse results from training log"""
    manager = ResultsManager()
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Example regex patterns (adjust to your log format)
    patterns = {
        'model': r'Testing\s+(\w+)',
        'dataset': r'on\s+(\w+)',
        'pred_len': r'pred_len=(\d+)',
        'mse': r'MSE:\s+([\d.]+)',
        'mae': r'MAE:\s+([\d.]+)',
    }
    
    # Extract values
    model = re.search(patterns['model'], content)
    dataset = re.search(patterns['dataset'], content)
    pred_len = re.search(patterns['pred_len'], content)
    mse = re.search(patterns['mse'], content)
    mae = re.search(patterns['mae'], content)
    
    # Add if all found
    if all([model, dataset, pred_len, mse, mae]):
        manager.add_result(
            model_name=model.group(1),
            dataset=dataset.group(1),
            pred_len=int(pred_len.group(1)),
            mse=float(mse.group(1)),
            mae=float(mae.group(1))
        )
        return True
    return False

# Usage
if parse_training_log('training.log'):
    print("✅ Results extracted from log")
else:
    print("❌ Could not parse log")
```

## Method 6: One-Liner Commands

### Quick Add via Terminal

```bash
# Interactive add
python add_results.py interactive

# Or with one command (using Python)
python -c "
from results_manager import ResultsManager
m = ResultsManager()
m.add_result('TimesMamba', 'ETTh1', 96, 0.375, 0.397)
print('Added!')
"
```

## Method 7: Combine with Streamlit UI

### Workflow for Comparison

1. **Run Tests** - Get MSE, MAE values
2. **Use Streamlit UI** - Open dashboard
3. **Click Comparison Tab** - Navigate there
4. **Expand Manage Results** - Find add form
5. **Fill & Click Add** - Results saved
6. **Refresh Page** - See in comparison tables

### Complete Workflow Script

```python
#!/usr/bin/env python3
"""
Complete workflow: Test models → Add results → View comparison
"""

import os
import sys
sys.path.append(os.getcwd())

from results_manager import ResultsManager
import subprocess

def main():
    print("🚀 TimesMamba Model Comparison Workflow\n")
    
    # Step 1: Run tests
    print("Step 1: Running tests...")
    # subprocess.run(["python", "test/test.py"])
    
    # Step 2: Collect results
    print("Step 2: Collecting results...")
    manager = ResultsManager()
    
    # Example: Add test results
    test_results = [
        ("TimesMamba", "ETTh1", 96, 0.375, 0.397),
        ("iTransformer", "ETTh1", 96, 0.386, 0.405),
        ("PatchTST", "ETTh1", 96, 0.414, 0.419),
    ]
    
    for model, dataset, pred_len, mse, mae in test_results:
        manager.add_result(model, dataset, pred_len, mse, mae)
        print(f"  ✅ Added {model} on {dataset}")
    
    # Step 3: View summary
    print("\nStep 3: Summary Statistics")
    summary = manager.get_summary_stats()
    print(f"  Total experiments: {summary['total_experiments']}")
    print(f"  Models: {summary['models']}")
    
    # Step 4: Show comparison
    print("\nStep 4: MSE Comparison")
    pivot = manager.get_pivot_table(metric='mse')
    print(pivot)
    
    # Step 5: View in Streamlit
    print("\n✅ Results saved! Now run:")
    print("   streamlit run streamlit_app.py")
    print("   Then go to 🏆 Comparison tab to view results!")

if __name__ == "__main__":
    main()
```

## Practical Examples

### Example 1: After Training TimesMamba

```python
# After training completes
val_metrics = {
    'mse': 0.375,
    'mae': 0.397
}

manager = ResultsManager()
manager.add_result(
    model_name="TimesMamba",
    dataset="ETTh1",
    pred_len=96,
    mse=val_metrics['mse'],
    mae=val_metrics['mae'],
    notes="Trained with lr=0.001, epochs=100"
)
```

### Example 2: Run Multiple Tests

```python
datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']
pred_lengths = [96, 192, 336]

manager = ResultsManager()

for dataset in datasets:
    for pred_len in pred_lengths:
        # Run test
        mse, mae = test_timesmamba(dataset, pred_len)
        
        # Save result
        manager.add_result(
            model_name="TimesMamba",
            dataset=dataset,
            pred_len=pred_len,
            mse=mse,
            mae=mae
        )
```

### Example 3: Track Model Improvements

```python
from results_manager import ResultsManager
import datetime

manager = ResultsManager()

# Version 1.0
manager.add_result("TimesMamba_v1.0", "ETTh1", 96, 0.400, 0.410, 
                   notes="Initial version")

# After optimization
manager.add_result("TimesMamba_v1.1", "ETTh1", 96, 0.375, 0.397,
                   notes="With data augmentation")

# After tuning
manager.add_result("TimesMamba_v1.2", "ETTh1", 96, 0.370, 0.390,
                   notes="Optimized hyperparameters")

# View improvement
df = manager.get_model_comparison(models=["TimesMamba_v1.0", "TimesMamba_v1.1", "TimesMamba_v1.2"])
print(df)  # See MSE decrease: 0.400 → 0.375 → 0.370 ✅
```

## Best Practices

✅ **Do:**
- Use consistent dataset names (ETTh1, not ETT_h1)
- Record notes about hyperparameters
- Add results immediately after testing
- Use pred_len values from paper (96, 192, 336, 720)

❌ **Don't:**
- Mix dataset names (ETTh1, ETT-h1, ETT_h1, ett_h1)
- Leave pred_len blank
- Add results without knowing MSE/MAE
- Forget to refresh Streamlit to see new results

## Next Steps

1. ✅ Choose a method above that fits your workflow
2. ✅ Integrate into your test/train scripts
3. ✅ Run tests with TimesMamba, iTransformer, PatchTST
4. ✅ View results in Streamlit comparison tab
5. ✅ Share comparison table with your team!

---

**Ready to add your results?** Pick a method and start! 🎯
