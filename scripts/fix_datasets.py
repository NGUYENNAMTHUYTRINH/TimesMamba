import os
import pandas as pd
from pathlib import Path

def split_and_export_datasets():
    """Fixed dataset splitting that properly creates train/val/test splits"""
    print("[SETUP] Fixing dataset splits...")
    
    # Source and target directories
    source_dir = "dataset/ETT-small"
    
    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']
    
    for dataset in datasets:
        print(f"[PROCESSING] {dataset}")
        
        # Read original dataset
        source_path = f"{source_dir}/{dataset}.csv"
        if not os.path.exists(source_path):
            print(f"[ERROR] {source_path} not found")
            continue
            
        df = pd.read_csv(source_path)
        print(f"[INFO] Original data: {len(df)} rows")
        
        # Calculate split indices - ensure test has at least 150 rows
        total_len = len(df)
        min_test_size = max(150, int(0.15 * total_len))  # At least 150 or 15%
        min_val_size = max(100, int(0.1 * total_len))    # At least 100 or 10%
        
        test_start = total_len - min_test_size
        val_start = test_start - min_val_size 
        train_end = val_start
        
        # Create splits
        train_data = df[:train_end]
        val_data = df[val_start:test_start]
        test_data = df[test_start:]
        
        print(f"[SPLIT] Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Create directories and save files
        train_dir = f"train/datasets/{dataset}"
        test_dir = f"test/datasets/{dataset}"
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Save train and validation data
        train_data.to_csv(f"{train_dir}/{dataset}_train.csv", index=False)
        val_data.to_csv(f"{train_dir}/{dataset}_val.csv", index=False)
        
        # Save test data  
        test_data.to_csv(f"{test_dir}/{dataset}_test.csv", index=False)
        
        print(f"[SUCCESS] {dataset} splits created")
        
    print("[COMPLETE] All datasets fixed!")

if __name__ == "__main__":
    split_and_export_datasets()