#!/usr/bin/env python3
"""
Export train/test datasets to structured folders
"""
import os
import pandas as pd
import shutil

def export_datasets():
    """Export train/test splits to train/datasets and test/datasets folders"""
    source_dir = "dataset/ETT-small/splits"
    
    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']
    
    for dataset in datasets:
        print(f"Processing {dataset}...")
        
        # Create dataset folders
        train_dataset_dir = f"train/datasets/{dataset}"
        test_dataset_dir = f"test/datasets/{dataset}"
        os.makedirs(train_dataset_dir, exist_ok=True)
        os.makedirs(test_dataset_dir, exist_ok=True)
        
        # Copy train file
        train_src = f"{source_dir}/{dataset}_train.csv"
        train_dst = f"{train_dataset_dir}/{dataset}_train.csv"
        if os.path.exists(train_src):
            shutil.copy2(train_src, train_dst)
            print(f"  ✅ Copied {train_src} → {train_dst}")
        else:
            print(f"  ❌ Train file not found: {train_src}")
        
        # Copy test file  
        test_src = f"{source_dir}/{dataset}_test.csv"
        test_dst = f"{test_dataset_dir}/{dataset}_test.csv"
        if os.path.exists(test_src):
            shutil.copy2(test_src, test_dst)
            print(f"  ✅ Copied {test_src} → {test_dst}")
        else:
            print(f"  ❌ Test file not found: {test_src}")

def export_remaining_splits():
    """Export remaining dataset splits"""
    source_dir = "dataset/ETT-small"
    datasets = ['ETTh2', 'ETTm1', 'ETTm2']  # ETTh1 already done
    
    for dataset in datasets:
        print(f"Creating splits for {dataset}...")
        
        # Create splits
        cmd = f"python scripts/export_splits.py --file {source_dir}/{dataset}.csv --out_dir {source_dir}/splits --seq_len 96"
        os.system(cmd)

if __name__ == "__main__":
    print("📁 Creating dataset structure...")
    
    # First create remaining splits
    export_remaining_splits()
    
    # Then export to structured folders
    export_datasets()
    
    print("✅ Dataset structure created!")