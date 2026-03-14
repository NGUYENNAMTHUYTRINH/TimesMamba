#!/usr/bin/env python3
"""
TimesMamba Testing Script
Test trained TimesMamba models on ETT datasets
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Add parent directory to path

# Load mock mamba-ssm first (path-resolved relative to this file)
mock_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mamba_ssm_mock.py')
try:
    exec(open(mock_path).read())
    print(f"Mamba SSM mock module loaded from {mock_path}")
except Exception as e:
    # Fallback: try importing if project root on sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        import mamba_ssm_mock  # type: ignore
        sys.modules['mamba_ssm'] = mamba_ssm_mock  # type: ignore
        print("Mamba SSM mock module imported via sys.path fallback")
    except Exception as e2:
        print(f"[WARN] Could not load mamba_ssm_mock: {e2}")

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from model import TimesMamba
from data_provider.data_factory import data_provider

class Args:
    def __init__(self, dataset='ETTh1'):
        # Dataset config - resolve to test/datasets/<dataset> relative to repo
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.root_path = os.path.join(project_root, 'test', 'datasets', dataset)
        self.data = dataset
        self.data_path = f'{dataset}_test.csv'  # Use test split
        
        # Model config (same as training for compatibility)
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 24
        self.enc_in = 4  # OT, M1, M2, M3
        self.d_model = 64
        self.d_ff = 128
        self.dropout = 0.1
        self.e_layers = 2
        
        # Test config
        self.features = 'M'  # Multivariate
        self.target = 'OT'
        self.batch_size = 1  # Single batch for visualization
        
        # Other configs
        self.channel_independence = False
        self.ssm_expand = 0
        self.r_ff = 1
        self.use_norm = True
        self.revin_affine = False
        self.use_mark = False
        self.num_workers = 0
        self.embed = 'timeF'
        self.freq = 'h'

def test_model(args, model_path=None):
    """Test TimesMamba model"""
    print(f"[TESTING] TimesMamba on {args.data}")
    print("=" * 50)
    
    # Create model
    model = TimesMamba.Model(args)
    
    # Load trained weights if available
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"[SUCCESS] Loaded trained model from {model_path}")
    else:
        print("⚠️  Using randomly initialized model (no trained weights)")
    
    model.eval()
    
    # Load test data
    try:
        test_data, test_loader = data_provider(args, flag='test')
        print(f"[SUCCESS] Test data loaded: {len(test_data)} samples")
    except Exception as e:
        print(f"[ERROR] Error loading test data: {e}")
        print(f"Expected file: {args.root_path}/{args.data_path}")
        return None
    
    # Testing
    predictions = []
    targets = []
    
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            # Convert tensors to float32 to fix dtype mismatch
            batch_x = batch_x.float()
            batch_y = batch_y.float()  
            batch_x_mark = batch_x_mark.float()
            batch_y_mark = batch_y_mark.float()
            
            # Forward pass
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :])
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # Store results
            f_dim = -1 if args.features == 'MS' else 0
            target = batch_y[:, -args.pred_len:, f_dim:]
            
            predictions.append(outputs.cpu().numpy())
            targets.append(target.cpu().numpy())
            
            if i >= 10:  # Limit to first 10 batches for demo
                break
    
    # Convert to arrays
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Calculate metrics
    mse = mean_squared_error(targets.flatten(), predictions.flatten())
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    
    print(f"[RESULTS] Test Results for {args.data}:")
    print(f"   MSE: {mse:.6f}")
    print(f"   MAE: {mae:.6f}")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Plot first 3 samples, first variable (OT)
    for i in range(min(3, len(predictions))):
        plt.subplot(3, 1, i+1)
        pred = predictions[i, :, 0]  # First variable
        true = targets[i, :, 0]
        
        plt.plot(true, 'b-', label='Ground Truth', linewidth=2)
        plt.plot(pred, 'r--', label='Prediction', linewidth=2)
        plt.title(f'{args.data} - Sample {i+1} - Target Variable (OT)')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'test_results_{args.data}.png', dpi=300, bbox_inches='tight')
    print(f"[PLOT] Test visualization saved as test_results_{args.data}.png")
    
    return {'mse': mse, 'mae': mae, 'predictions': predictions, 'targets': targets}

def compare_models(results_dict):
    """Compare results across different datasets"""
    print(f"\n[COMPARISON] Model Comparison:")
    print("=" * 60)
    print(f"{'Dataset':<10} {'MSE':<12} {'MAE':<12} {'Status'}")
    print("-" * 60)
    
    for dataset, results in results_dict.items():
        if results:
            mse = results['mse']
            mae = results['mae']
            status = "[SUCCESS] Success"
        else:
            mse = float('inf')
            mae = float('inf')
            status = "[FAILED] Failed"
        
        print(f"{dataset:<10} {mse:<12.6f} {mae:<12.6f} {status}")
    
    # Find best model
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    if valid_results:
        best_dataset = min(valid_results.keys(), key=lambda x: valid_results[x]['mse'])
        print(f"\n[BEST] Best Model: {best_dataset} (MSE: {valid_results[best_dataset]['mse']:.6f})")

def main():
    """Main testing function"""
    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Testing {dataset}")
        print(f"{'='*60}")
        
        args = Args(dataset=dataset)
        model_path = f'../train/best_model_{dataset}.pth'  # Look for trained model
        
        result = test_model(args, model_path)
        results[dataset] = result
    
    # Compare all models
    compare_models(results)
    
    print(f"\n[SUMMARY] Testing Summary:")
    print(f"All test results saved in test/ folder")

if __name__ == "__main__":
    main()