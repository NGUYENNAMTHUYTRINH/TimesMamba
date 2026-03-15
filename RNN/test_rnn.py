#!/usr/bin/env python3
"""
Testing script for RNN model
Calculate MSE and MAE metrics
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from model_rnn import RNNModel
from data_provider.data_factory import data_provider


class Args:
    def __init__(self, dataset='ETTh1'):
        script_dir = os.path.dirname(__file__)
        self.data = dataset
        if dataset.lower() == 'weather':
            self.root_path = os.path.abspath(os.path.join(script_dir, '..', 'dataset', 'weather'))
            self.data_path = 'weather.csv'
        else:
            self.root_path = f'../train/datasets/{dataset}'
            self.data_path = f'{dataset}_train.csv'
        
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 24
        self.enc_in = 4
        self.d_model = 64
        self.n_layers = 2
        self.dropout = 0.1
        
        self.features = 'M'
        self.target = 'OT'
        self.batch_size = 1
        
        self.channel_independence = False
        self.freq = 'h'
        self.embed = 'timeF'
        self.num_workers = 0


def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = 'weather'
    args = Args(dataset)
    
    # Load test data
    test_data, test_loader = data_provider(args, 'test')
    
    # Load model
    model = RNNModel(
        enc_in=args.enc_in,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)
    
    model_path = f'saved_models/best_model_RNN_{dataset}.pth'
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found: {model_path}")
        print("Please run train_rnn.py first")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded: {model_path}")
    
    # Testing
    all_preds = []
    all_true = []
    
    print(f"\nTesting RNN on {dataset}...")
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            if i >= 10:  # Limit to 10 batches for demo
                break
            
            batch_x = batch_x.float().to(device)
            pred = model(batch_x)

            # Ensure batch_y is a tensor and slice to prediction length
            if not torch.is_tensor(batch_y):
                batch_y = torch.tensor(batch_y)
            true = batch_y.float()[:, -args.pred_len:, :].cpu().numpy()

            all_preds.append(pred.cpu().numpy())
            all_true.append(true)
    
    # Concatenate all predictions and true values
    all_preds = np.concatenate(all_preds, axis=0)  # (n_samples, pred_len, enc_in)
    all_true = np.concatenate(all_true, axis=0)

    # Reshape to 2D for sklearn (samples*timesteps, features)
    try:
        y_pred = all_preds.reshape(-1, all_preds.shape[-1])
        y_true = all_true.reshape(-1, all_true.shape[-1])
    except Exception:
        y_pred = all_preds.ravel()
        y_true = all_true.ravel()

    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Save results
    results = {
        'model': 'RNN',
        'dataset': dataset,
        'pred_len': args.pred_len,
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'n_samples': len(all_preds)
    }
    
    print("\n" + "=" * 60)
    print("TEST RESULTS - RNN")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Prediction Length: {args.pred_len}")
    print(f"Number of Test Samples: {len(all_preds)}")
    print(f"MSE:  {mse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print("=" * 60)
    
    # Visualization: plot first 3 samples
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx in range(min(3, len(all_preds))):
        ax = axes[idx]
        
        # Take first variable (OT - electricity)
        true_values = all_true[idx, :, 0]
        pred_values = all_preds[idx, :, 0]
        
        x = np.arange(len(true_values))
        ax.plot(x, true_values, 'o-', label='Actual', linewidth=2, markersize=6)
        ax.plot(x, pred_values, 's-', label='Predicted', linewidth=2, markersize=6)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('OT (Electricity Load)')
        ax.set_title(f'Sample {idx+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/test_results_RNN_{dataset}.png', dpi=100, bbox_inches='tight')
    print(f"Test plot saved: figures/test_results_RNN_{dataset}.png")
    plt.close()
    
    # Save predictions
    os.makedirs('predictions', exist_ok=True)
    pred_df = pd.DataFrame({
        'mse': [mse],
        'mae': [mae],
        'rmse': [rmse],
        'model': ['RNN'],
        'dataset': [dataset],
        'pred_len': [args.pred_len]
    })
    pred_df.to_csv(f'predictions/results_RNN_{dataset}.csv', index=False)
    print(f"Results saved: predictions/results_RNN_{dataset}.csv")
    
    return results


if __name__ == '__main__':
    results = test_model()
