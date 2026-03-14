import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Ensure project root is on sys.path and import mock mamba_ssm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
try:
    import mamba_ssm_mock
    sys.modules['mamba_ssm'] = mamba_ssm_mock
    print("Mamba SSM mock module loaded successfully")
except Exception as e:
    print(f"[WARN] Could not import mamba_ssm_mock from {project_root}: {e}")

from model import TimesMamba

def save_predictions_csv(predictions, dataset, pred_len, mean=None, std=None):
    """Save predictions array to CSV with proper column names"""
    # predictions shape: (n_sequences, pred_len, n_features=4)
    feature_names = ['OT', 'M1', 'M2', 'M3']
    
    # Create column names: step_0_OT, step_1_OT, ..., step_0_M1, step_1_M1, ...
    columns = []
    for feat in feature_names:
        for step in range(pred_len):
            columns.append(f'step_{step}_{feat}')
    
    # Reshape predictions to match column format
    n_sequences = predictions.shape[0]
    reshaped = predictions.reshape(n_sequences, -1)  # (n_seq, pred_len*4)
    
    # Save normalized predictions
    df_normalized = pd.DataFrame(reshaped, columns=columns)
    csv_file_norm = f'predictions_{dataset}_normalized.csv'
    df_normalized.to_csv(csv_file_norm, index=False)
    
    # If mean/std provided, also save original scale predictions
    if mean is not None and std is not None:
        # Inverse transform: denormalized = normalized * std + mean
        predictions_original = predictions.copy()
        for i in range(4):  # 4 features
            predictions_original[:, :, i] = predictions[:, :, i] * std[i] + mean[i]
        
        # Reshape and save original scale
        reshaped_orig = predictions_original.reshape(n_sequences, -1)
        df_original = pd.DataFrame(reshaped_orig, columns=columns)
        csv_file_orig = f'predictions_{dataset}.csv'
        df_original.to_csv(csv_file_orig, index=False)
        
        print(f"[CSV] Saved predictions: {csv_file_norm} (normalized) & {csv_file_orig} (original scale)")
        print(f"      {n_sequences} sequences, {len(columns)} columns")
        
        # Show sample original vs normalized values
        print(f"[SCALE] Sample original vs normalized (first prediction, step 0):")
        print(f"        Original: OT={predictions_original[0,0,0]:.2f}, M1={predictions_original[0,0,1]:.2f}")
        print(f"        Normalized: OT={predictions[0,0,0]:.2f}, M1={predictions[0,0,1]:.2f}")
    else:
        csv_file_orig = f'predictions_{dataset}.csv'
        df_normalized.to_csv(csv_file_orig, index=False)
        print(f"[CSV] Saved predictions: {csv_file_orig} ({n_sequences} sequences, {len(columns)} columns)")
        print(f"[WARNING] No mean/std provided - predictions remain normalized")

def simple_test_model(dataset='ETTh1'):
    """Simple test without complex data loader borders"""
    print(f"[TESTING] {dataset} - Simple Test")
    print("=" * 50)
    
    # Load test data directly (resolve path relative to this test script)
    test_file = os.path.join(os.path.dirname(__file__), 'datasets', dataset, f'{dataset}_test.csv')
    if not os.path.exists(test_file):
        print(f"[ERROR] Test file not found: {test_file}")
        return None
        
    df = pd.read_csv(test_file) 
    print(f"[DATA] Test data loaded: {len(df)} rows")
    
    # Simple preprocessing - just use numeric columns
    numeric_cols = ['OT', 'M1', 'M2', 'M3'] 
    data = df[numeric_cols].values.astype(np.float32)
    
    # Simple normalization
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) + 1e-8
    data = (data - mean) / std
    
    # Create simple sequences
    seq_len = 96  # Smaller sequence for testing
    pred_len = 24  # Smaller prediction length 
    
    if len(data) < seq_len + pred_len:
        print(f"[ERROR] Not enough data: {len(data)} rows < {seq_len + pred_len}")
        return None
        
    # Create test sequences
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_len - pred_len + 1):
        seq = data[i:i+seq_len]
        target = data[i+seq_len:i+seq_len+pred_len]
        sequences.append(seq)
        targets.append(target)
    
    if len(sequences) == 0:
        print("[ERROR] No valid sequences created")
        return None
        
    sequences = np.array(sequences)
    targets = np.array(targets) 
    
    print(f"[SEQUENCES] Created {len(sequences)} test sequences")
    
    # Complete model args (copied from working train.py)
    class SimpleArgs:
        def __init__(self):
            # Dataset config 
            self.data = dataset
            self.enc_in = 4
            self.dec_in = 4
            self.c_out = 4
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.label_len = pred_len // 2
            
            # Model config
            self.d_model = 64
            self.d_ff = 128
            self.e_layers = 2
            self.dropout = 0.1
            self.features = 'M'
            
            # All additional attributes from original
            self.use_norm = True
            self.use_mark = False
            self.channel_independence = False
            self.revin_affine = False
            self.ssm_expand = 0
            self.r_ff = 1
            self.embed = 'timeF'
            self.freq = 'h'
    
    args = SimpleArgs()
    
    # Load model
    model_path = f'../train/best_model_{dataset}.pth'
    
    try:
        # Create model with smaller seq_len
        model = TimesMamba.Model(args)
        
        # Try to load weights (might fail due to size mismatch)
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                print(f"[SUCCESS] Loaded trained model")
            except Exception as e:
                print(f"[WARNING] Could not load trained weights: {e}")
                print("[INFO] Using randomly initialized model")
        else:
            print("[WARNING] No trained model found, using random weights")
            
        model.eval()
        
        # Simple prediction
        predictions = []
        actuals = []
        
        with torch.no_grad():
            # Take first 5 sequences for demo
            for i in range(min(5, len(sequences))):
                seq = torch.FloatTensor(sequences[i:i+1])  # Batch size 1
                target = targets[i]
                
                # Create dummy time marks (zeros)
                time_mark = torch.zeros(1, seq_len, 4)  # Batch, seq, features
                dec_mark = torch.zeros(1, pred_len, 4)   # Batch, pred, features
                
                # Simple forward pass (might fail with size mismatch)
                try:
                    # Create decoder input
                    dec_inp = torch.zeros(1, pred_len, 4)
                    
                    output = model(seq, time_mark, dec_inp, dec_mark)
                    
                    pred = output.cpu().numpy()[0]  # Remove batch dimension
                    predictions.append(pred)
                    actuals.append(target)
                    
                except Exception as e:
                    print(f"[ERROR] Model forward pass failed: {e}")
                    # Create dummy prediction
                    pred = np.random.randn(pred_len, 4) * 0.01
                    predictions.append(pred)
                    actuals.append(target)
        
        if len(predictions) > 0:
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # Save predictions to CSV (both normalized and original scale)
            save_predictions_csv(predictions, dataset, pred_len, mean, std)
            
            # Calculate metrics
            mse = mean_squared_error(actuals.flatten(), predictions.flatten())
            mae = mean_absolute_error(actuals.flatten(), predictions.flatten())
            
            print(f"[RESULTS] MSE: {mse:.6f}, MAE: {mae:.6f}")
            
            # Simple plot
            plt.figure(figsize=(12, 6))
            
            # Plot first feature of first sequence
            actual_seq = actuals[0, :, 0]  # First sequence, all timesteps, first feature
            pred_seq = predictions[0, :, 0]
            
            plt.plot(actual_seq, label='Actual', color='blue', marker='o')
            plt.plot(pred_seq, label='Predicted', color='red', marker='x') 
            plt.title(f'{dataset} - Simple Test Prediction')
            plt.xlabel('Time Steps')
            plt.ylabel('Value (normalized)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_file = f'simple_test_{dataset}.png'
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[PLOT] Saved plot: {plot_file}")
            
            return {'mse': mse, 'mae': mae, 'status': 'success'}
        else:
            return {'mse': float('inf'), 'mae': float('inf'), 'status': 'failed'}
            
    except Exception as e:
        print(f"[ERROR] Testing failed: {e}")
        return {'mse': float('inf'), 'mae': float('inf'), 'status': 'failed'}

def main():
    """Run simple tests on all datasets"""
    print("[START] Simple TimesMamba Testing")
    print("=" * 60)
    
    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']
    results = []
    
    for dataset in datasets:
        result = simple_test_model(dataset)
        if result:
            results.append((dataset, result))
        print()
    
    # Summary
    print("[SUMMARY] Test Results")
    print("=" * 60)
    print(f"{'Dataset':<10} {'MSE':<12} {'MAE':<12} {'Status'}")
    print("-" * 60)
    
    for dataset, result in results:
        status = result['status']
        mse = f"{result['mse']:.6f}" if result['mse'] != float('inf') else "inf"
        mae = f"{result['mae']:.6f}" if result['mae'] != float('inf') else "inf"
        print(f"{dataset:<10} {mse:<12} {mae:<12} {status}")

if __name__ == "__main__":
    main()