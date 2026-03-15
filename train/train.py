#!/usr/bin/env python3
"""
TimesMamba Training Script
Train TimesMamba model on ETT datasets
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Add parent directory to path

# Load mock mamba-ssm first (resolve path relative to this script)
mock_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mamba_ssm_mock.py'))
try:
    exec(open(mock_path).read())
    print(f"Mamba SSM mock module loaded from {mock_path}")
except Exception as e:
    # Fallback: ensure project root on sys.path and try import
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
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt

from model import TimesMamba
from data_provider.data_factory import data_provider

class Args:
    def __init__(self, dataset='ETTh1'):
        # Dataset config
        script_dir = os.path.dirname(__file__)
        self.data = dataset
        # Special-case the provided weather dataset located at project_root/dataset/weather/weather.csv
        if dataset.lower() == 'weather':
            self.root_path = os.path.abspath(os.path.join(script_dir, '..', 'dataset', 'weather'))
            self.data_path = 'weather.csv'
        else:
            # datasets live under train/datasets by default
            self.root_path = os.path.join(script_dir, 'datasets', dataset)
            self.data_path = f'{dataset}_train.csv'  # Use train split
        
        # Model config
        self.seq_len = 96
        self.label_len = 48  
        self.pred_len = 24
        self.enc_in = 21  # OT, M1, M2, M3
        self.d_model = 64
        self.d_ff = 128
        self.dropout = 0.1
        self.e_layers = 2
        
        # Training config
        self.features = 'M'  # Multivariate
        self.target = 'OT'
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.train_epochs = 10
        self.patience = 3
        
        # Other configs
        self.channel_independence = False
        self.ssm_expand = 0  # Use FFN only (mock)
        self.r_ff = 1
        self.use_norm = True
        self.revin_affine = False
        self.use_mark = False
        self.num_workers = 0
        self.embed = 'timeF'
        self.freq = 'h'

def train_model(args):
    """Train TimesMamba model"""
    print(f"[TRAINING] TimesMamba on {args.data}")
    print("=" * 50)
    
    # Create model
    model = TimesMamba.Model(args)
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()
    
    # Load training data
    try:
        train_data, train_loader = data_provider(args, flag='train')
        print(f"[SUCCESS] Training data loaded: {len(train_data)} samples")
    except Exception as e:
        print(f"[ERROR] Error loading training data: {e}")
        print(f"Expected file: {args.root_path}/{args.data_path}")
        return None
    
    # Training loop
    train_losses = []
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.train_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Convert tensors to float32 to fix dtype mismatch
            batch_x = batch_x.float()
            batch_y = batch_y.float()  
            batch_x_mark = batch_x_mark.float()
            batch_y_mark = batch_y_mark.float()
            
            # Forward pass
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :])
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # Loss calculation
            f_dim = -1 if args.features == 'MS' else 0
            targets = batch_y[:, -args.pred_len:, f_dim:]
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{args.train_epochs}, Batch {i}, Loss: {loss.item():.6f}")
        
        # Calculate average epoch loss
        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.6f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'best_model_{args.data}.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'TimesMamba Training - {args.data}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'training_loss_{args.data}.png', dpi=300, bbox_inches='tight')
    print(f"[PLOT] Training plot saved as training_loss_{args.data}.png")
    
    return model, train_losses

def main(datasets=None, epochs=None):
    """Main training function
    If `datasets` is provided (list), train on those datasets; otherwise defaults to ['weather'].
    """
    if datasets is None:
        datasets = ['weather']

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Training on {dataset}")
        print(f"{'='*60}")
        
        args = Args(dataset=dataset)
        if epochs is not None:
            args.train_epochs = int(epochs)
        model, losses = train_model(args)
        
        if model is not None:
            print(f"[SUCCESS] {dataset} training completed!")
            print(f"Final loss: {losses[-1]:.6f}")
            print(f"Best loss: {min(losses):.6f}")
            # Ensure project-level directories exist and move/save artifacts there
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            saved_dir = os.path.join(project_root, 'saved_models')
            figures_dir = os.path.join(project_root, 'figures')
            os.makedirs(saved_dir, exist_ok=True)
            os.makedirs(figures_dir, exist_ok=True)
            # Move model to saved_models
            model_dest = os.path.join(saved_dir, f'best_model_TimesMamba_{dataset}.pth')
            try:
                torch.save(model.state_dict(), model_dest)
                print(f"Model saved: {model_dest}")
            except Exception:
                # fallback: save to current folder as before
                torch.save(model.state_dict(), f'best_model_{dataset}.pth')
                print(f"Model saved to fallback path: best_model_{dataset}.pth")
            # Save plot to figures dir
            plot_dest = os.path.join(figures_dir, f'training_loss_TimesMamba_{dataset}.png')
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(losses, 'b-', label='Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('MSE Loss')
                plt.title(f'TimesMamba Training - {dataset}')
                plt.legend()
                plt.grid(True)
                plt.savefig(plot_dest, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Training plot saved: {plot_dest}")
            except Exception as e:
                print(f"Failed to save plot to figures: {e}")
        else:
            print(f"[ERROR] {dataset} training failed!")
    
    print(f"\n[SUMMARY] Training Summary:")
    print(f"All models trained and saved in train/ folder")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='Dataset to train (e.g., weather, ETTh1)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train (overrides default)')
    args = parser.parse_args()

    if args.dataset:
        main(datasets=[args.dataset], epochs=args.epochs)
    else:
        main(epochs=args.epochs)