#!/usr/bin/env python3
"""
TimesMamba Training Script
Train TimesMamba model on ETT datasets
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Add parent directory to path

# Load mock mamba-ssm first
exec(open('../mamba_ssm_mock.py').read())

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
        self.root_path = f'datasets/{dataset}'
        self.data = dataset
        self.data_path = f'{dataset}_train.csv'  # Use train split
        
        # Model config
        self.seq_len = 96
        self.label_len = 48  
        self.pred_len = 24
        self.enc_in = 4  # OT, M1, M2, M3
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

def main():
    """Main training function"""
    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Training on {dataset}")
        print(f"{'='*60}")
        
        args = Args(dataset=dataset)
        model, losses = train_model(args)
        
        if model is not None:
            print(f"[SUCCESS] {dataset} training completed!")
            print(f"Final loss: {losses[-1]:.6f}")
            print(f"Best loss: {min(losses):.6f}")
        else:
            print(f"[ERROR] {dataset} training failed!")
    
    print(f"\n[SUMMARY] Training Summary:")
    print(f"All models trained and saved in train/ folder")

if __name__ == "__main__":
    main()