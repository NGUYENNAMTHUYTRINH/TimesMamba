#!/usr/bin/env python3
"""
Training script for RNN model
Uses same dataset split as TimesMamba for fair comparison
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

from model_rnn import RNNModel
from data_provider.data_factory import data_provider


class Args:
    def __init__(self, dataset='ETTh1'):
        # Dataset config
        script_dir = os.path.dirname(__file__)
        self.root_path = os.path.abspath(os.path.join(script_dir, '..', 'train', 'datasets', dataset))
        self.data = dataset
        self.data_path = f'{dataset}_train.csv'
        
        # Model config
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 24
        self.enc_in = 4
        self.d_model = 64
        self.n_layers = 2
        self.dropout = 0.1
        
        # Training config
        self.features = 'M'
        self.target = 'OT'
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.train_epochs = 10
        self.patience = 3
        
        # Other
        self.channel_independence = False
        self.freq = 'h'
        self.embed = 'timeF'
        self.num_workers = 0


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
        batch_x = batch_x.to(device).float()
        batch_y = batch_y.to(device).float()
        batch_x_mark = batch_x_mark.to(device).float()
        batch_y_mark = batch_y_mark.to(device).float()
        
        # Forward
        pred = model(batch_x)
        # batch_y contains [label_len + pred_len] timesteps; take only prediction tail
        # Take last pred_len timesteps across all features
        targets = batch_y[:, -model.pred_len:, :]
        loss = criterion(pred, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            batch_x_mark = batch_x_mark.to(device).float()
            batch_y_mark = batch_y_mark.to(device).float()

            pred = model(batch_x)
            targets = batch_y[:, -model.pred_len:, :]
            loss = criterion(pred, targets)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = 'ETTh1'
    args = Args(dataset)
    
    # Create data loaders
    train_data, train_loader = data_provider(args, 'train')
    val_data, val_loader = data_provider(args, 'val')
    
    # Model
    model = RNNModel(
        enc_in=args.enc_in,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)
    
    print(f"RNN Model created. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\nTraining RNN on {dataset}...")
    print("=" * 60)
    
    for epoch in range(args.train_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1:02d}/{args.train_epochs} | "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print("  Best model saved")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  ✗ Early stopping (patience {args.patience})")
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save model
    os.makedirs('saved_models', exist_ok=True)
    model_path = f'saved_models/best_model_RNN_{dataset}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'RNN Training Curves - {dataset}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/training_loss_RNN_{dataset}.png', dpi=100, bbox_inches='tight')
    print(f"Plot saved: figures/training_loss_RNN_{dataset}.png")
    plt.close()
    
    print("\n" + "=" * 60)
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    main()
