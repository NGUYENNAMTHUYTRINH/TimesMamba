#!/usr/bin/env python3
"""
Training script for ITransformer
Individual Series Transformer
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model_itransformer import ITransformer
from data_provider.data_factory import data_provider


class Args:
    def __init__(self, dataset='ETTh1'):
        # Dataset config
        script_dir = os.path.dirname(__file__)
        self.data = dataset
        if dataset.lower() == 'weather':
            self.root_path = os.path.abspath(os.path.join(script_dir, '..', 'dataset', 'weather'))
            self.data_path = 'weather.csv'
        else:
            self.root_path = os.path.abspath(os.path.join(script_dir, '..', 'train', 'datasets', dataset))
            self.data_path = f'{dataset}_train.csv'
        
        # Model config
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 24
        self.enc_in = 4
        self.d_model = 64
        self.n_heads = 4
        self.n_layers = 2
        self.d_ff = 256
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


def train_epoch(model, train_loader, criterion, optimizer, device, epoch=None, total_epochs=None):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total_batches = len(train_loader)

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        batch_x = batch_x.to(device).float()
        batch_y = batch_y.to(device).float()
        batch_x_mark = batch_x_mark.to(device).float()
        batch_y_mark = batch_y_mark.to(device).float()

        # Ensure batch shape matches model expectation: (batch, seq_len, enc_in)
        if batch_x.dim() == 3 and hasattr(model, 'enc_in'):
            if batch_x.shape[2] != model.enc_in and batch_x.shape[1] == model.enc_in:
                # common case: data is (batch, features, seq_len) -> transpose
                batch_x = batch_x.permute(0, 2, 1).contiguous()
                batch_x_mark = batch_x_mark.permute(0, 2, 1).contiguous() if batch_x_mark.dim() == 3 else batch_x_mark
                batch_y = batch_y.permute(0, 2, 1).contiguous() if batch_y.dim() == 3 else batch_y

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

        # Periodic batch progress logging
        if (i + 1) % 50 == 0 or (i + 1) == total_batches:
            if epoch is not None and total_epochs is not None:
                print(f"[Epoch {epoch+1}/{total_epochs}] Batch {i+1}/{total_batches} - batch_loss: {loss.item():.6f}", flush=True)
            else:
                print(f"Batch {i+1}/{total_batches} - batch_loss: {loss.item():.6f}", flush=True)

    return total_loss / total_batches if total_batches > 0 else 0.0


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
    
    dataset = os.getenv('DATASET', 'weather')
    args = Args(dataset)
    # allow quick override of epochs via env var for short runs
    try:
        args.train_epochs = int(os.getenv('TRAIN_EPOCHS', args.train_epochs))
    except Exception:
        pass
    
    # Create data loaders
    train_data, train_loader = data_provider(args, 'train')
    val_data, val_loader = data_provider(args, 'val')
    
    # Determine input feature dimension from dataset (important for custom datasets like 'weather')
    enc_in = getattr(train_data, 'data_x', None)
    if enc_in is not None:
        try:
            enc_in = train_data.data_x.shape[1]
        except Exception:
            enc_in = args.enc_in
    else:
        enc_in = args.enc_in

    # Model
    model = ITransformer(
        enc_in=enc_in,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout
    ).to(device)
    
    print(f"ITransformer Model created. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\nTraining ITransformer on {dataset}...")
    print("=" * 60)
    
    for epoch in range(args.train_epochs):
        print(f"Starting Epoch {epoch+1}/{args.train_epochs}", flush=True)
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch=epoch, total_epochs=args.train_epochs)
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
                print(f"  Early stopping (patience {args.patience})")
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save model
    os.makedirs('saved_models', exist_ok=True)
    model_path = f'saved_models/best_model_ITransformer_{dataset}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'ITransformer Training Curves - {dataset}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/training_loss_ITransformer_{dataset}.png', dpi=100, bbox_inches='tight')
    print(f"Plot saved: figures/training_loss_ITransformer_{dataset}.png")
    plt.close()
    
    print("\n" + "=" * 60)
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    main()
