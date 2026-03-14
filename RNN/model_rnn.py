#!/usr/bin/env python3
"""
RNN (LSTM-based) Model for Time Series Forecasting
Simple yet effective baseline model
"""
import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """
    LSTM-based model for time series forecasting
    
    Args:
        enc_in: input size (number of features)
        seq_len: input sequence length
        pred_len: prediction length
        d_model: hidden dimension
        n_layers: number of LSTM layers
        dropout: dropout rate
    """
    
    def __init__(self, 
                 enc_in=4, 
                 seq_len=96, 
                 pred_len=24,
                 d_model=64,
                 n_layers=2,
                 dropout=0.1):
        super(RNNModel, self).__init__()
        
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Input projection
        self.embedding = nn.Linear(enc_in, d_model)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, enc_in * pred_len)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, enc_in)
        Returns:
            y: (batch_size, pred_len, enc_in)
        """
        # Embedding: (batch_size, seq_len, d_model)
        x_embed = self.embedding(x)
        
        # LSTM: (batch_size, seq_len, d_model)
        lstm_out, (h_n, c_n) = self.lstm(x_embed)
        
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch_size, d_model)
        
        # Predict multiple timesteps
        # We'll use the last state repeatedly
        out = self.fc(last_hidden)  # (batch_size, enc_in * pred_len)
        
        # Reshape to (batch_size, pred_len, enc_in)
        batch_size = x.shape[0]
        out = out.view(batch_size, self.pred_len, self.enc_in)
        
        return out


class GRUModel(nn.Module):
    """
    GRU-based model for time series forecasting
    Alternative to LSTM with fewer parameters
    """
    
    def __init__(self,
                 enc_in=4,
                 seq_len=96,
                 pred_len=24,
                 d_model=64,
                 n_layers=2,
                 dropout=0.1):
        super(GRUModel, self).__init__()
        
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        
        # Input projection
        self.embedding = nn.Linear(enc_in, d_model)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, enc_in * pred_len)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, enc_in)
        Returns:
            y: (batch_size, pred_len, enc_in)
        """
        # Embedding
        x_embed = self.embedding(x)
        
        # GRU
        gru_out, h_n = self.gru(x_embed)
        
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch_size, d_model)
        
        # Predict
        out = self.fc(last_hidden)
        
        # Reshape
        batch_size = x.shape[0]
        out = out.view(batch_size, self.pred_len, self.enc_in)
        
        return out
