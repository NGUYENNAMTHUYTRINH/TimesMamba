#!/usr/bin/env python3
"""
ITransformer Model - Individual Series Transformer
Applies transformer to each variable independently
Paper: https://arxiv.org/abs/2310.06625
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """x: (batch_size, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1), :]


class ITransformerLayer(nn.Module):
    """Single transformer block for individual channel"""
    
    def __init__(self, d_model=64, n_heads=4, d_ff=256, dropout=0.1):
        super(ITransformerLayer, self).__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, 
            n_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """x: (batch_size, seq_len, d_model)"""
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class ITransformer(nn.Module):
    """
    ITransformer: Individual Series Transformer
    
    Args:
        enc_in: number of features (variables)
        seq_len: input sequence length
        pred_len: prediction length
        d_model: embedding dimension
        n_heads: number of attention heads
        n_layers: number of transformer layers
        d_ff: feed-forward dimension
        dropout: dropout rate
    """
    
    def __init__(self,
                 enc_in=4,
                 seq_len=96,
                 pred_len=24,
                 d_model=64,
                 n_heads=4,
                 n_layers=2,
                 d_ff=256,
                 dropout=0.1):
        super(ITransformer, self).__init__()
        
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        
        # Embedding: project each feature to d_model
        self.embedding = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, seq_len)
        
        # Transformer layers (applied to each channel independently)
        self.transformer_layers = nn.ModuleList([
            ITransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection: d_model -> pred_len
        self.proj = nn.Linear(d_model, pred_len)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, enc_in)
        Returns:
            y: (batch_size, pred_len, enc_in)
        """
        batch_size = x.shape[0]
        
        # Process each channel independently
        outputs = []
        
        for i in range(self.enc_in):
            # Extract channel i: (batch_size, seq_len)
            x_channel = x[:, :, i:i+1]  # (batch_size, seq_len, 1)
            
            # Embedding: (batch_size, seq_len, d_model)
            x_embed = self.embedding(x_channel)
            
            # Add positional encoding
            x_embed = self.pos_encoding(x_embed)
            
            # Apply transformer layers
            for layer in self.transformer_layers:
                x_embed = layer(x_embed)
            
            # Project to prediction length: (batch_size, seq_len, d_model) -> (batch_size, pred_len)
            # Take mean across seq_len dimension
            x_mean = torch.mean(x_embed, dim=1)  # (batch_size, d_model)
            out = self.proj(x_mean)  # (batch_size, pred_len)
            
            outputs.append(out)
        
        # Stack outputs: (enc_in, batch_size, pred_len) -> (batch_size, pred_len, enc_in)
        y = torch.stack(outputs, dim=2)
        
        return y
