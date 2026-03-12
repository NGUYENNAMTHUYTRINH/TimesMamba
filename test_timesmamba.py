#!/usr/bin/env python3
"""
Simple test script for TimesMamba project
This script loads the mock mamba-ssm and tests the basic functionality
"""

# Load mock mamba-ssm first
exec(open('mamba_ssm_mock.py').read())

# Test imports
print("Testing imports...")
try:
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from model import TimesMamba
    from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from utils.tools import EarlyStopping
    from utils.metrics import metric
    print("✅ All imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Test basic model instantiation
print("\nTesting model instantiation...")
try:
    # Create a simple model config
    class Args:
        def __init__(self):
            self.seq_len = 96
            self.pred_len = 96
            self.enc_in = 7
            self.d_model = 128
            self.d_ff = 256
            self.dropout = 0.1
            self.channel_independence = False
            self.ssm_expand = 0  # Disable mamba for now
            self.r_ff = 1
            self.use_norm = True
            self.revin_affine = False
            self.use_mark = False  # Temporal features
            self.features = 'M'  # Multivariate
            self.e_layers = 2  # Number of encoder layers
    
    args = Args()
    model = TimesMamba.Model(args)
    print(f"✅ Model created successfully: {type(model).__name__}")
    
    # Test forward pass with dummy data
    batch_size = 4
    seq_len = args.seq_len
    pred_len = args.pred_len
    channels = args.enc_in
    
    x_enc = torch.randn(batch_size, seq_len, channels)
    x_mark_enc = torch.randn(batch_size, seq_len, 4) if args.use_mark else None
    x_dec = torch.randn(batch_size, pred_len, channels)
    x_mark_dec = torch.randn(batch_size, pred_len, 4) if args.use_mark else None
    
    print(f"Input shape: {x_enc.shape}")
    
    with torch.no_grad():
        if args.use_mark:
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            # Create dummy mark tensors when use_mark is False
            x_mark_enc_dummy = torch.zeros(batch_size, seq_len, 4)
            x_mark_dec_dummy = torch.zeros(batch_size, pred_len, 4)
            output = model(x_enc, x_mark_enc_dummy, x_dec, x_mark_dec_dummy)
    print(f"Output shape: {output.shape}")
    print("✅ Forward pass successful!")
    
except Exception as e:
    print(f"❌ Model test error: {e}")
    import traceback
    traceback.print_exc()

print("\n🎉 TimesMamba basic test completed!")
print("\nNext steps:")
print("1. Add real dataset to test training")
print("2. Implement proper mamba-ssm when CUDA driver is updated")
print("3. Run full experiments with 'python run.py'")