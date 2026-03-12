#!/usr/bin/env python3
"""
Text experiment with TimesMamba (Educational Purpose Only)
This demonstrates how to adapt TimesMamba for a simple text task,
but NOTE: TimesMamba is designed for time series, not text processing!
"""

# Load mock mamba-ssm first
exec(open('mamba_ssm_mock.py').read())

import torch
import numpy as np
from model import TimesMamba

def text_to_sequence(text, max_len=96):
    """
    Simple text to sequence conversion (character-based)
    Each character becomes a number (ASCII value normalized)
    """
    # Convert to lowercase and get ASCII values
    chars = [ord(c.lower()) for c in text if c.isalnum() or c.isspace()]
    
    # Normalize to 0-1 range (rough normalization)
    normalized = [(c - 32) / (126 - 32) for c in chars]
    
    # Pad or truncate to max_len
    if len(normalized) < max_len:
        normalized.extend([0.0] * (max_len - len(normalized)))
    else:
        normalized = normalized[:max_len]
    
    return np.array(normalized)

def extract_name_from_text(text):
    """
    Simple name extraction - find word after "NAME IS"
    """
    text_upper = text.upper()
    if "NAME IS" in text_upper:
        parts = text_upper.split("NAME IS")
        if len(parts) > 1:
            name_part = parts[1].strip()
            # Get first word after "NAME IS"
            name = name_part.split()[0] if name_part.split() else ""
            return name.lower()
    return ""

print("🔤 Text Processing Experiment with TimesMamba")
print("=" * 50)

# Test input
input_text = "HELLO! MY NAME IS MARTEN"
expected_output = "marten"

print(f"Input text: '{input_text}'")
print(f"Expected output: '{expected_output}'")

# Extract actual name for comparison
actual_name = extract_name_from_text(input_text)
print(f"Extracted name: '{actual_name}'")

print("\n📊 Converting text to sequence...")

# Convert text to sequence
sequence = text_to_sequence(input_text, max_len=96)
print(f"Text sequence shape: {sequence.shape}")
print(f"First 10 values: {sequence[:10]}")

print("\n🤖 Testing with TimesMamba...")

try:
    # Create model config for text experiment
    class Args:
        def __init__(self):
            self.seq_len = 96  # Input sequence length
            self.pred_len = 6   # Output length (max name length)
            self.enc_in = 1     # Single channel (character values)
            self.d_model = 64   # Smaller model for simple task
            self.d_ff = 128
            self.dropout = 0.1
            self.channel_independence = True  # Treat as single channel
            self.ssm_expand = 0  # Disable mamba (using mock anyway)
            self.r_ff = 1
            self.use_norm = True
            self.revin_affine = False
            self.use_mark = False
            self.features = 'M'
            self.e_layers = 2
    
    args = Args()
    model = TimesMamba.Model(args)
    print(f"✅ Model created: {type(model).__name__}")
    
    # Prepare input data
    batch_size = 1
    
    # Reshape sequence for model input [batch, seq_len, channels]
    x_enc = torch.FloatTensor(sequence).reshape(1, args.seq_len, 1)
    
    # Create dummy decoder input and marks
    x_dec = torch.zeros(batch_size, args.pred_len, args.enc_in)
    x_mark_enc = torch.zeros(batch_size, args.seq_len, 4)
    x_mark_dec = torch.zeros(batch_size, args.pred_len, 4)
    
    print(f"Model input shape: {x_enc.shape}")
    
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    print(f"Model output shape: {output.shape}")
    print(f"Output values: {output.flatten()[:6].numpy()}")
    
    print("\n✅ Model forward pass successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("🚨 IMPORTANT NOTES:")
print("=" * 50)

print("""
1. ❌ TimesMamba is NOT designed for text processing!
   - It's built for multivariate time series forecasting
   - Architecture expects numerical time series data
   - No text understanding capabilities

2. 🔍 For text tasks like name extraction, you should use:
   - Transformer models (BERT, GPT, etc.)
   - NLP libraries (spaCy, NLTK)
   - Simple regex/string processing

3. 📚 What this demo shows:
   - How to adapt model inputs technically
   - TimesMamba can process sequences numerically
   - BUT outputs are meaningless for text understanding

4. 🎯 To actually solve "extract name from text":
   - Use rule-based approach (regex)
   - Use pre-trained NLP models
   - Fine-tune text classification models
""")

print(f"\n🎯 Simple solution for your task:")
print(f"Input: '{input_text}'")
print(f"Output: '{extract_name_from_text(input_text)}'")
print("(This uses simple string processing, which is appropriate for this task)")

print(f"\n💡 TimesMamba is best used for:")
print("- Stock price prediction")
print("- Weather forecasting") 
print("- Sensor data analysis")
print("- Any multivariate time series data")