#!/usr/bin/env python3
"""
Generate a sample ETTh1.csv compatible with Dataset_ETT_hour
Writes to dataset/ETT-small/ETTh1.csv
"""
import os
import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.getcwd()))
out_dir = os.path.join(ROOT, "dataset", "ETT-small")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "ETTh1.csv")

# Parameters
n_hours = 600  # enough rows for training/validation/test
start = datetime(2016, 1, 1, 0, 0, 0)

dates = [start + timedelta(hours=i) for i in range(n_hours)]

# Generate signals
# OT: target (hourly electricity temperature-like signal)
base = 10 + 5 * np.sin(np.arange(n_hours) * 2 * math.pi / 24)  # daily cycle
trend = np.linspace(0, 1.0, n_hours)
noise = np.random.normal(0, 0.5, n_hours)
OT = base + trend + noise

# Additional multivariate columns
M1 = 20 + 3 * np.sin(np.arange(n_hours) * 2 * math.pi / 24 + 0.5) + np.random.normal(0,0.8,n_hours)
M2 = 30 + 4 * np.sin(np.arange(n_hours) * 2 * math.pi / 24 + 1.0) + np.random.normal(0,0.8,n_hours)
M3 = 5 + 2 * np.cos(np.arange(n_hours) * 2 * math.pi / 24) + np.random.normal(0,0.3,n_hours)

# Build DataFrame
df = pd.DataFrame({
    'date': [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates],
    'OT': OT,
    'M1': M1,
    'M2': M2,
    'M3': M3,
})

# Save CSV
df.to_csv(out_file, index=False)
print(f"Wrote sample ETTh1 to: {out_file}")
print(df.head().to_string(index=False))
