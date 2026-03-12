#!/usr/bin/env python3
"""
Generate sample ETTh2, ETTm1, and ETTm2 CSVs in dataset/ETT-small
ETTh2: hourly like ETTh1 but different phase/seed
ETTm1: 15-minute frequency (minute field used by loader)
ETTm2: another hourly file
"""
import os
import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.getcwd()))
out_dir = os.path.join(ROOT, "dataset", "ETT-small")
os.makedirs(out_dir, exist_ok=True)

# ETTh2 (hourly)
n_hours = 600
start = datetime(2016, 7, 1, 0, 0, 0)
dates = [start + timedelta(hours=i) for i in range(n_hours)]
base = 12 + 6 * np.sin(np.arange(n_hours) * 2 * math.pi / 24 + 1.2)
trend = np.linspace(0, -0.5, n_hours)
noise = np.random.normal(0, 0.6, n_hours)
OT = base + trend + noise
M1 = 18 + 2.5 * np.sin(np.arange(n_hours) * 2 * math.pi / 24 + 0.2) + np.random.normal(0,0.6,n_hours)
M2 = 28 + 3.5 * np.sin(np.arange(n_hours) * 2 * math.pi / 24 + 0.8) + np.random.normal(0,0.7,n_hours)
M3 = 6 + 1.5 * np.cos(np.arange(n_hours) * 2 * math.pi / 24 + 0.5) + np.random.normal(0,0.2,n_hours)

df2 = pd.DataFrame({
    'date': [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates],
    'OT': OT,
    'M1': M1,
    'M2': M2,
    'M3': M3,
})
file2 = os.path.join(out_dir, 'ETTh2.csv')
df2.to_csv(file2, index=False)
print('Wrote', file2)

# ETTm1 (15-minute frequency)
n_minutes = 600 * 4  # equivalent hours * 4
start_m = datetime(2016, 1, 1, 0, 0, 0)
dates_m = [start_m + timedelta(minutes=15*i) for i in range(n_minutes)]
base_m = 5 + 2 * np.sin(np.arange(n_minutes) * 2 * math.pi / (24*4))
trend_m = np.linspace(0, 0.2, n_minutes)
noise_m = np.random.normal(0, 0.2, n_minutes)
OTm = base_m + trend_m + noise_m
M1m = 10 + 1.2 * np.sin(np.arange(n_minutes) * 2 * math.pi / (24*4) + 0.3) + np.random.normal(0,0.25,n_minutes)
M2m = 15 + 1.8 * np.sin(np.arange(n_minutes) * 2 * math.pi / (24*4) + 0.7) + np.random.normal(0,0.3,n_minutes)
M3m = 2 + 0.8 * np.cos(np.arange(n_minutes) * 2 * math.pi / (24*4)) + np.random.normal(0,0.05,n_minutes)

dfm = pd.DataFrame({
    'date': [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates_m],
    'OT': OTm,
    'M1': M1m,
    'M2': M2m,
    'M3': M3m,
})
filem = os.path.join(out_dir, 'ETTm1.csv')
dfm.to_csv(filem, index=False)
print('Wrote', filem)

# ETTm2 (hourly but different seed)
n_hours2 = 700
start2 = datetime(2017, 1, 1, 0, 0, 0)
dates2 = [start2 + timedelta(hours=i) for i in range(n_hours2)]
base2 = 9 + 4.5 * np.sin(np.arange(n_hours2) * 2 * math.pi / 24 + 0.9)
trend2 = np.linspace(0.2, -0.2, n_hours2)
noise2 = np.random.normal(0, 0.55, n_hours2)
OT2 = base2 + trend2 + noise2
M12 = 19 + 2.8 * np.sin(np.arange(n_hours2) * 2 * math.pi / 24 + 0.4) + np.random.normal(0,0.7,n_hours2)
M22 = 26 + 3.2 * np.sin(np.arange(n_hours2) * 2 * math.pi / 24 + 0.9) + np.random.normal(0,0.6,n_hours2)
M32 = 7 + 1.3 * np.cos(np.arange(n_hours2) * 2 * math.pi / 24) + np.random.normal(0,0.25,n_hours2)

df3 = pd.DataFrame({
    'date': [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates2],
    'OT': OT2,
    'M1': M12,
    'M2': M22,
    'M3': M32,
})
file3 = os.path.join(out_dir, 'ETTm2.csv')
df3.to_csv(file3, index=False)
print('Wrote', file3)

print('All files written to', out_dir)
