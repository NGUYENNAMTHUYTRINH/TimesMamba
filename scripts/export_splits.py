#!/usr/bin/env python3
"""Export train/val/test CSVs for ETT-small datasets using same split logic as loader."""
import os
import pandas as pd
import argparse


def compute_borders(n_rows, seq_len=96, minute=False):
    if minute:
        period = 12 * 30 * 24 * 4
        period2 = 4 * 30 * 24 * 4
    else:
        period = 12 * 30 * 24
        period2 = 4 * 30 * 24
    border1s = [0, period - seq_len, period + period2 - seq_len]
    border2s = [period, period + period2, period + 2 * period2]
    border1s = [max(0, min(n_rows, b)) for b in border1s]
    border2s = [max(0, min(n_rows, b)) for b in border2s]
    return border1s, border2s


def export(file_path, out_dir, seq_len=96):
    df = pd.read_csv(file_path)
    is_minute = 'm' in os.path.basename(file_path).lower()
    border1s, border2s = compute_borders(len(df), seq_len=seq_len, minute=is_minute)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(file_path))[0]
    df[border1s[0]:border2s[0]].to_csv(os.path.join(out_dir, f"{base}_train.csv"), index=False)
    df[border1s[1]:border2s[1]].to_csv(os.path.join(out_dir, f"{base}_val.csv"), index=False)
    df[border1s[2]:border2s[2]].to_csv(os.path.join(out_dir, f"{base}_test.csv"), index=False)
    print(f"Exported splits for {base} to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    parser.add_argument('--out_dir', default='dataset/ETT-small/splits')
    parser.add_argument('--seq_len', type=int, default=96)
    args = parser.parse_args()
    export(args.file, args.out_dir, seq_len=args.seq_len)


if __name__ == '__main__':
    main()
