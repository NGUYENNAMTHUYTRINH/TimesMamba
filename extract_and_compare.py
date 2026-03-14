#!/usr/bin/env python3
"""
Auto-extract metrics from all models and populate results.json
This script runs all models and collects their results for comparison
"""
import os
import sys
import subprocess
import json
import pandas as pd
import glob
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

sys.path.append(os.getcwd())

from results_manager import ResultsManager


def run_train_script(model_name, model_dir):
    """Run training script for a model"""
    script_name = {
        'TimesMamba': 'train/train.py',
        'RNN': 'RNN/train_rnn.py',
        'ITransformer': 'ITransformer/train_itransformer.py'
    }.get(model_name)
    
    if not script_name:
        print(f"⚠️ Unknown model: {model_name}")
        return False
    
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING {model_name}")
    print(f"{'='*60}")
    
    try:
        # Run the training script with its directory as cwd so relative paths inside the script resolve
        script_dir = os.path.join(os.getcwd(), os.path.dirname(script_name))
        script_base = os.path.basename(script_name)
        result = subprocess.run(
            f"{sys.executable} {script_base}",
            shell=True,
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print(f"✅ {model_name} training completed")
            return True
        else:
            print(f"❌ {model_name} training failed")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ {model_name} training timed out")
        return False
    except Exception as e:
        print(f"❌ Error training {model_name}: {e}")
        return False


def run_test_script(model_name):
    """Run testing script for a model"""
    script_name = {
        'TimesMamba': 'test/test.py',
        'RNN': 'RNN/test_rnn.py',
        'ITransformer': 'ITransformer/test_itransformer.py'
    }.get(model_name)
    
    if not script_name:
        print(f"⚠️ Unknown model: {model_name}")
        return False, None
    
    print(f"\n{'='*60}")
    print(f"🧪 TESTING {model_name}")
    print(f"{'='*60}")
    
    try:
        # Run the test script from its own directory so relative file paths inside tests work
        script_dir = os.path.join(os.getcwd(), os.path.dirname(script_name))
        script_base = os.path.basename(script_name)
        result = subprocess.run(
            f"{sys.executable} {script_base}",
            shell=True,
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        if result.returncode == 0:
            print(f"✅ {model_name} testing completed")

            # Try to find any results CSV for this model in common locations
            pattern = f"**/results_{model_name}_*.csv"
            matches = glob.glob(pattern, recursive=True)

            if matches:
                # pick the first match
                results_file = matches[0]
                try:
                    df = pd.read_csv(results_file)
                    return True, df.to_dict('records')[0] if len(df) > 0 else None
                except Exception as e:
                    print(f"⚠️ Failed to read results file {results_file}: {e}")
                    return True, None
            else:
                # fallback: original location used by some scripts
                results_file = f"{model_name.lower()}/predictions/results_{model_name}_ETTh1.csv"
                if os.path.exists(results_file):
                    df = pd.read_csv(results_file)
                    return True, df.to_dict('records')[0] if len(df) > 0 else None
                print(f"⚠️ Results file not found for {model_name} (looked for pattern {pattern})")
                return True, None
        else:
            print(f"❌ {model_name} testing failed")
            print(result.stderr)
            return False, None
    except subprocess.TimeoutExpired:
        print(f"❌ {model_name} testing timed out")
        return False, None
    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        return False, None


def extract_existing_results():
    """Extract results from existing prediction files"""
    manager = ResultsManager()
    models = ['TimesMamba', 'RNN', 'ITransformer']
    extracted_count = 0
    
    print(f"\n{'='*60}")
    print("📊 EXTRACTING EXISTING RESULTS")
    print(f"{'='*60}")
    
    for model in models:
        # search for any results files matching the model pattern
        pattern = f"**/results_{model}_*.csv"
        matches = glob.glob(pattern, recursive=True)

        # Special-case TimesMamba older output
        if model == 'TimesMamba':
            matches += glob.glob('test/predictions_*.csv')

            for result_file in matches:
                if os.path.exists(result_file):
                    try:
                        df = pd.read_csv(result_file)

                        # If file already contains metrics columns, use them directly
                        if 'mse' in df.columns and 'mae' in df.columns:
                            for idx, row in df.iterrows():
                                dataset = row.get('dataset', None)
                                if dataset is None:
                                    # infer dataset from filename
                                    if 'ETTh1' in result_file:
                                        dataset = 'ETTh1'
                                    elif 'ETTh2' in result_file:
                                        dataset = 'ETTh2'
                                    elif 'ETTm1' in result_file:
                                        dataset = 'ETTm1'
                                    elif 'ETTm2' in result_file:
                                        dataset = 'ETTm2'
                                    else:
                                        dataset = 'ETTh1'

                                manager.add_result(
                                    model_name=model,
                                    dataset=dataset,
                                    pred_len=int(row.get('pred_len', 24)),
                                    mse=float(row.get('mse', 0)),
                                    mae=float(row.get('mae', 0)),
                                    notes=f"Extracted from {result_file}"
                                )
                                extracted_count += 1
                                print(f"  ✓ {model} - {dataset}: MSE={float(row.get('mse')):.4f}")
                        else:
                            # Likely a predictions CSV (step_... columns). Compute metrics vs test set.
                            # Infer dataset from filename
                            dataset = None
                            for d in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
                                if d in result_file:
                                    dataset = d
                                    break
                            if dataset is None:
                                dataset = 'ETTh1'

                            # Load predictions dataframe and reshape
                            try:
                                pred_df = df
                                # detect normalized filename
                                is_normalized = 'normalized' in os.path.basename(result_file)

                                # columns like step_0_OT ... step_23_M3
                                cols = [c for c in pred_df.columns if c.startswith('step_')]
                                if not cols:
                                    print(f"  ⚠️ No step_ columns in {result_file}, skipping")
                                    continue

                                # determine pred_len and num_features
                                steps = sorted(list({int(c.split('_')[1]) for c in cols}))
                                pred_len = max(steps) + 1
                                features = sorted(list({c.split('_')[2] for c in cols}))
                                n_features = len(features)

                                # reshape
                                arr = pred_df[cols].values
                                n_seq = arr.shape[0]
                                try:
                                    preds = arr.reshape(n_seq, pred_len, n_features)
                                except Exception:
                                    preds = arr.reshape(n_seq, n_features, pred_len).transpose(0,2,1)

                                # Load test dataset to build actual targets
                                test_file = os.path.join('test', 'datasets', dataset, f"{dataset}_test.csv")
                                if not os.path.exists(test_file):
                                    print(f"  ⚠️ Test file not found for dataset {dataset}: {test_file}")
                                    continue

                                test_df = pd.read_csv(test_file)
                                numeric_cols = ['OT', 'M1', 'M2', 'M3']
                                data = test_df[numeric_cols].values.astype(float)

                                # If predictions are normalized, normalize actuals same way
                                if is_normalized:
                                    mean = data.mean(axis=0)
                                    std = data.std(axis=0) + 1e-8
                                    data = (data - mean) / std

                                seq_len = 96
                                # build targets same as simple_test.py
                                targets = []
                                for i in range(n_seq):
                                    start = i + seq_len
                                    end = start + pred_len
                                    if end <= len(data):
                                        targets.append(data[start:end])
                                    else:
                                        pass

                                if len(targets) == 0:
                                    print(f"  ⚠️ Not enough test data to compute metrics for {result_file}")
                                    continue

                                targets = np.array(targets[:n_seq])
                                preds = preds[:targets.shape[0]]

                                # compute mse/mae across all features/time
                                mse = mean_squared_error(targets.flatten(), preds.flatten())
                                mae = mean_absolute_error(targets.flatten(), preds.flatten())

                                manager.add_result(
                                    model_name=model,
                                    dataset=dataset,
                                    pred_len=pred_len,
                                    mse=float(mse),
                                    mae=float(mae),
                                    notes=f"Computed from {result_file}"
                                )
                                extracted_count += 1
                                print(f"  ✓ {model} - {dataset}: MSE={mse:.6f}")
                            except Exception as e:
                                print(f"  ⚠️ Failed computing metrics for {result_file}: {e}")
                    except Exception as e:
                        print(f"  ⚠️ Error reading {result_file}: {e}")
    
    return extracted_count


def main():
    print("\n")
    print("🎯" * 30)
    print("   AUTO-EXTRACT METRICS FOR MODEL COMPARISON")
    print("🎯" * 30)
    
    # Initialize
    manager = ResultsManager()
    manager.clear_results()  # Start fresh
    
    models = ['TimesMamba', 'RNN', 'ITransformer']
    
    print(f"\n📋 Models to process: {', '.join(models)}")
    print(f"📊 Dataset: ETTh1, ETTh2, ETTm1, ETTm2")
    print(f"⏱️  This may take 30-60 minutes depending on hardware")
    
    input("\n✋ Press Enter to continue, or Ctrl+C to cancel...")
    
    # Check if PyTorch is available; if not, skip training/testing and only extract existing predictions
    try:
        import torch
        # ensure torch.nn is available (some stub packages may import as 'torch' without full API)
        if hasattr(torch, 'nn'):
            has_torch = True
        else:
            has_torch = False
            print("[INFO] 'torch' module found but 'torch.nn' missing — skipping train/test steps.")
    except Exception:
        has_torch = False
        print("[INFO] PyTorch not found in environment — skipping train/test steps.")

    # Train and test each model (only if torch available)
    results_summary = {}

    if has_torch:
        for model in models:
            print(f"\n🔄 Processing {model}...")

            # Train
            train_success = run_train_script(model, model.lower())
            if not train_success:
                print(f"⚠️ Skipping {model} test due to training failure")
                continue

            # Test
            test_success, test_results = run_test_script(model)
            if test_success and test_results:
                results_summary[model] = test_results

                # Add to results manager
                manager.add_result(
                    model_name=test_results.get('model', model),
                    dataset=test_results.get('dataset', 'ETTh1'),
                    pred_len=int(test_results.get('pred_len', 24)),
                    mse=float(test_results.get('mse', 0)),
                    mae=float(test_results.get('mae', 0)),
                    notes=f"Auto-extracted from test results"
                )
    else:
        print("[INFO] Running extraction from existing prediction files only.")
    
    # Extract from existing files
    extracted = extract_existing_results()
    # Reload manager to pick up newly written results
    manager.load_results()
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 EXTRACTION COMPLETE")
    print(f"{'='*60}")
    
    summary = manager.get_summary_stats()
    if summary:
        print(f"✅ Total experiments: {summary['total_experiments']}")
        print(f"✅ Models: {', '.join(summary['models'])}")
        print(f"✅ Datasets: {', '.join(summary['datasets'])}")
        print(f"\n💾 Results saved to: experiment_results/results.json")
        print(f"\n🚀 Run Streamlit to view comparison:")
        print(f"   streamlit run streamlit_app.py")
    else:
        print("⚠️ No results extracted")
    
    print(f"\n{'='*60}")
    print("Done! 🎉")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
