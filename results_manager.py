#!/usr/bin/env python3
"""
Results Manager for Model Comparisons
Handles storing, loading, and comparing results from different models
"""

import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

class ResultsManager:
    """Manages experimental results for model comparison"""
    
    def __init__(self, results_dir: str = "experiment_results"):
        """
        Initialize Results Manager
        
        Args:
            results_dir: Directory to store results
        """
        self.results_dir = results_dir
        Path(results_dir).mkdir(exist_ok=True)
        self.results_file = os.path.join(results_dir, "results.json")
        self.load_results()
    
    def load_results(self) -> None:
        """Load results from file"""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r') as f:
                    self.results = json.load(f)
            except:
                self.results = {}
        else:
            self.results = {}
    
    def save_results(self) -> None:
        """Save results to file"""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def add_result(self, 
                   model_name: str,
                   dataset: str,
                   pred_len: int,
                   mse: float,
                   mae: float,
                   rmse: Optional[float] = None,
                   mape: Optional[float] = None,
                   notes: str = "",
                   timestamp: Optional[str] = None) -> None:
        """
        Add a test result
        
        Args:
            model_name: Name of the model (e.g., 'TimesMamba', 'iTransformer')
            dataset: Dataset name (e.g., 'ETTh1', 'Electricity')
            pred_len: Prediction length (forecast horizon)
            mse: Mean Squared Error
            mae: Mean Absolute Error
            rmse: Root Mean Squared Error (optional)
            mape: Mean Absolute Percentage Error (optional)
            notes: Additional notes about the experiment
            timestamp: When the experiment was run (auto if None)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        key = f"{model_name}_{dataset}_{pred_len}"
        
        self.results[key] = {
            "model": model_name,
            "dataset": dataset,
            "pred_len": pred_len,
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse) if rmse is not None else None,
            "mape": float(mape) if mape is not None else None,
            "notes": notes,
            "timestamp": timestamp
        }
        
        self.save_results()
    
    def get_model_comparison(self, 
                           models: Optional[List[str]] = None,
                           datasets: Optional[List[str]] = None,
                           pred_lens: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Get comparison data as DataFrame
        
        Args:
            models: List of model names to include (None = all)
            datasets: List of dataset names to include (None = all)
            pred_lens: List of prediction lengths to include (None = all)
        
        Returns:
            DataFrame with comparison data
        """
        filtered_results = []
        
        for result in self.results.values():
            if models and result['model'] not in models:
                continue
            if datasets and result['dataset'] not in datasets:
                continue
            if pred_lens and result['pred_len'] not in pred_lens:
                continue
            
            filtered_results.append(result)
        
        if not filtered_results:
            return pd.DataFrame()
        
        return pd.DataFrame(filtered_results)
    
    def get_pivot_table(self,
                       metric: str = "mse",
                       models: Optional[List[str]] = None,
                       datasets: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create a pivot table comparing models and datasets
        Format: Rows = Datasets/Prediction Length, Columns = Models, Values = Metric
        
        Args:
            metric: Metric to display ('mse', 'mae', 'rmse', 'mape')
            models: List of model names to include
            datasets: List of dataset names to include
        
        Returns:
            Pivot table DataFrame
        """
        df = self.get_model_comparison(models=models, datasets=datasets)
        
        if df.empty:
            return pd.DataFrame()
        
        # Create a combined key for multiple datasets/pred_lens
        df['key'] = df.apply(lambda x: f"{x['dataset']}_L{x['pred_len']}", axis=1)
        
        # Pivot: rows = key, columns = model, values = metric
        pivot = df.pivot_table(
            index='key',
            columns='model',
            values=metric,
            aggfunc='first'
        )
        
        return pivot
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics across all results"""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results.values())
        
        summary = {
            'total_experiments': len(self.results),
            'num_models': df['model'].nunique(),
            'num_datasets': df['dataset'].nunique(),
            'models': sorted(df['model'].unique().tolist()),
            'datasets': sorted(df['dataset'].unique().tolist()),
            'mse_stats': {
                'mean': float(df['mse'].mean()),
                'min': float(df['mse'].min()),
                'max': float(df['mse'].max())
            },
            'mae_stats': {
                'mean': float(df['mae'].mean()),
                'min': float(df['mae'].min()),
                'max': float(df['mae'].max())
            }
        }
        
        return summary
    
    def get_best_model(self, dataset: str, pred_len: int, metric: str = "mse") -> Optional[tuple]:
        """
        Get the best model for a specific dataset and prediction length
        
        Args:
            dataset: Dataset name
            pred_len: Prediction length
            metric: Metric to compare ('mse' or 'mae')
        
        Returns:
            Tuple of (model_name, metric_value) or None
        """
        df = self.get_model_comparison(datasets=[dataset], pred_lens=[pred_len])
        
        if df.empty:
            return None
        
        best_idx = df[metric].idxmin()
        best_result = df.loc[best_idx]
        
        return (best_result['model'], best_result[metric])
    
    def export_to_csv(self, filename: str = None) -> str:
        """
        Export results to CSV
        
        Args:
            filename: Output filename (default: results.csv in results_dir)
        
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = os.path.join(self.results_dir, "results.csv")
        
        df = pd.DataFrame(self.results.values())
        df.to_csv(filename, index=False)
        
        return filename
    
    def clear_results(self) -> None:
        """Clear all results (use with caution!)"""
        self.results = {}
        self.save_results()


# Example usage and test data
def create_sample_results():
    """Create sample comparison results (like in the image)"""
    manager = ResultsManager()
    
    # Sample data from the image
    sample_data = [
        # ETTh1
        ("TimesMamba", "ETTh1", 96, 0.375, 0.397),
        ("iTransformer", "ETTh1", 96, 0.386, 0.405),
        ("PatchTST", "ETTh1", 96, 0.414, 0.419),
        
        ("TimesMamba", "ETTh1", 192, 0.425, 0.424),
        ("iTransformer", "ETTh1", 192, 0.441, 0.436),
        ("PatchTST", "ETTh1", 192, 0.460, 0.445),
        
        ("TimesMamba", "ETTh1", 336, 0.468, 0.446),
        ("iTransformer", "ETTh1", 336, 0.487, 0.458),
        ("PatchTST", "ETTh1", 336, 0.501, 0.466),
        
        ("TimesMamba", "ETTh1", 720, 0.479, 0.473),
        ("iTransformer", "ETTh1", 720, 0.503, 0.491),
        ("PatchTST", "ETTh1", 720, 0.500, 0.488),
        
        # Electricity
        ("TimesMamba", "Electricity", 96, 0.141, 0.237),
        ("iTransformer", "Electricity", 96, 0.148, 0.240),
        ("PatchTST", "Electricity", 96, 0.195, 0.285),
        
        ("TimesMamba", "Electricity", 192, 0.156, 0.249),
        ("iTransformer", "Electricity", 192, 0.162, 0.253),
        ("PatchTST", "Electricity", 192, 0.199, 0.289),
        
        ("TimesMamba", "Electricity", 336, 0.172, 0.266),
        ("iTransformer", "Electricity", 336, 0.178, 0.269),
        ("PatchTST", "Electricity", 336, 0.215, 0.305),
        
        ("TimesMamba", "Electricity", 720, 0.204, 0.295),
        ("iTransformer", "Electricity", 720, 0.225, 0.317),
        ("PatchTST", "Electricity", 720, 0.256, 0.337),
        
        # Traffic
        ("TimesMamba", "Traffic", 96, 0.376, 0.257),
        ("iTransformer", "Traffic", 96, 0.395, 0.268),
        ("PatchTST", "Traffic", 96, 0.544, 0.359),
        
        ("TimesMamba", "Traffic", 192, 0.386, 0.265),
        ("iTransformer", "Traffic", 192, 0.417, 0.276),
        ("PatchTST", "Traffic", 192, 0.540, 0.354),
        
        ("TimesMamba", "Traffic", 336, 0.412, 0.277),
        ("iTransformer", "Traffic", 336, 0.433, 0.283),
        ("PatchTST", "Traffic", 336, 0.551, 0.358),
        
        ("TimesMamba", "Traffic", 720, 0.458, 0.296),
        ("iTransformer", "Traffic", 720, 0.467, 0.302),
        ("PatchTST", "Traffic", 720, 0.586, 0.375),
    ]
    
    # Add all sample data
    for model, dataset, pred_len, mse, mae in sample_data:
        manager.add_result(model, dataset, pred_len, mse, mae)
    
    return manager


if __name__ == "__main__":
    # Create sample data
    manager = create_sample_results()
    
    # Display summary
    summary = manager.get_summary_stats()
    print("📊 Results Summary:")
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Models: {summary['models']}")
    print(f"Datasets: {summary['datasets']}")
    
    # Display pivot table for MSE
    print("\n📈 MSE Comparison by Dataset and Prediction Length:")
    pivot_mse = manager.get_pivot_table(metric='mse')
    print(pivot_mse)
    
    # Display pivot table for MAE
    print("\n📈 MAE Comparison by Dataset and Prediction Length:")
    pivot_mae = manager.get_pivot_table(metric='mae')
    print(pivot_mae)
    
    # Export to CSV
    csv_file = manager.export_to_csv()
    print(f"\n✅ Results exported to {csv_file}")
