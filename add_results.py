#!/usr/bin/env python3
"""
Quick Results Adder Script
Easily add experimental results to the comparison database
"""

from results_manager import ResultsManager
import sys

def add_result_interactive():
    """Interactive mode to add a single result"""
    manager = ResultsManager()
    
    print("\n" + "="*60)
    print("📊 Add Experimental Result")
    print("="*60)
    
    model_name = input("Model name (e.g., TimesMamba, iTransformer): ").strip()
    if not model_name:
        print("❌ Model name required!")
        return
    
    dataset = input("Dataset name (e.g., ETTh1, Electricity, Traffic): ").strip()
    if not dataset:
        print("❌ Dataset required!")
        return
    
    try:
        pred_len = int(input("Prediction length (e.g., 96, 192, 336, 720): ").strip())
    except ValueError:
        print("❌ Invalid prediction length!")
        return
    
    try:
        mse = float(input("MSE value: ").strip())
        mae = float(input("MAE value: ").strip())
    except ValueError:
        print("❌ Invalid MSE or MAE value!")
        return
    
    rmse_input = input("RMSE (press Enter to skip): ").strip()
    rmse = float(rmse_input) if rmse_input else None
    
    mape_input = input("MAPE (press Enter to skip): ").strip()
    mape = float(mape_input) if mape_input else None
    
    notes = input("Notes/Comments (optional): ").strip()
    
    # Add the result
    manager.add_result(
        model_name=model_name,
        dataset=dataset,
        pred_len=pred_len,
        mse=mse,
        mae=mae,
        rmse=rmse,
        mape=mape,
        notes=notes
    )
    
    print(f"\n✅ Result added successfully!")
    print(f"   Model: {model_name}")
    print(f"   Dataset: {dataset}")
    print(f"   Prediction Length: {pred_len}")
    print(f"   MSE: {mse:.6f}")
    print(f"   MAE: {mae:.6f}")


def bulk_import_from_dict(results_list):
    """
    Import multiple results at once from a dictionary list
    
    Usage:
        results = [
            {"model": "TimesMamba", "dataset": "ETTh1", "pred_len": 96, "mse": 0.375, "mae": 0.397},
            {"model": "iTransformer", "dataset": "ETTh1", "pred_len": 96, "mse": 0.386, "mae": 0.405},
            ...
        ]
        bulk_import_from_dict(results)
    """
    manager = ResultsManager()
    
    added = 0
    for result in results_list:
        try:
            manager.add_result(
                model_name=result.get('model'),
                dataset=result.get('dataset'),
                pred_len=result.get('pred_len'),
                mse=result.get('mse'),
                mae=result.get('mae'),
                rmse=result.get('rmse'),
                mape=result.get('mape'),
                notes=result.get('notes', '')
            )
            added += 1
        except Exception as e:
            print(f"⚠️ Failed to add result: {e}")
    
    print(f"✅ Added {added}/{len(results_list)} results")


def load_example_results():
    """
    Load example results that look like your comparison image
    """
    from results_manager import create_sample_results
    manager = create_sample_results()
    print("✅ Example results loaded!")
    
    # Display summary
    summary = manager.get_summary_stats()
    print(f"\nLoaded {summary['total_experiments']} experimental results:")
    print(f"  Models: {', '.join(summary['models'])}")
    print(f"  Datasets: {', '.join(summary['datasets'])}")
    
    # Show sample pivot table
    print("\n📊 Sample MSE Comparison Table:")
    pivot = manager.get_pivot_table(metric='mse')
    print(pivot)


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "example":
            # Load example results
            load_example_results()
        elif sys.argv[1] == "interactive":
            # Interactive mode
            add_result_interactive()
        else:
            print("Usage:")
            print("  python add_results.py example          - Load example comparison data")
            print("  python add_results.py interactive      - Add results one by one")
    else:
        # Default: show menu
        print("\n" + "="*60)
        print("📊 Results Manager - Quick Add Tool")
        print("="*60)
        print("\nOptions:")
        print("  1) Load example comparison data (from your image)")
        print("  2) Add a single result interactively")
        print("  3) Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            load_example_results()
        elif choice == "2":
            add_result_interactive()
        else:
            print("Goodbye!")
