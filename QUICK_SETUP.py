#!/usr/bin/env python3
"""
🏆 MODEL COMPARISON SYSTEM
Quick Setup & Usage Guide

Everything you need is ready. Here's how to start:
"""

# ============================================================================
# QUICK START (30 seconds)
# ============================================================================

"""
1. Load example data (already done ✅):
   python add_results.py example

2. Start dashboard:
   streamlit run streamlit_app.py

3. Click: 🏆 Comparison tab

4. See: Model comparison tables like your image ✅
"""

# ============================================================================
# WHAT WAS CREATED
# ============================================================================

FILES_CREATED = {
    "results_manager.py": {
        "purpose": "Core comparison system",
        "features": [
            "Store results in JSON",
            "Create pivot tables",
            "Generate rankings",
            "Export to CSV"
        ]
    },
    "add_results.py": {
        "purpose": "Add results easily",
        "features": [
            "Interactive mode",
            "Bulk import",
            "Load examples"
        ]
    },
    "streamlit_app.py": {
        "purpose": "Updated dashboard with comparison tab",
        "features": [
            "MSE/MAE comparison tables",
            "Green highlights best values",
            "Filter by models/datasets",
            "Performance charts",
            "Best model rankings",
            "Manual result entry"
        ]
    },
    "Documentation": {
        "MODEL_COMPARISON_GUIDE.md": "Complete feature documentation",
        "COMPARISON_VISUAL_PREVIEW.md": "What you'll see in Streamlit",
        "INTEGRATE_TEST_RESULTS.md": "How to add your own results"
    }
}

# ============================================================================
# THE TABLE YOU WANTED
# ============================================================================

"""
Your image showed this format:

        TimesMamba  iTransformer  PatchTST      MSE Scores
ETTh1                                               (Lower is Better ↓)
  L96      0.375 ✅   0.386       0.414
  L192     0.425 ✅   0.441       0.460
  L336     0.468 ✅   0.487       0.501
  L720     0.479 ✅   0.503       0.500

Electricity
  L96      0.141 ✅   0.148       0.195
  ...

Traffic
  ...

YOU NOW HAVE IT IN STREAMLIT! ✅
"""

# ============================================================================
# THREE WAYS TO USE
# ============================================================================

USAGE_OPTIONS = {
    "Option 1: View Example Results (DEMO)": {
        "steps": [
            "python add_results.py example",
            "streamlit run streamlit_app.py",
            "Click 🏆 Comparison tab"
        ],
        "result": "See sample model comparisons"
    },
    
    "Option 2: Add Your Results via UI": {
        "steps": [
            "streamlit run streamlit_app.py",
            "Go to 🏆 Comparison → Manage Results",
            "Click Add New Experimental Result",
            "Fill form and save"
        ],
        "result": "Results appear in comparison tables"
    },
    
    "Option 3: Add Results via Python": {
        "code": """
from results_manager import ResultsManager
m = ResultsManager()
m.add_result(
    model_name="TimesMamba",
    dataset="ETTh1",
    pred_len=96,
    mse=0.375,
    mae=0.397
)
""",
        "result": "Quick programmatic integration"
    }
}

# ============================================================================
# COMPARISON TAB FEATURES
# ============================================================================

COMPARISON_TAB = """
When you click 🏆 Comparison Tab, you get:

1. 📊 EXPERIMENT SUMMARY
   - Total experiments count
   - Number of models
   - Number of datasets
   - Average metrics

2. 📈 MODEL COMPARISON TABLES
   [MSE Comparison]  [MAE Comparison]
   - Green = Best (lowest value) in each row
   - Rows = Dataset_LengthXX
   - Columns = Model names
   - Values = MSE/MAE scores

3. 🔍 DETAILED COMPARISON VIEW
   - Filter by models
   - Filter by datasets
   - View performance charts
   - See trends across prediction lengths

4. 🥇 BEST MODEL RANKINGS
   - Best model for each dataset/pred_len
   - Ranked by MSE and MAE
   - Shows exact scores

5. ⚙️ RESULTS MANAGEMENT
   - Add new results manually
   - Clear all results
   - Export to CSV
"""

# ============================================================================
# UNDERSTANDING THE DATA STRUCTURE
# ============================================================================

DATA_STRUCTURE = """
Results stored in: experiment_results/results.json

Format:
{
  "TimesMamba_ETTh1_96": {
    "model": "TimesMamba",
    "dataset": "ETTh1",
    "pred_len": 96,
    "mse": 0.375,
    "mae": 0.397,
    "rmse": null,
    "mape": null,
    "notes": "",
    "timestamp": "2026-03-14T13:54:00.111715"
  },
  ...
}

Key: model_dataset_predlen (unique identifier)
Each result tracks what, when, and how well it performed
"""

# ============================================================================
# COMMON OPERATIONS
# ============================================================================

OPERATIONS = {
    "Add single result": "python add_results.py interactive",
    
    "Load example data": "python add_results.py example",
    
    "View in Streamlit": "streamlit run streamlit_app.py",
    
    "Export results": "(In Streamlit tab: download button)",
    
    "Clear all results": "(In Streamlit tab: click 🗑️ Clear)",
    
    "View results programmatically": """
from results_manager import ResultsManager
m = ResultsManager()
df = m.get_model_comparison()
print(df)
""",
    
    "Get best model": """
m = ResultsManager()
best = m.get_best_model(dataset='ETTh1', pred_len=96, metric='mse')
print(f"Best: {best[0]} with MSE={best[1]}")
""",
    
    "Create pivot table": """
m = ResultsManager()
pivot = m.get_pivot_table(metric='mse')
print(pivot)
"""
}

# ============================================================================
# INTEGRATION WITH YOUR TESTS
# ============================================================================

INTEGRATION = """
To add results from your test.py:

1. At end of your test function, add:

   from results_manager import ResultsManager
   manager = ResultsManager()
   manager.add_result(
       model_name="TimesMamba",
       dataset=dataset_name,
       pred_len=prediction_length,
       mse=mse_score,
       mae=mae_score
   )

2. Or batch add multiple results:

   from add_results import bulk_import_from_dict
   results_list = [
       {"model": "M1", "dataset": "D1", "pred_len": 96, "mse": 0.1, "mae": 0.2},
       {"model": "M2", "dataset": "D1", "pred_len": 96, "mse": 0.15, "mae": 0.25},
   ]
   bulk_import_from_dict(results_list)
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

TROUBLESHOOTING = {
    "Results not showing?": [
        "✓ Make sure results.json exists",
        "✓ Refresh browser (F5) in Streamlit",
        "✓ Check experiment_results/ folder"
    ],
    
    "Need to update results?": [
        "✓ Edit experiment_results/results.json directly",
        "✓ Or clear and re-add via script"
    ],
    
    "Want to compare specific models?": [
        "✓ Use filter in Detailed Comparison View",
        "✓ Select models and datasets you want"
    ],
    
    "Export for presentation?": [
        "✓ Use CSV export in Streamlit",
        "✓ Import to Excel/PowerPoint",
        "✓ Or print from Streamlit directly"
    ]
}

# ============================================================================
# NEXT STEPS
# ============================================================================

NEXT_STEPS = """
1. ✅ Load example data (DONE):
   python add_results.py example

2. ✅ Try Streamlit dashboard:
   streamlit run streamlit_app.py
   Click 🏆 Comparison tab

3. 📊 Train TimesMamba on your data:
   python train/train.py

4. 📈 Test and get metrics (MSE, MAE)

5. ➕ Add results via:
   - Interactive: python add_results.py interactive
   - Or Streamlit UI: Manage Results section
   - Or Python code: See examples above

6. 🏆 View comparison table:
   Refresh browser in Streamlit

7. 📥 Export results:
   Click download button in Streamlit
"""

# ============================================================================
# QUICK REFERENCE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🏆 MODEL COMPARISON SYSTEM - READY TO USE")
    print("="*70)
    
    print("\n📁 FILES CREATED:")
    for file, info in FILES_CREATED.items():
        print(f"\n  {file}:")
        if isinstance(info, dict):
            for key, val in info.items():
                if key == "features":
                    for feature in val:
                        print(f"    • {feature}")
                elif isinstance(val, str):
                    print(f"    {key}: {val}")
    
    print("\n\n🚀 GET STARTED IN 3 STEPS:")
    print("\n  1. python add_results.py example")
    print("  2. streamlit run streamlit_app.py")
    print("  3. Click 🏆 Comparison tab")
    
    print("\n\n📊 YOU GET:")
    print("  ✅ Model comparison tables (your image format)")
    print("  ✅ MSE & MAE metrics side-by-side")
    print("  ✅ Green highlighting best models")
    print("  ✅ Filter & chart tools")
    print("  ✅ Best model rankings")
    print("  ✅ CSV export")
    
    print("\n\n📖 DOCUMENTATION:")
    print("  • MODEL_COMPARISON_GUIDE.md - Full feature guide")
    print("  • COMPARISON_VISUAL_PREVIEW.md - What you'll see")
    print("  • INTEGRATE_TEST_RESULTS.md - Add your results")
    
    print("\n\n✨ Everything is ready! Happy comparing! 🎯\n")
