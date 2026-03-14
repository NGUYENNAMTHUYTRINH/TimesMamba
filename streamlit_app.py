import os
import sys
import pandas as pd
import numpy as np
try:
    import streamlit as st
except Exception:
    print("Streamlit is not installed. Install it in your environment with:\n.venv\\Scripts\\python.exe -m pip install streamlit")
    raise
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path

# Resolve project root and ensure it's on sys.path for imports
script_dir = Path(__file__).resolve().parent
project_root = str(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from results_manager import ResultsManager

def show_train_tab():
    """Training tab content"""
    st.header("🚀 TimesMamba Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dataset = st.selectbox("Select Dataset", ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'])
        epochs = st.number_input("Training Epochs", value=10, min_value=1, max_value=100)
        batch_size = st.number_input("Batch Size", value=32, min_value=1, max_value=128)
        learning_rate = st.number_input("Learning Rate", value=0.0001, min_value=0.00001, max_value=0.01, format="%.5f")
    
    with col2:
        st.subheader("Training Data Info")
        train_file = f"train/datasets/{dataset}/{dataset}_train.csv"
        if os.path.exists(train_file):
            df = pd.read_csv(train_file)
            st.write(f"📊 Rows: {len(df)}")
            st.write(f"📋 Columns: {list(df.columns)}")
            st.write(f"📈 Sample data:")
            st.dataframe(df.head(3))
        else:
            st.error(f"Training data not found: {train_file}")
    
    # Training controls
    if st.button("🚀 Start Training", type="primary"):
        if os.path.exists(train_file):
            with st.spinner(f"Training TimesMamba on {dataset}..."):
                try:
                    # Run training script using the same Python interpreter
                    train_script = os.path.join(project_root, 'train', 'train.py')
                    result = subprocess.run([sys.executable, train_script], capture_output=True, text=True, cwd=project_root)

                    if result.returncode == 0:
                        st.success(f"✅ Training completed for {dataset}!")
                        st.text(result.stdout)

                        # Show training plot if exists
                        plot_file = os.path.join(project_root, f"train/training_loss_{dataset}.png")
                        if os.path.exists(plot_file):
                            st.image(plot_file, caption=f"Training Loss - {dataset}")
                    else:
                        st.error(f"❌ Training failed!")
                        st.text(result.stderr)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.error("Please ensure training data exists!")
    
    # Show existing training results
    st.subheader("📊 Training Results")
    
    models = []
    for ds in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
        # models are saved in saved_models with names like best_model_<Model>_<Dataset>.pth
        saved_models_dir = os.path.join(project_root, 'saved_models')
        model_found = False
        if os.path.exists(saved_models_dir):
            for fname in os.listdir(saved_models_dir):
                if fname.endswith(f"_{ds}.pth"):
                    model_found = True
                    model_file = os.path.join(saved_models_dir, fname)
                    break

        # training plots are saved under figures/
        plot_file = None
        figures_dir = os.path.join(project_root, 'figures')
        if os.path.exists(figures_dir):
            for fname in os.listdir(figures_dir):
                if f"training_loss" in fname and fname.endswith(f"_{ds}.png"):
                    plot_file = os.path.join(figures_dir, fname)
                    break

        if model_found:
            models.append(ds)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"✅ **{ds}** - Model trained")
                size_mb = os.path.getsize(model_file) / (1024*1024)
                st.write(f"📦 Model size: {size_mb:.2f} MB")
                st.write(f"📈 Status: Best model saved during training")
                st.caption("(Validation loss monitored internally)")

            with col2:
                if plot_file and os.path.exists(plot_file):
                    st.image(plot_file, caption=f"Training Loss Curve - {ds}", width=400)
                else:
                    st.info(f"Training plot not yet available for {ds}")
    
    if not models:
        st.info("""❌ No trained models found. Start training to see results.

**Note:** During training, validation data is automatically used to:
- Monitor overfitting (val_loss curve)
- Save the best model
- Implement early stopping""")
    
    # Explain validation process
    with st.expander("ℹ️ Understanding Training & Validation"):
        st.markdown("""
        ### How Validation Works (Internal Process):
        
        **During Training (Each Epoch):**
        1. **Train Phase:** Run model on training data → calculate train_loss
        2. **Validation Phase:** Run model on validation data → calculate val_loss
        3. **Decision Making:**
           - If val_loss improves → Save as "best_model"
           - If val_loss doesn't improve for 3 epochs → Stop training (Early Stopping)
        
        **Why Validation?**
        - Prevent overfitting (when model memorizes data)
        - Choose the best checkpoint automatically
        - Use all data efficiently (70% train, 15% val, 15% test)
        
        **You don't need to run Testing on validation data** because:
        - Validation data was seen during training (affects model learning)
        - Test data is completely new (true evaluation)
        - Best model is already selected using validation
        
        **Result:** `best_model_ETTh1.pth` = model with lowest validation loss
        """)

def show_test_tab():
    """Testing tab content"""
    st.header("🧪 TimesMamba Testing")
    # common directories
    figures_dir = os.path.join(project_root, 'figures')
    
    st.markdown("""
    **Why Test on Test Data, Not Validation Data?**
    - ✅ **Test Data:** Completely new (model never seen before) → True evaluation
    - ❌ **Validation Data:** Was seen during training (affects model learning)
    
    **Data Split in Our Project:**
    - 70% Training data: Used to teach model
    - 15% Validation data: Used to monitor training & select best model
    - 15% Test data: Used to evaluate final performance (true metric)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        dataset = st.selectbox("Select Dataset for Testing", ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'])
        
        # Check if model exists
        model_file = f"train/best_model_{dataset}.pth"
        model_exists = os.path.exists(model_file)
        
        if model_exists:
            st.success(f"✅ Trained model found for {dataset}")
        else:
            st.warning(f"⚠️ No trained model for {dataset}. Will use random weights.")
    
    with col2:
        st.subheader("Test Data Info")
        test_file = f"test/datasets/{dataset}/{dataset}_test.csv"
        if os.path.exists(test_file):
            df = pd.read_csv(test_file)
            st.write(f"📊 Rows: {len(df)}")
            st.write(f"📋 Columns: {list(df.columns)}")
            st.write(f"📈 Sample data:")
            st.dataframe(df.head(3))
        else:
            st.error(f"Test data not found: {test_file}")
    
    # Testing controls
        if st.button("🧪 Run Test", type="primary"):
            if os.path.exists(test_file):
                with st.spinner(f"Testing TimesMamba on {dataset}..."):
                    try:
                        test_script = os.path.join(project_root, 'test', 'simple_test.py')
                        result = subprocess.run([sys.executable, test_script], capture_output=True, text=True, cwd=project_root)

                        if result.returncode == 0:
                            st.success(f"✅ Testing completed for {dataset}!")
                            st.text(result.stdout)

                            # Show test plot if exists (search figures directory)
                            test_plot = None
                            if os.path.exists(figures_dir):
                                for fname in os.listdir(figures_dir):
                                    if f"test_results" in fname and fname.endswith(f"_{dataset}.png"):
                                        test_plot = os.path.join(figures_dir, fname)
                                        break

                            if test_plot and os.path.exists(test_plot):
                                st.image(test_plot, caption=f"Test Results - {dataset}")
                        else:
                            st.error(f"❌ Testing failed!")
                            st.text(result.stderr)
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.error("Please ensure test data exists!")
    
    # Show existing test results
    st.subheader("📈 Test Results")
    
    results = []
    for ds in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
        test_plot = None
        if os.path.exists(figures_dir):
            for fname in os.listdir(figures_dir):
                if f"test_results" in fname and fname.endswith(f"_{ds}.png"):
                    test_plot = os.path.join(figures_dir, fname)
                    break

        if test_plot and os.path.exists(test_plot):
            results.append(ds)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"✅ **{ds}** - Test completed")

            with col2:
                st.image(test_plot, width=400)

    if not results:
        st.info("No test results found. Run tests to see results.")

def show_comparison_tab():
    """Model comparison tab"""
    st.header("🏆 Model Comparison")
    
    # Model info expander
    with st.expander("ℹ️ About the 3 Models"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🔵 TimesMamba")
            st.write("""
            **Architecture:** State Space Model (Mamba)
            - **Complexity:** O(n) Linear
            - **Speed:** ⚡⚡⚡ Fast
            - **SOTA:** Yes (2024)
            - **Expected MSE:** 0.37-0.40
            
            State-of-the-art accuracy with minimal parameters.
            """)
        
        with col2:
            st.subheader("🟢 RNN (LSTM)")
            st.write("""
            **Architecture:** Recurrent Neural Network
            - **Complexity:** O(n) Sequential
            - **Speed:** ⚡⚡ Moderate
            - **SOTA:** Classic Baseline
            - **Expected MSE:** 0.42-0.48
            
            Traditional approach, good for comparison.
            """)
        
        with col3:
            st.subheader("🟡 ITransformer")
            st.write("""
            **Architecture:** Individual Transformer
            - **Complexity:** O(n) per channel
            - **Speed:** ⚡⚡⚡ Fast
            - **Novel:** Recent (2023)
            - **Expected MSE:** 0.38-0.45
            
            Independent attention per variable.
            """)
    
    # Initialize results manager
    manager = ResultsManager()
    summary = manager.get_summary_stats()
    

    
    # Display summary statistics
    st.subheader("📊 Experiment Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Experiments", summary['total_experiments'])
    with col2:
        st.metric("Number of Models", summary['num_models'])
    with col3:
        st.metric("Number of Datasets", summary['num_datasets'])
    with col4:
        st.metric("Avg MSE", f"{summary['mse_stats']['mean']:.4f}")
    
    # Explain evaluation metrics
    st.info("""
    📌 **Understanding These Metrics:**
    - **MSE (Mean Squared Error):** Penalizes larger errors more (squared)
    - **MAE (Mean Absolute Error):** Average absolute deviation (easier to interpret)
    - **Lower scores are better!** ↓
    
    ⚠️ **These are TEST metrics** (on new data never seen by model)
    """)
    
    # First comparison tool: By Dataset and Prediction Length
    st.subheader("📈 Model Comparison by Dataset & Prediction Length")
    
    # Tabs for different metrics
    metric_tab1, metric_tab2 = st.tabs(["MSE Comparison", "MAE Comparison"])
    
    with metric_tab1:
        st.markdown("**Mean Squared Error (Lower is Better ↓)**")
        
        # Get pivot table for MSE
        pivot_mse = manager.get_pivot_table(metric='mse')
        
        if not pivot_mse.empty:
            # Style the dataframe to highlight best values
            def highlight_best(s):
                min_val = s.min()
                return [f'background-color: lightgreen' if v == min_val else '' for v in s]
            
            styled_df = pivot_mse.style.apply(highlight_best, axis=1).format("{:.3f}")
            st.dataframe(styled_df, use_container_width=True)
            
            # Export option
            csv = manager.export_to_csv()
            with open(csv, 'r') as f:
                csv_content = f.read()
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_content,
                file_name="model_comparison.csv",
                mime="text/csv"
            )
    
    with metric_tab2:
        st.markdown("**Mean Absolute Error (Lower is Better ↓)**")
        
        # Get pivot table for MAE
        pivot_mae = manager.get_pivot_table(metric='mae')
        
        if not pivot_mae.empty:
            def highlight_best(s):
                min_val = s.min()
                return [f'background-color: lightgreen' if v == min_val else '' for v in s]
            
            styled_df = pivot_mae.style.apply(highlight_best, axis=1).format("{:.3f}")
            st.dataframe(styled_df, use_container_width=True)
    
    # Get all data for best model rankings
    df_all = manager.get_model_comparison()
    
    # Second comparison tool: Filter by specific datasets/models
    st.subheader("🔍 Detailed Comparison View")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_models = st.multiselect(
            "Select Models to Compare",
            options=summary['models'],
            default=summary['models'][:2] if len(summary['models']) >= 2 else summary['models']
        )
    
    with col2:
        selected_datasets = st.multiselect(
            "Select Datasets",
            options=summary['datasets'],
            default=summary['datasets']
        )
    
    if selected_models and selected_datasets:
        # Get filtered comparison data
        df = manager.get_model_comparison(models=selected_models, datasets=selected_datasets)
        
        if not df.empty:
            # Create sortable table
            display_df = df[['model', 'dataset', 'pred_len', 'mse', 'mae']].copy()
            display_df.columns = ['Model', 'Dataset', 'Prediction Length', 'MSE', 'MAE']
            display_df = display_df.sort_values(['Dataset', 'Prediction Length', 'MSE'])
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Visualization
            st.subheader("📊 Performance Visualization")
            
            # Chart: MSE by Model
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Prepare data for plotting
            for idx, dataset in enumerate(selected_datasets[:2]):  # Plot first 2 datasets
                # Filter data for this dataset
                dataset_df = df[df['dataset'] == dataset].sort_values('pred_len')
                
                if not dataset_df.empty:
                    ax = axes[idx] if len(selected_datasets) <= 2 else axes[0]
                    
                    for model in selected_models:
                        model_data = dataset_df[dataset_df['model'] == model]
                        if not model_data.empty:
                            ax.plot(model_data['pred_len'], model_data['mse'], marker='o', label=model, linewidth=2)
                    
                    ax.set_xlabel('Prediction Length')
                    ax.set_ylabel('MSE')
                    ax.set_title(f'MSE vs Prediction Length - {dataset}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Best model by dataset
    st.subheader("🥇 Best Model Rankings")
    
    best_models_mse = []
    best_models_mae = []
    
    if not df_all.empty:
        for dataset in summary['datasets']:
            pred_lens = df_all[df_all['dataset'] == dataset]['pred_len'].unique()
            for pred_len in sorted(pred_lens):
                best_mse = manager.get_best_model(dataset, int(pred_len), metric='mse')
                best_mae = manager.get_best_model(dataset, int(pred_len), metric='mae')
                
                if best_mse:
                    best_models_mse.append({
                        'Dataset': dataset,
                        'Prediction Length': pred_len,
                        'Best Model (MSE)': best_mse[0],
                        'MSE Score': f"{best_mse[1]:.4f}"
                    })
                
                if best_mae:
                    best_models_mae.append({
                        'Dataset': dataset,
                        'Prediction Length': pred_len,
                        'Best Model (MAE)': best_mae[0],
                        'MAE Score': f"{best_mae[1]:.4f}"
                    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Best Models by MSE**")
        if best_models_mse:
            st.dataframe(pd.DataFrame(best_models_mse), use_container_width=True, hide_index=True)
    
    with col2:
        st.write("**Best Models by MAE**")
        if best_models_mae:
            st.dataframe(pd.DataFrame(best_models_mae), use_container_width=True, hide_index=True)
    
    # Results management - Clear option
    st.subheader("⚙️ Manage Results")
    
    if st.button("🗑️ Clear All Results", type="secondary"):
        manager.clear_results()
        st.success("All results cleared")
        st.rerun()


def main():
    st.set_page_config(
        page_title="TimesMamba Dashboard",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("⚡ TimesMamba Training & Testing Dashboard")
    st.markdown("Train and test TimesMamba models on ETT datasets")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🚀 Training", "🧪 Testing", "🏆 Comparison"])
    
    with tab1:
        show_train_tab()
    
    with tab2:
        show_test_tab()
    
    with tab3:
        show_comparison_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("TimesMamba: State-of-the-art time series forecasting with Mamba architecture")

if __name__ == '__main__':
    main()
