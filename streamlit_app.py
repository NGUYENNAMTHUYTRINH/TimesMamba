import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.getcwd())

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
                    # Run training script
                    cmd = f"cd train && python train.py"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd())
                    
                    if result.returncode == 0:
                        st.success(f"✅ Training completed for {dataset}!")
                        st.text(result.stdout)
                        
                        # Show training plot if exists
                        plot_file = f"train/training_loss_{dataset}.png"
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
        model_file = f"train/best_model_{ds}.pth"
        plot_file = f"train/training_loss_{ds}.png"
        
        if os.path.exists(model_file):
            models.append(ds)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"✅ **{ds}** - Model trained")
                size_mb = os.path.getsize(model_file) / (1024*1024)
                st.write(f"Model size: {size_mb:.2f} MB")
            
            with col2:
                if os.path.exists(plot_file):
                    st.image(plot_file, width=400)
    
    if not models:
        st.info("No trained models found. Start training to see results.")

def show_test_tab():
    """Testing tab content"""
    st.header("🧪 TimesMamba Testing")
    
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
                    # Run simple test script (the working one)
                    cmd = f"cd test && python simple_test.py"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd())
                    
                    if result.returncode == 0:
                        st.success(f"✅ Testing completed for {dataset}!")
                        st.text(result.stdout)
                        
                        # Show test plot if exists
                        plot_file = f"test/simple_test_{dataset}.png"
                        if os.path.exists(plot_file):
                            st.image(plot_file, caption=f"Test Results - {dataset}")
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
        plot_file = f"test/simple_test_{ds}.png"
        
        if os.path.exists(plot_file):
            results.append(ds)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"✅ **{ds}** - Test completed")
            
            with col2:
                st.image(plot_file, width=400)
    
    if not results:
        st.info("No test results found. Run tests to see results.")

def show_comparison_tab():
    """Model comparison tab"""
    st.header("🏆 Model Comparison")
    
    # Collect results from both training and testing
    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']
    comparison_data = []
    
    for ds in datasets:
        model_file = f"train/best_model_{ds}.pth"
        train_plot = f"train/training_loss_{ds}.png"
        test_plot = f"test/simple_test_{ds}.png"
        
        row = {
            'Dataset': ds,
            'Model Trained': '✅' if os.path.exists(model_file) else '❌',
            'Training Plot': '✅' if os.path.exists(train_plot) else '❌', 
            'Test Results': '✅' if os.path.exists(test_plot) else '❌',
        }
        comparison_data.append(row)
    
    # Display comparison table
    st.subheader("📊 Status Overview")
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, width=800)
    
    # Show plots side by side
    st.subheader("📈 Training Loss Comparison")
    
    cols = st.columns(2)
    plot_idx = 0
    
    for ds in datasets:
        train_plot = f"train/training_loss_{ds}.png"
        if os.path.exists(train_plot):
            with cols[plot_idx % 2]:
                st.image(train_plot, caption=f"{ds} Training Loss")
                plot_idx += 1
    
    st.subheader("🧪 Test Results Comparison") 
    
    cols = st.columns(2)
    plot_idx = 0
    
    for ds in datasets:
        test_plot = f"test/simple_test_{ds}.png"
        if os.path.exists(test_plot):
            with cols[plot_idx % 2]:
                st.image(test_plot, caption=f"{ds} Test Results")
                plot_idx += 1
    
    # Summary
    trained_count = sum(1 for row in comparison_data if row['Model Trained'] == '✅')
    tested_count = sum(1 for row in comparison_data if row['Test Results'] == '✅')
    
    st.subheader("📋 Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Datasets", len(datasets))
    with col2: 
        st.metric("Models Trained", trained_count)
    with col3:
        st.metric("Tests Completed", tested_count)

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
