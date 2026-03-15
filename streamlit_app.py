import os
import sys
import pandas as pd
import subprocess
import streamlit as st
from pathlib import Path

# ======================================
# PATH SETUP
# ======================================

script_dir = Path(__file__).resolve().parent
project_root = str(script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

FIGURES_DIR = os.path.join(project_root, "figures")
MODELS_DIR = os.path.join(project_root, "saved_models")


# ======================================
# TRAIN TAB
# ======================================

def show_train_tab():

    st.header("🚀 TimesMamba Training")

    col1, col2 = st.columns(2)

    with col1:

        dataset = st.selectbox(
            "Select Dataset",
            ["weather", "ETTh1", "ETTh2", "ETTm1", "ETTm2"],
            key="train_dataset"
        )

        epochs = st.number_input(
            "Epochs",
            min_value=1,
            max_value=100,
            value=10,
            key="train_epochs"
        )

    with col2:

        st.subheader("Dataset Info")

        candidates = [
            os.path.join(project_root, "dataset", dataset, f"{dataset}.csv"),
            os.path.join(project_root, "dataset", f"{dataset}.csv"),
            os.path.join(project_root, "train", "datasets", dataset, f"{dataset}_train.csv"),
        ]

        data_file = None

        for p in candidates:
            if os.path.exists(p):
                data_file = p
                break

        if data_file:

            df = pd.read_csv(data_file)

            st.write("Rows:", len(df))
            st.write("Columns:", list(df.columns))
            st.dataframe(df.head(3))

        else:

            st.warning("Dataset not found")

    if st.button("🚀 Start Training", key="train_button"):

        train_script = os.path.join(project_root, "train", "train.py")

        cmd = [
            sys.executable,
            train_script,
            "--dataset",
            dataset
        ]

        with st.spinner("Training model..."):

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=project_root
            )

            if result.returncode == 0:

                st.success("Training completed")
                st.text(result.stdout)

            else:

                st.error("Training failed")
                st.text(result.stderr)

    st.subheader("Saved Models")

    if os.path.exists(MODELS_DIR):

        files = os.listdir(MODELS_DIR)

        if files:

            for f in files:
                if f.endswith(".pth"):
                    st.write("✅", f)

        else:
            st.info("No models found")


# ======================================
# TEST TAB
# ======================================

def show_test_tab():

    st.header("🧪 TimesMamba Testing")

    dataset = st.selectbox(
        "Select Dataset",
        ["weather", "ETTh1", "ETTh2", "ETTm1", "ETTm2"],
        key="test_dataset"
    )

    model_file = f"best_model_TimesMamba_{dataset}.pth"
    model_path = os.path.join(MODELS_DIR, model_file)

    if os.path.exists(model_path):

        st.success("Model found")

    else:

        st.warning("Model not found")

    candidates = [
        os.path.join(project_root, "dataset", dataset, f"{dataset}.csv"),
        os.path.join(project_root, "dataset", f"{dataset}.csv"),
        os.path.join(project_root, "test", "datasets", dataset, f"{dataset}_test.csv")
    ]

    data_file = None

    for p in candidates:
        if os.path.exists(p):
            data_file = p
            break

    if data_file:

        df = pd.read_csv(data_file)

        st.write("Rows:", len(df))
        st.write("Columns:", list(df.columns))
        st.dataframe(df.head(3))

    else:

        st.error("Test dataset not found")

    if st.button("🧪 Run Test", key="test_button"):

        test_script = os.path.join(project_root, "test", "simple_test.py")

        cmd = [
            sys.executable,
            test_script,
            "--dataset",
            dataset
        ]

        with st.spinner("Running test..."):

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=project_root
            )

            if result.returncode == 0:

                st.success("Testing completed")
                st.text(result.stdout)

            else:

                st.error("Testing failed")
                st.text(result.stderr)

    st.subheader("Forecast Plot")

    if os.path.exists(FIGURES_DIR):

        for f in os.listdir(FIGURES_DIR):

            if "forecast" in f:

                st.image(os.path.join(FIGURES_DIR, f))


# ======================================
# COMPARISON TAB
# ======================================

def show_comparison_tab():

    st.header("🏆 Model Comparison")

    st.markdown(
        """
Example benchmark table for your thesis:

| Model | Dataset | MSE | MAE |
|------|------|------|------|
| TimesMamba | weather | 0.316 | 0.208 |
| RNN (LSTM) | weather | 0.42 | 0.30 |
| ITransformer | weather | 0.34 | 0.23 |
"""
    )

    st.info(
        """
You can extend this dashboard to compare:

• TimesMamba  
• RNN (LSTM)  
• ITransformer  

using the same datasets.
"""
    )


# ======================================
# MAIN
# ======================================

def main():

    st.set_page_config(
        page_title="TimesMamba Dashboard",
        page_icon="⚡",
        layout="wide"
    )

    st.title("⚡ TimesMamba Forecasting Dashboard")

    tab1, tab2, tab3 = st.tabs(
        ["🚀 Training", "🧪 Testing", "🏆 Comparison"]
    )

    with tab1:
        show_train_tab()

    with tab2:
        show_test_tab()

    with tab3:
        show_comparison_tab()

    st.markdown("---")
    st.caption("TimesMamba Time Series Forecasting Dashboard")


if __name__ == "__main__":
    main()