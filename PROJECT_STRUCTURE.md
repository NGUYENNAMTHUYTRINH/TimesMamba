# TimesMamba Project Structure Guide

## What is TimesMamba?

TimesMamba is a time series forecasting model that uses the **Mamba architecture** instead of traditional Transformers. It's designed for:

- **Long-term time series forecasting** 
- **Multivariate data** (multiple features)
- **Efficient processing** with linear complexity
- **Better context modeling** than RNNs/LSTMs

## New Organized Project Structure

```
TimesMamba/
├── train/                    # 🚀 Training workflow
│   ├── train.py             # Streamlined training script
│   ├── datasets/            # Training datasets (auto-generated)
│   │   ├── ETTh1/          # ETTh1 train/val splits
│   │   ├── ETTh2/          # ETTh2 train/val splits  
│   │   ├── ETTm1/          # ETTm1 train/val splits
│   │   └── ETTm2/          # ETTm2 train/val splits
│   ├── best_model_*.pth    # Saved trained models
│   └── training_loss_*.png # Training curves
├── test/                    # 🧪 Testing workflow  
│   ├── test.py             # Testing and evaluation script
│   ├── datasets/           # Test datasets (auto-generated)
│   │   ├── ETTh1/          # ETTh1 test data
│   │   ├── ETTh2/          # ETTh2 test data
│   │   ├── ETTm1/          # ETTm1 test data  
│   │   └── ETTm2/          # ETTm2 test data
│   └── test_results_*.png  # Test result plots
├── scripts/                 # 🔧 Utilities
│   └── setup_datasets.py   # Export/organize datasets
├── streamlit_app.py        # 📊 Interactive dashboard (NEW!)
├── experiments/            # Original experiment framework
├── data_provider/          # Data loading utilities
├── model/                  # TimesMamba model code
├── layers/                 # Model components
├── utils/                  # Helper functions
└── dataset/ETT-small/      # Original raw datasets
```

## Quick Start Guide

### 🚀 Method 1: Interactive Dashboard (Recommended)
```bash
streamlit run streamlit_app.py
```

The dashboard provides three tabs:
- **🚀 Training**: Select datasets, configure parameters, start training
- **🧪 Testing**: Run tests on trained models, view results  
- **🏆 Comparison**: Compare performance across all datasets

### 🔧 Method 2: Command Line

#### Training
```bash
cd train
python train.py
```

#### Testing
```bash
cd test  
python test.py
```

#### Setup Datasets
```bash
cd scripts
python setup_datasets.py
```

## Key Features of New Structure

### ✅ Organized Workflows
- **Clear Separation**: Train and test in separate folders
- **Auto Dataset Management**: Pre-split datasets in proper folders
- **Model Persistence**: Trained models saved automatically
- **Visual Results**: Automatic plotting of results

### 🚀 Training System (`train/train.py`)
- Trains models on all ETT datasets (ETTh1, ETTh2, ETTm1, ETTm2)
- Early stopping based on validation loss
- Automatic model saving with best checkpoints  
- Training loss visualization and saving

### 🧪 Testing System (`test/test.py`) 
- Loads trained models automatically
- Generates predictions on test data
- Calculates MSE and MAE metrics
- Creates comparison plots (actual vs predicted)

### 📊 Interactive Dashboard (`streamlit_app.py`)
- **Training Tab**: Start training on any dataset with custom parameters
- **Testing Tab**: Run tests and view results
- **Comparison Tab**: Side-by-side comparison of all results

## Environment Setup

**Python**: 3.11.15 in `TimesMamba` conda environment

**Key Packages**:
- PyTorch 2.10.0+cpu (CPU-only for compatibility)
- mamba-ssm (mocked for CPU compatibility)
- pandas, numpy, matplotlib, streamlit
- scikit-learn for evaluation metrics

## Datasets

Four ETT (Electricity Transforming Temperature) datasets included:

- **ETTh1**: Hourly data (17,420 points, 7 features)
- **ETTh2**: Hourly data (17,420 points, 7 features)  
- **ETTm1**: 15-minute data (69,680 points, 7 features)
- **ETTm2**: 15-minute data (69,680 points, 7 features)

**Features**: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT (oil temperature)

**Auto-Split**:
- **Train**: 70% of data for model training
- **Val**: 15% for validation during training  
- **Test**: 15% for final evaluation

## Model Configuration

### Architecture 
- **Input Length**: 96 time steps
- **Prediction Length**: 96 future steps  
- **Model Dimension**: 32 (lightweight for CPU)
- **Mamba Layers**: 1 layer (efficient training)
- **Features**: Multivariate (all 7 features)

### Benefits of Mamba vs Transformer
1. **Linear Complexity**: O(n) vs O(n²) for self-attention
2. **Long Context**: Handles very long sequences efficiently
3. **Selective Memory**: Learns what information to retain/forget  
4. **No Positional Encoding**: Built-in temporal understanding

## File Structure Details

### Training Files
- `train/datasets/ETTh1/ETTh1_train.csv` - Training data
- `train/datasets/ETTh1/ETTh1_val.csv` - Validation data  
- `train/best_model_ETTh1.pth` - Saved model checkpoint
- `train/training_loss_ETTh1.png` - Training curve

### Testing Files  
- `test/datasets/ETTh1/ETTh1_test.csv` - Test data
- `test/test_results_ETTh1.png` - Prediction vs actual plot

## Expected Performance

TimesMamba typically demonstrates:
- **Lower MSE/MAE** than baseline models on ETT datasets
- **Faster Training** than equivalent Transformer models  
- **Stable Convergence** with consistent validation improvement
- **Good Generalization** across different ETT dataset variants

## Legacy Files (Still Available)

- `run.py` - Original TimesMamba training script with full CLI
- `experiments/` - Original experiment framework
- `scripts/multivariate_forecasting/` - Shell scripts for batch training

These are maintained for compatibility but the new `train/test/` workflow is recommended for easier use.

## Usage Examples

### Train on Specific Dataset
```bash
# Using Streamlit (Easy)
streamlit run streamlit_app.py
# Go to Training tab, select ETTh1, click "Start Training"

# Using Command Line
cd train && python train.py  # Trains all datasets
```

### Test Trained Model
```bash  
# Using Streamlit (Easy)
streamlit run streamlit_app.py
# Go to Testing tab, select ETTh1, click "Run Test"

# Using Command Line
cd test && python test.py    # Tests all available models
```

### Compare All Results
```bash
# Using Streamlit 
streamlit run streamlit_app.py
# Go to Comparison tab for side-by-side view
```

This new structure provides a clean, organized approach to experimenting with TimesMamba while maintaining all the original functionality.