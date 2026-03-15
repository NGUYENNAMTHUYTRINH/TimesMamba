import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ===============================
# PATH SETUP
# ===============================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# load mock mamba if needed
try:
    import mamba_ssm_mock
    sys.modules['mamba_ssm'] = mamba_ssm_mock
    print("Mamba SSM mock module loaded")
except:
    pass

from model import TimesMamba


# ===============================
# CONFIG
# ===============================

DATASET = "weather"

DATA_PATH = os.path.join(
    project_root,
    "dataset",
    "weather",
    "weather.csv"
)

MODEL_PATH = r"D:\KLTN\TimesMamba\saved_models\best_model_TimesMamba_weather.pth"

SEQ_LEN = 96
PRED_LEN = 24


# ===============================
# LOAD DATA
# ===============================

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
data = df[numeric_cols].values.astype(np.float32)

print("Rows:", len(data))
print("Features:", [str(c).encode("ascii", "ignore").decode() for c in numeric_cols])


# ===============================
# NORMALIZATION
# ===============================

mean = np.mean(data, axis=0)
std = np.std(data, axis=0) + 1e-8

data = (data - mean) / std


# ===============================
# CREATE SEQUENCES
# ===============================

sequences = []
targets = []

for i in range(len(data) - SEQ_LEN - PRED_LEN + 1):

    seq = data[i:i + SEQ_LEN]
    target = data[i + SEQ_LEN:i + SEQ_LEN + PRED_LEN]

    sequences.append(seq)
    targets.append(target)

sequences = np.array(sequences)
targets = np.array(targets)

print("Test sequences:", len(sequences))


# ===============================
# MODEL ARGS
# ===============================

class Args:

    def __init__(self):

        self.data = DATASET

        self.enc_in = len(numeric_cols)
        self.dec_in = len(numeric_cols)
        self.c_out = len(numeric_cols)

        self.seq_len = SEQ_LEN
        self.pred_len = PRED_LEN
        self.label_len = PRED_LEN // 2

        self.d_model = 64
        self.d_ff = 128
        self.e_layers = 2

        self.dropout = 0.1
        self.features = 'M'

        self.use_norm = True
        self.use_mark = False
        self.channel_independence = False
        self.revin_affine = False

        self.ssm_expand = 0
        self.r_ff = 1

        self.embed = 'timeF'
        self.freq = 'h'


args = Args()


# ===============================
# LOAD MODEL
# ===============================

print("\nLoading model...")

model = TimesMamba.Model(args)

state_dict = torch.load(MODEL_PATH, map_location="cpu")

model.load_state_dict(state_dict)

model.eval()

print("Model loaded successfully")


# ===============================
# INFERENCE
# ===============================

predictions = []
actuals = []

print("\nRunning inference...")

with torch.no_grad():

    for i in range(len(sequences)):

        seq = torch.FloatTensor(sequences[i:i+1])

        target = targets[i]

        time_mark = torch.zeros(1, SEQ_LEN, len(numeric_cols))
        dec_mark = torch.zeros(1, PRED_LEN, len(numeric_cols))

        dec_inp = torch.zeros(1, PRED_LEN, len(numeric_cols))

        output = model(seq, time_mark, dec_inp, dec_mark)

        pred = output.cpu().numpy()[0]

        predictions.append(pred)
        actuals.append(target)


predictions = np.array(predictions)
actuals = np.array(actuals)


# ===============================
# METRICS
# ===============================

mse = mean_squared_error(actuals.flatten(), predictions.flatten())
mae = mean_absolute_error(actuals.flatten(), predictions.flatten())

print("\nRESULTS")
print("MSE:", mse)
print("MAE:", mae)


# ===============================
# SAVE CSV
# ===============================

print("\nSaving predictions...")

pred_flat = predictions.reshape(predictions.shape[0], -1)

columns = []
for feat in numeric_cols:
    for step in range(PRED_LEN):
        columns.append(f"{feat}_t{step}")

df_pred = pd.DataFrame(pred_flat, columns=columns)

df_pred.to_csv("predictions_weather.csv", index=False)

print("Saved predictions_weather.csv")


# ===============================
# PLOT
# ===============================

print("\nSaving plot...")

actual_seq = actuals[0,:,0]
pred_seq = predictions[0,:,0]

plt.figure(figsize=(12,6))

plt.plot(actual_seq,label="Actual")
plt.plot(pred_seq,label="Predicted")

plt.legend()
plt.title("TimesMamba Forecast")
plt.xlabel("Time")
plt.ylabel(numeric_cols[0])

plt.grid(True)

plt.savefig("forecast_plot.png",dpi=150)

print("Saved forecast_plot.png")