"""
Standalone inference + visualization script for the Fish Trajectory
Forecasting model (FishSeq2Seq), using the trained weights from:
Faruk0713/Fish-Trajectory-Forecast-for-Obstacle-Avoidance-System-in-AUV

HOW TO RUN
----------
1. Clone or download the repo, or just make sure you have these two files
   locally and update the paths below:
       models/fish_seq2seq_triple_split.pth
       Demo_Training_Data/B1_3_final_flip.csv   (or any of your own trajectory CSVs)

2. Install dependencies:
       pip install torch pandas numpy matplotlib

3. Run:
       python run_inference.py

4. Adjust START_IDX below to look at a different window of the trajectory,
   or point CSV_PATH at any of your own standardized trajectory CSVs.

This script:
  - Loads the trained Seq2Seq model
  - Picks a 20-frame input window from your CSV
  - Detects its direction (L->R or R->L) automatically
  - Mirrors the input if needed, runs the model, and mirrors the output back
  - Plots Past / Actual / Forecast and saves it as a PNG
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==================================================================
# CONFIG -- EDIT THESE PATHS / SETTINGS
# ==================================================================
MODEL_PATH = "models/fish_seq2seq_triple_split.pth"
CSV_PATH   = "Demo_Training_Data/B1_3_final_flip.csv"
OUTPUT_DIR = "inference_results"   # each example saved as inference_results/example_{i}.png

IN_W, OUT_W, HIDDEN_DIM = 20, 50, 128   # must match training config
X_MIN, X_MAX = 0.0, 100.0               # standardization range used at training time
N_SAMPLES = 100                         # frames used per example (20 input + 50 actual + margin)

# List of starting indices to test. Each becomes one example/figure.
# Add or remove values here to try more windows across the trajectory.
START_INDICES = [0, 10, 20, 30, 40]

TEST_REVERSED = True  # also run the mirrored-direction sanity check for each window


# ==================================================================
# 1. MODEL DEFINITION (must match the trained checkpoint exactly)
# ==================================================================
class FishSeq2Seq(nn.Module):
    def __init__(self, hidden_dim, out_w):
        super().__init__()
        self.out_w = out_w
        self.encoder = nn.LSTM(2, hidden_dim, batch_first=True)
        self.decoder_cell = nn.LSTMCell(2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 2)

    def forward(self, x, target=None, tf_ratio=0.0):
        batch_size = x.size(0)
        _, (hidden, cell) = self.encoder(x)
        hidden, cell = hidden.squeeze(0), cell.squeeze(0)
        curr_input = x[:, -1, :]
        outputs = torch.zeros(batch_size, self.out_w, 2).to(x.device)
        for t in range(self.out_w):
            hidden, cell = self.decoder_cell(curr_input, (hidden, cell))
            pred = self.fc_out(hidden)
            outputs[:, t, :] = pred
            if self.training and target is not None and torch.rand(1).item() < tf_ratio:
                curr_input = target[:, t, :]
            else:
                curr_input = pred
        return outputs


# ==================================================================
# 2. DIRECTION-AWARE INFERENCE HELPERS
# ==================================================================
def detect_direction(window_xy: np.ndarray) -> str:
    """Fits a line through the x-values of the input window and checks its slope."""
    x = window_xy[:, 0]
    t = np.arange(len(x))
    slope = np.polyfit(t, x, 1)[0]
    return "L->R" if slope >= 0 else "R->L"

def mirror_x(arr: np.ndarray, x_min=X_MIN, x_max=X_MAX) -> np.ndarray:
    """Reflects the x-coordinate around the midpoint of [x_min, x_max]. Y untouched."""
    mirrored = arr.copy()
    mirrored[..., 0] = (x_min + x_max) - arr[..., 0]
    return mirrored

def predict_trajectory(model, window_xy: np.ndarray, device="cpu") -> dict:
    """
    Direction-aware single-window inference:
      - Detects direction of the 20-frame input
      - If R->L: mirrors input, runs model, mirrors output back
      - If L->R: runs model directly
    """
    model.eval()
    direction = detect_direction(window_xy)
    flipped = direction == "R->L"
    model_input = mirror_x(window_xy) if flipped else window_xy

    x_tensor = torch.from_numpy(model_input.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x_tensor, tf_ratio=0.0)
    pred_np = pred.squeeze(0).cpu().numpy()

    if flipped:
        pred_np = mirror_x(pred_np)

    return {"direction": direction, "flipped": flipped, "prediction": pred_np}


# ==================================================================
# 3. LOAD MODEL + DATA
# ==================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FishSeq2Seq(HIDDEN_DIM, OUT_W).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Loaded model on {DEVICE}")

import os

df = pd.read_csv(CSV_PATH)
coords = df[["x_standardized", "y_standardized"]].values.astype(np.float32)
print(f"✅ Loaded trajectory: {len(coords)} frames from {CSV_PATH}")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_one_example(start_idx: int, example_idx: int):
    if len(coords) < start_idx + N_SAMPLES:
        print(f"⚠️ Skipping start_idx={start_idx}: not enough frames "
              f"(need {N_SAMPLES}, got {len(coords) - start_idx})")
        return None

    samples = coords[start_idx : start_idx + N_SAMPLES]

    # --- CASE 1: Native window ---
    input1  = samples[0:20]
    actual1 = samples[20:70]
    result1 = predict_trajectory(model, input1, device=DEVICE)

    print(f"\n=== Example {example_idx} (start_idx={start_idx}) ===")
    print(f"Case 1 -- direction: {result1['direction']} | flip: {result1['flipped']}")

    # --- CASE 2: Same input, spatially mirrored ---
    result2, raw_match = None, None
    if TEST_REVERSED:
        input2  = mirror_x(input1)
        actual2 = mirror_x(actual1)
        result2 = predict_trajectory(model, input2, device=DEVICE)
        raw_match = np.allclose(mirror_x(result2["prediction"]), result1["prediction"], atol=1e-4)
        print(f"Case 2 -- direction: {result2['direction']} | flip: {result2['flipped']} "
              f"| raw output matches Case 1: {raw_match}")

    # --- PLOT ---
    n_panels = 2 if TEST_REVERSED else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(input1[:, 0], input1[:, 1], "o--", color="gray", label="Past (input, 20 steps)")
    ax.plot(actual1[:, 0], actual1[:, 1], "-", color="green", label="Actual future")
    ax.plot(result1["prediction"][:, 0], result1["prediction"][:, 1], "--", color="red", label="Forecast")
    ax.set_title(f"Example {example_idx} | Case 1: Native (dir={result1['direction']})")
    ax.set_xlabel("x_standardized")
    ax.set_ylabel("y_standardized")
    ax.legend()
    ax.grid(alpha=0.3)

    if TEST_REVERSED:
        ax = axes[1]
        ax.plot(input2[:, 0], input2[:, 1], "o--", color="gray", label="Past (input, mirrored)")
        ax.plot(actual2[:, 0], actual2[:, 1], "-", color="green", label="Actual future (mirrored)")
        ax.plot(result2["prediction"][:, 0], result2["prediction"][:, 1], "--", color="red", label="Forecast (mirrored back)")
        ax.set_title(f"Example {example_idx} | Case 2: Mirrored (dir={result2['direction']})")
        ax.set_xlabel("x_standardized")
        ax.set_ylabel("y_standardized")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"example_{example_idx}_start{start_idx}.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"✅ Saved: {out_path}")

    return {
        "start_idx": start_idx,
        "direction1": result1["direction"],
        "direction2": result2["direction"] if result2 else None,
        "raw_match": raw_match,
        "mse_case1": float(np.mean((result1["prediction"] - actual1) ** 2)),
    }


# ==================================================================
# RUN ALL EXAMPLES
# ==================================================================
summary_rows = []
for i, start_idx in enumerate(START_INDICES):
    row = run_one_example(start_idx, i)
    if row is not None:
        summary_rows.append(row)

print("\n" + "=" * 60)
print("SUMMARY ACROSS ALL EXAMPLES")
print("=" * 60)
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
print(f"\n✅ Summary saved to {os.path.join(OUTPUT_DIR, 'summary.csv')}")