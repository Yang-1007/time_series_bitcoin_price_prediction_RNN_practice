#!/usr/bin/env python
# coding: utf-8

# In[ ]:

"""
This demo script demonstrates inference only.
It does not reproduce training or exact evaluation metrics.
"""

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)


import numpy as np
import torch
import matplotlib.pyplot as plt

from src.model import BaselineLSTM

from src.utils import inverse_scale_sequence
from sklearn.preprocessing import MinMaxScaler


def load_sample_input(path="demo/sample.npy"):
    """
    Loads one 30-day sample window.
    Shape must be (seq_len, num_features).
    """
    sample = np.load(path).astype(np.float32)
    return sample


def main():
    # -----------------------------
    # Setup
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = "checkpoints/baseline_lstm_single.pth"
    sample_path = "demo/sample.npy"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}.\n"
            "Please download the pre-trained model and place it in the checkpoints/ directory."
        )

    if not os.path.exists(sample_path):
        raise FileNotFoundError(
            "Sample input not found at demo/sample.npy.\n"
            "Please place a sample window into the demo/ directory."
        )

    # -----------------------------
    # Load sample input
    # -----------------------------
    sample = load_sample_input(sample_path)  # (30, num_features)
    seq_len, num_features = sample.shape

    print("Loaded sample window (first 5 timesteps):")
    print(sample[:5])

    # -----------------------------
    # Demo scaler (for visualization only)
    # -----------------------------
    # NOTE: This scaler is fitted only for demo purposes.
    # It is not intended to reproduce the training-time normalization.
    scaler = MinMaxScaler()
    scaler.fit(sample)
    scaled_sample = scaler.transform(sample)

    # -----------------------------
    # Model definition (must match training)
    # -----------------------------
    prediction_length = 7
    hidden_dim = 64

    model = BaselineLSTM(
        number_of_input_features=num_features,
        hidden_dimension=hidden_dim,
        prediction_length=prediction_length,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded pre-trained model.")
    model.eval()

    # -----------------------------
    # Run inference
    # -----------------------------
    with torch.no_grad():
        input_tensor = torch.tensor(scaled_sample).unsqueeze(0).to(device)
        prediction_scaled = model(input_tensor).cpu().numpy().squeeze()

    # -----------------------------
    # Inverse scaling
    # -----------------------------
    prediction_actual = inverse_scale_sequence(
        scaled_sequence=prediction_scaled,
        scaler=scaler,
        num_features=num_features,
    )

    print("\nPredicted next 7 days:")
    print(prediction_actual[:, 0])  # print Close price

    # -----------------------------
    # Save output plot
    # -----------------------------
    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(prediction_actual[:, 0], marker="o", label="Predicted Price")
    plt.title("7-Day Bitcoin Price Forecast (Demo)")
    plt.xlabel("Day Ahead")
    plt.ylabel("BTC Price")
    plt.grid(True)
    plt.legend()

    output_path = "results/prediction_plot.png"
    plt.savefig(output_path)
    plt.close()

    print(f"\nSaved plot to: {output_path}")


if __name__ == "__main__":
    main()


