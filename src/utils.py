#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import random
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


# -----------------------------
# Reproducibility
# -----------------------------

def set_global_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Data loading / preprocessing
# -----------------------------

def load_raw_csv(csv_path: str) -> pd.DataFrame:
    """
    Load the raw BTC CSV file.
    """
    return pd.read_csv(csv_path)


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert minute-level BTC data into daily OHLCV candles.
    """
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
    df = df.sort_values(by="Timestamp")
    df = df[df["Close"] > 0]  # remove invalid rows
    df = df.set_index("Timestamp")

    df_daily = df.resample("D").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    })

    return df_daily.dropna()


def select_features(df_daily: pd.DataFrame, use_multifeature: bool):
    """
    Select either:
    - single feature: Close
    - multi features: OHLCV
    """
    if use_multifeature:
        feature_columns = ["Open", "High", "Low", "Close", "Volume"]
    else:
        feature_columns = ["Close"]

    data = df_daily[feature_columns].copy()
    return data, feature_columns


def split_train_val_test(
    data: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    """
    Time-ordered split without shuffling.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def scale_splits(train_data, val_data, test_data):
    """
    Fit MinMaxScaler on training data only, then transform all splits.

    Returns:
        scaler, train_scaled, val_scaled, test_scaled
    """
    scaler = MinMaxScaler()
    scaler.fit(train_data)

    train_scaled = scaler.transform(train_data).astype(np.float32)
    val_scaled = scaler.transform(val_data).astype(np.float32)
    test_scaled = scaler.transform(test_data).astype(np.float32)

    return scaler, train_scaled, val_scaled, test_scaled


# -----------------------------
# Sliding window dataset
# -----------------------------

class SequenceDataset(Dataset):
    """
    Convert a time-series array into (X, Y) sliding windows.

    series shape: (num_timesteps, num_features)
    X shape:      (seq_len, num_features)
    Y shape:      (pred_len, num_features)
    """

    def __init__(self, series: np.ndarray, seq_len: int, pred_len: int):
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.X = []
        self.Y = []

        # +1 ensures the last valid window is included
        for i in range(len(series) - seq_len - pred_len + 1):
            self.X.append(series[i : i + seq_len])
            self.Y.append(series[i + seq_len : i + seq_len + pred_len])

        self.X = np.array(self.X, dtype=np.float32)
        self.Y = np.array(self.Y, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.Y[idx], dtype=torch.float32),
        )


# -----------------------------
# Training / evaluation helpers
# -----------------------------

def run_one_epoch(
    model: torch.nn.Module,
    data_loader,
    loss_function,
    optimizer,
    device,
    is_training: bool,
    is_seq2seq_model: bool = False,
    use_teacher_forcing: bool = False,
):
    """
    Run one training or evaluation epoch.
    """
    model.train() if is_training else model.eval()

    total_loss = 0.0
    num_batches = 0

    context = torch.enable_grad if is_training else torch.no_grad

    with context():
        for input_sequence, target_sequence in data_loader:
            input_sequence = input_sequence.to(device)
            target_sequence = target_sequence.to(device)

            if is_training:
                optimizer.zero_grad()

            if is_seq2seq_model:
                prediction = model(
                    input_sequence,
                    target_sequence=target_sequence if use_teacher_forcing else None,
                )
            else:
                prediction = model(input_sequence)

            loss = loss_function(prediction, target_sequence)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(1, num_batches)


# Mean Squared Error is used for regression consistency across models
def train_model(
    model: torch.nn.Module,
    train_loader,
    validation_loader,
    number_of_epochs: int,
    learning_rate: float,
    device,
    is_seq2seq_model: bool = False,
    teacher_forcing_enabled: bool = True,
    early_stopping_patience: int = 5,
):
    """
    Unified training loop with early stopping.
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, number_of_epochs + 1):
        train_loss = run_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            is_training=True,
            is_seq2seq_model=is_seq2seq_model,
            use_teacher_forcing=(teacher_forcing_enabled and is_seq2seq_model),
        )

        val_loss = run_one_epoch(
            model,
            validation_loader,
            criterion,
            optimizer,
            device,
            is_training=False,
            is_seq2seq_model=is_seq2seq_model,
            use_teacher_forcing=False,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Stopping early at epoch {epoch} "
                f"due to no improvement in validation loss."
            )
            break

    return model, None, None


def evaluate_on_test_set(
    model: torch.nn.Module,
    test_loader,
    device,
    is_seq2seq_model: bool = False,
):
    """
    Compute average MSE and MAE on the test set.
    """
    model.to(device)
    model.eval()

    mse_values = []
    mae_values = []

    mse_loss = nn.MSELoss(reduction="none")
    mae_loss = nn.L1Loss(reduction="none")

    with torch.no_grad():
        for input_sequence, target_sequence in test_loader:
            input_sequence = input_sequence.to(device)
            target_sequence = target_sequence.to(device)

            if is_seq2seq_model:
                prediction = model(input_sequence, target_sequence=None)
            else:
                prediction = model(input_sequence)

            mse_values.append(mse_loss(prediction, target_sequence).mean().item())
            mae_values.append(mae_loss(prediction, target_sequence).mean().item())

    return float(np.mean(mse_values)), float(np.mean(mae_values))


def inverse_scale_sequence(
    scaled_sequence: np.ndarray,
    scaler: MinMaxScaler,
    num_features: int,
) -> np.ndarray:
    """
    Inverse MinMax scaling for predicted sequences.
    """
    scaled_sequence = scaled_sequence.reshape(-1, num_features)
    return scaler.inverse_transform(scaled_sequence)


def plot_test_forecast(
    model: torch.nn.Module,
    test_dataset,
    scaler: MinMaxScaler,
    device,
    feature_columns,
    model_name: str,
    sample_index: int | None = None,
    is_seq2seq_model: bool = False,
):
    """
    Plot prediction vs ground truth for one test sample
    and save the figure to results/ directory.
    """
    import os

    model.to(device) # Move model to evaluation mode and correct device
    model.eval()

    num_features = len(feature_columns)

    if sample_index is None:
        sample_index = np.random.randint(0, len(test_dataset))

    input_sequence, target_sequence = test_dataset[sample_index]
    input_sequence = input_sequence.unsqueeze(0).to(device)

    with torch.no_grad():
        if is_seq2seq_model:
            prediction = model(input_sequence, target_sequence=None)
        else:
            prediction = model(input_sequence)

    predicted_actual = inverse_scale_sequence(
        prediction.squeeze().cpu().numpy(),
        scaler,
        num_features=num_features,
    )
    true_actual = inverse_scale_sequence(
        target_sequence.cpu().numpy(),
        scaler,
        num_features=num_features,
    )

    close_index = (
        feature_columns.index("Close") if "Close" in feature_columns else 0
    )

    # -----------------------------
    # Save plot
    # -----------------------------
    os.makedirs("results", exist_ok=True)
    output_path = f"results/test_forecast_{model_name}.png"

    plt.figure(figsize=(9, 4))
    plt.plot(true_actual[:, close_index], label="True Future", marker="o")
    plt.plot(predicted_actual[:, close_index], label="Predicted Future", marker="x")
    plt.title(f"7-Day Forecast – {model_name}")
    plt.xlabel("Future Day")
    plt.ylabel("BTC Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()

    print(f"Saved {model_name} forecast plot to {output_path}")
