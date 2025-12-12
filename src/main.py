#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import torch
from torch.utils.data import DataLoader

from src.config import (
    SEQ_LEN,
    PRED_LEN,
    FEATURE_MODE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    LSTM_HIDDEN_DIM,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_HEADS,
    TRANSFORMER_LAYERS,
    SEED,
)

from src.model import (
    BaselineLSTM,
    Seq2SeqLSTM,
    TransformerPredictor,
    TransformerWithPositionalEncoding,
)

from src.utils import (
    set_global_seed,
    load_raw_csv,
    resample_to_daily,
    select_features,
    split_train_val_test,
    scale_splits,
    SequenceDataset,
    train_model,
    evaluate_on_test_set,
    plot_test_forecast,
)



def main():

    # -----------------------------
    # Environment / device / seed
    # -----------------------------
    set_global_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # Dataset path check
    # -----------------------------
    DATA_PATH = os.path.join("data", "btcusd_1-min_data.csv")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Please place btcusd_1-min_data.csv in the data/ directory."
        )
    
    # -----------------------------
    # Load and preprocess data
    # -----------------------------
    print("Loading raw data...")
    df_raw = load_raw_csv(DATA_PATH)

    print("Resampling to daily candles...")
    df_daily = resample_to_daily(df_raw)

    # FEATURE_MODE determines which features are used:
    # "single" -> Close price only
    # "multi"  -> OHLCV
    use_multifeature = (FEATURE_MODE == "multi")
    data, feature_columns = select_features(
        df_daily, use_multifeature=use_multifeature
    )

    print("Daily data shape:", data.shape)

    train_data, val_data, test_data = split_train_val_test(data)
    print(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")

    scaler, train_scaled, val_scaled, test_scaled = scale_splits(
        train_data, val_data, test_data
    )

    # -----------------------------
    # Build datasets / loaders
    # -----------------------------
    train_ds = SequenceDataset(train_scaled, SEQ_LEN, PRED_LEN)
    val_ds   = SequenceDataset(val_scaled, SEQ_LEN, PRED_LEN)
    test_ds  = SequenceDataset(test_scaled, SEQ_LEN, PRED_LEN)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False
    )

    num_features = len(feature_columns)
    print(f"Number of features: {num_features}")

    # -----------------------------
    # Instantiate models
    # -----------------------------
    baseline_model = BaselineLSTM(
        number_of_input_features=num_features,
        hidden_dimension=LSTM_HIDDEN_DIM,
        prediction_length=PRED_LEN,
    ).to(device)

    seq2seq_model = Seq2SeqLSTM(
        number_of_input_features=num_features,
        hidden_dimension=LSTM_HIDDEN_DIM,
        prediction_length=PRED_LEN,
    ).to(device)

    if use_multifeature:
        transformer_model = TransformerWithPositionalEncoding(
            number_of_input_features=num_features,
            prediction_length=PRED_LEN,
            transformer_embedding_dimension=TRANSFORMER_D_MODEL,
            number_of_attention_heads=TRANSFORMER_HEADS,
            number_of_encoder_layers=TRANSFORMER_LAYERS,
        ).to(device)
    else:
        transformer_model = TransformerPredictor(
            number_of_input_features=num_features,
            prediction_length=PRED_LEN,
            transformer_embedding_dimension=TRANSFORMER_D_MODEL,
            number_of_attention_heads=TRANSFORMER_HEADS,
            number_of_encoder_layers=TRANSFORMER_LAYERS,
        ).to(device)

    # -----------------------------
    # Train models
    # -----------------------------
    print(f"\n=== Training Baseline LSTM with {FEATURE_MODE} feature ===")
    baseline_model, _, _ = train_model(
        baseline_model,
        train_loader,
        val_loader,
        number_of_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device,
        is_seq2seq_model=False,
        teacher_forcing_enabled=False,
        early_stopping_patience=5,
    )

    print(f"\n=== Training Seq2Seq LSTM with {FEATURE_MODE} feature ===")
    seq2seq_model, _, _ = train_model(
        seq2seq_model,
        train_loader,
        val_loader,
        number_of_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device,
        is_seq2seq_model=True,
        teacher_forcing_enabled=True,
        early_stopping_patience=5,
    )

    print(f"\n=== Training Transformer with {FEATURE_MODE} feature ===")
    transformer_model, _, _ = train_model(
        transformer_model,
        train_loader,
        val_loader,
        number_of_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device,
        is_seq2seq_model=False,
        teacher_forcing_enabled=False,
        early_stopping_patience=5,
    )

    # -----------------------------
    # Evaluate on test set
    # -----------------------------
    print("\n=== Test Evaluation ===")

    baseline_test_mse, baseline_test_mae = evaluate_on_test_set(
        baseline_model, test_loader, device, is_seq2seq_model=False
    )
    print(f"Baseline Test MSE: {baseline_test_mse}")
    print(f"Baseline Test MAE: {baseline_test_mae}")

    seq2seq_test_mse, seq2seq_test_mae = evaluate_on_test_set(
        seq2seq_model, test_loader, device, is_seq2seq_model=True
    )
    print(f"Seq2Seq Test MSE: {seq2seq_test_mse}")
    print(f"Seq2Seq Test MAE: {seq2seq_test_mae}")

    transformer_test_mse, transformer_test_mae = evaluate_on_test_set(
        transformer_model, test_loader, device, is_seq2seq_model=False
    )
    print(f"Transformer Test MSE: {transformer_test_mse}")
    print(f"Transformer Test MAE: {transformer_test_mae}")

    # -----------------------------
    # Save models
    # -----------------------------
    os.makedirs("checkpoints", exist_ok=True)

    torch.save(
        baseline_model.state_dict(),
        f"checkpoints/baseline_lstm_{FEATURE_MODE}.pth",
    )
    torch.save(
        seq2seq_model.state_dict(),
        f"checkpoints/seq2seq_lstm_{FEATURE_MODE}.pth",
    )
    torch.save(
        transformer_model.state_dict(),
        f"checkpoints/transformer_{FEATURE_MODE}.pth",
    )

    print("\nSaved trained models to checkpoints/ directory.")

    # -----------------------------
    # forecast plot
    # -----------------------------
    print("\nSaving example forecast plots...")

    plot_test_forecast(
        model=baseline_model,
        test_dataset=test_ds,
        scaler=scaler,
        device=device,
        feature_columns=feature_columns,
        model_name="baseline_lstm",
        is_seq2seq_model=False,
    )
    
    plot_test_forecast(
        model=seq2seq_model,
        test_dataset=test_ds,
        scaler=scaler,
        device=device,
        feature_columns=feature_columns,
        model_name="seq2seq_lstm",
        is_seq2seq_model=True,
    )
    
    plot_test_forecast(
        model=transformer_model,
        test_dataset=test_ds,
        scaler=scaler,
        device=device,
        feature_columns=feature_columns,
        model_name="transformer",
        is_seq2seq_model=False,
    )


if __name__ == "__main__":
    main()

