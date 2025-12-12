#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
All experiments in this project were conducted using a fixed and shared training configuration
to ensure fair comparison across models.

The hyperparameters below were arbitrary selected and are not optimal for any individual model.
"""

# ===============================
# Data configuration
# ===============================
SEQ_LEN = 30
PRED_LEN = 7
FEATURE_MODE = "single"  # "single" = Close price only, "multi" = OHLCV

# ===============================
# Training configuration
# ===============================
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 3e-4
LOSS_FUNCTION = "MSE"
OPTIMIZER = "Adam"

# ===============================
# Model configuration
# ===============================
LSTM_HIDDEN_DIM = 64
TRANSFORMER_D_MODEL = 64
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 3

# ===============================
# Reproducibility
# ===============================
SEED = 42

