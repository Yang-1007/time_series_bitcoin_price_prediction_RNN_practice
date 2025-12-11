# Multistep Time-Series Forecasting for Bitcoin Prices
This project explores short-term financial forecasting using deep learning sequence models.  
Given the previous 30 days of Bitcoin price data, the goal is to predict the next 7 days using three architectures:

1. **Baseline LSTM**
2. **Seq2Seq Encoder–Decoder**
3. **Transformer (Encoder-only) with/without Positional Encoding**

The project compares these architectures on both **single-feature** (Close price) and **multivariate** (OHLCV) prediction tasks.

---

## Project Overview
This project was developed for the final course project in Deep Learning.  
The objectives are:

- Learn and practice RNN-based and attention-based sequence models  
- Compare forecasting accuracy across architectures  
- Understand the impact of architectural choices on short-horizon prediction  
- Train models using consistent settings for fair comparison  

The dataset used is Bitcoin historical OHLCV price data from **Kaggle (https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)**.

---

## Setup Instructions (Required Section)

### Step 1 — Clone the Repository

### Step 2 — Create and Activate an Environment
Option A — Using Conda (Recommended)
conda env create -f environment.yml
conda activate btc-forecasting

Option B — Using pip + virtual environment
python3 -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows

### Step 3 — Install Dependencies
pip install -r requirements.txt

---

## How to Run the Demo
A simple demonstration script is provided to verify that the environment and pretrained model work correctly.
Run: python demo.py
This script will:

Load a pretrained model

Load a sample 30-day Bitcoin window from demo/sample.npy

Predict the next 7 days

Save a forecast plot into results/prediction_plot.png

---
