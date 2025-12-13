# Multistep Time-Series Forecasting for Bitcoin Prices
This project explores short-term financial forecasting using deep learning sequence models.  
Given the previous **30 days** of Bitcoin price data, the goal is to predict the next **7 days** using three architectures:

1. **Baseline LSTM**
2. **Seq2Seq Encoder–Decoder**
3. **Transformer (Encoder-only) with/without Positional Encoding**

The models are evaluated on both:
- **Single-feature forecasting** (Close price only)
- **Multivariate forecasting** (OHLCV features)

---

## Project Overview
This project was developed for the final course project in Deep Learning.  
The objectives are:

- Learn and practice RNN-based and attention-based sequence models  
- Compare forecasting accuracy across architectures  
- Understand the impact of architectural choices on short-horizon prediction  
- Train models using consistent settings for fair comparison  

The dataset used is **Bitcoin historical OHLCV price data** from Kaggle:
https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── environment.yml
├── checkpoints/        # Placeholder for saved model weights
├── demo/               # Demo script and sample input
│   ├── demo.py
│   └── sample.npy
├── results/            # Placeholder for generated results/plots
├── ipynb_files/        # original notebooks
└── src/
    ├── main.py         # Training and evaluation entry point
    ├── model.py        # Model definitions
    ├── utils.py        # Data processing and training utilities
    └── config.py       # Hyperparameters and experiment settings

```
---

## Setup Instructions

### Platform Notes (Important)

- This project has been tested on **Windows**.
- On **Windows**, I encountered PyTorch DLL loading issues when using Conda.
  If this happens to you, please try use the **CPU-only PyTorch installation** as described below.
- On **macOS and Linux**, the provided `environment.yml` should works out of the box.

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Yang-1007/time_series_bitcoin_price_prediction_RNN_practice.git
cd <your-cloned-repo> (e.g. time_series_bitcoin_price_prediction_RNN_practice)
```

### Step 2 — Create and Activate an Environment 
If you do not already have an environment set up or you want to create a new one.

Option A — Using Conda 
```bash
conda env create -f environment.yml
conda activate btc_forecasting_env
```

Option B — Using pip + virtual environment
```bash
python3 -m venv venv
source venv/bin/activate     # Mac / Linux
venv\Scripts\activate        # Windows
```

#### Windows Users (If PyTorch Import Error Occurs)

If you see an error related to `fbgemm.dll` or PyTorch failing to import:

1. Remove existing PyTorch:
```bash
pip uninstall torch -y
```
2. Install CPU-only PyTorch via pip:
```
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
```
Then re-run the demo or training script.

### Step 3 — Install Dependencies (Optional)
Note: `requirements.txt` is provided as a fallback for pip users or Windows users
who encounter Conda-related PyTorch issues.

```bash
pip install -r requirements.txt
```
  
---

## How to Run the Demo
A minimal demo script is provided that loads a pretrained model, runs inference on a sample input, and saves the resulting forecast plot.
Before running the demo script, download the pretrained models (link below) and place them into the `checkpoints/` directory. 

Run the demo from the project root:
```bash
python demo/demo.py
```

This demo script will:
1. Load a pretrained model
2. Load a sample 30-day Bitcoin window from demo/sample.npy
3. Predict the next 7 days
4. Save a forecast plot into results/prediction_plot.png

---

## Expected Output

After running the demo, you should see:

- Printed predicted values for the next 7 days in the terminal

- A saved plot comparing predicted prices over the forecast horizon:
    `results/prediction_plot.png`
  
---

## Pre-trained Models

The pretrained model weights are available at:

🔗 Google Drive link:
[[https://drive.google.com/xxxxxx](https://drive.google.com/drive/u/0/folders/1bW-P4a6qlVLmGqeyJh-wcdsnwQA2UE6L)]

Included models:
```
- baseline_lstm_single.pth
- baseline_lstm_multi.pth

- seq2seq_lstm_single.pth
- seq2seq_lstm_multi.pth

- transformer_single.pth
- transformer_posenc_multi.pth
```

---

## Training Your Own Models

All training settings are defined in src/config.py.
- To train single-feature models:
  `FEATURE_MODE = "single"`

- To train multivariate (OHLCV) models:
  `FEATURE_MODE = "multi"`

Then run:
```
python -m src.main
```

---

## Jupyter Notebook Files

For reference and transparency, the original development notebooks used during the project are also included.

- The folder ipynb_files/ contains:

    `final_project_single_feature_prediction.ipynb`

    `final_project_multi_feature_prediction.ipynb`
