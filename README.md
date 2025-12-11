# Multistep Time-Series Forecasting for Bitcoin Prices
This project explores short-term financial forecasting using deep learning sequence models.  
Given the previous 30 days of Bitcoin price data, the goal is to predict the next 7 days using three architectures:

1. **Baseline LSTM**
2. **Seq2Seq Encoder–Decoder**
3. **Transformer (Encoder-only) with/without Positional Encoding**

The project compares these architectures on both **single-feature** (Close price) and **multivariate** (OHLCV) prediction tasks.

---

## 🚀 Project Overview
This project was developed for the final course project in Deep Learning.  
The objectives are:

- Learn and practice RNN-based and attention-based sequence models  
- Compare forecasting accuracy across architectures  
- Understand the impact of architectural choices on short-horizon prediction  
- Train models using consistent settings for fair comparison  

The dataset used is Bitcoin historical OHLCV price data from **Kaggle (2012–2024)**.

---
