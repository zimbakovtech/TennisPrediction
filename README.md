# Tennis Match Prediction

Predict ATP tennis match outcomes using machine learning on 2015–2024 data.

---

## Overview

This repository contains code and data pipelines to build, train, and evaluate models that predict the winner of ATP tennis matches. We use Jeff Sackmann’s publicly available match data (2015–2024), engineer features (Elo ratings, head‑to‑head, recent form, etc.), and compare several classifiers (Decision Tree, Random Forest, XGBoost, Neural Network).

---

## Features

- **Data Ingestion & Cleaning**  
  Load raw CSVs (2015–2024), filter main‑draw matches, standardize surfaces, and handle missing values.

- **Feature Engineering**  
  - Global and surface‑specific Elo ratings  
  - Head‑to‑head counts  
  - Recent‑form metrics (last N matches)  
  - Ranking and rank‑points differences  
  - Tournament importance and round encoding

- **Modeling**  
  - Naive Bayes
  - Decision Tree 
  - Random Forest  
  - XGBoost  
  - CatBoost
  - LightGBM
  - Neural Network

- **Evaluation**  
  Accuracy, AUC‑ROC, confusion matrix, feature importance.

- **Prediction**  
  Script to load a trained model and predict match outcome given two player IDs, surface, and date.

---

## Repository Structure

```
TennisPrediction/
├── data/
│   ├── players/                 # Player Elo ratings
│   │   └── player_elo_ratings.csv
│   ├── processed/               # Engineered and cleaned datasets
│   │   ├── all_matches.csv
│   │   ├── all_matches_pre_engineering.csv
│   │   └── all_matches_elo.csv
│   └── raw/                     # Original Jeff Sackmann match CSVs
│       ├── atp_matches_2015.csv
│       ├── atp_matches_2016.csv
│       ├── …
│       └── atp_matches_2024.csv
│
├── src/
│   ├── feature_engineering/     # Elo, statistics, head‑to‑head
│   │   ├── calculate_elo.py
│   │   ├── generate_stats.py
│   │   └── head2head.py
│   │
│   ├── functions/               # Preprocessing and utilities
│   │   ├── duplicate_entries.py
│   │   ├── preprocessing.py
│   │   └── utils.py
│   │
│   ├── models/                  # Model definitions
│   │   ├── catboost.py
│   │   ├── decision_tree.py
│   │   ├── light_gbm.py
│   │   ├── naive_bayes.py
│   │   ├── neural_network.py
│   │   ├── random_forest.py
│   │   └── xgboost.py
│   │
│   ├── model_evaluations.py     # Evaluation routines
│   ├── processing_data.py       # Data pipeline
│   └── train_models.py          # Training scripts
│
├── 
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup & Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/zimbakovtech/TennisPrediction.git
   cd TennisPrediction
   ```

2. **Install Dependencies**  
   - Create and activate a virtual environment (recommended):  
     ```bash
     python3 -m venv venv
     source venv/bin/activate        # Linux/macOS
     # venv\Scripts\activate       # Windows
     ```
   - Install required packages:  
     ```bash
     pip install -r requirements.txt
     ```

---

## Data Preparation

1. **Process Data**  
   Run `processing_data.py` to recreate `all_matches.csv`. This script will:
   - Load and combine all raw files
   - Encode categorical values (rounds, surfaces, hand, tournament levels)
   - Compute differences: `rank_diff`, `points_diff`, `age_diff`, `ace_diff`, `df_diff`, `bp_diff`, `h2h_diff`
   - Drop redundant or leaking columns
   - Mirror each match for bidirectional modeling
   - Calculate pre‑match Elo ratings (K = 32)

---

## Training & Evaluating Models

Run `train_models.py` to train and evaluate the following classifiers on a 70%/30% train/test split:

### Test & Train Accuracies

| Model             | Test Accuracy | Train Accuracy |
|-------------------|--------------:|---------------:|
| Naive Bayes       |        64.54% |         65.29% |
| Decision Tree     |        64.07% |         65.64% |
| Random Forest     |        65.26% |         66.97% |
| **XGBoost (best)**|   **66.30%**  |      **67.76%**|
| CatBoost          |        65.63% |         66.26% |
| LightGBM          |        66.11% |         68.12% |
| Neural Network    |        65.72% |         66.36% |

> **XGBoost achieved the highest test accuracy of 66.30%.**

---

## Feature Importances

Below are the relative importances (as percentages) for the XGBoost model:

| Feature             | Importance (%) |
|---------------------|---------------:|
| points_diff         |          48.47 |
| rank_diff           |           8.02 |
| player_elo_before   |           6.04 |
| opponent_elo_before |           5.85 |
| best_of             |           4.88 |
| age_diff            |           4.21 |
| bp_diff             |           4.13 |
| df_diff             |           3.86 |
| h2h_diff            |           3.22 |
| tourney_level       |           3.17 |
| surface             |           2.81 |
| round               |           2.74 |
| ace_diff            |           2.61 |

---

## Future Work

- Build a Flask/FastAPI REST API for live predictions.  
- Develop a web dashboard (React or static HTML) for visualization.  
- Automate weekly data updates as new match CSVs become available.  
- Explore additional features (injury flags, travel distance, betting odds).  
- Schedule periodic retraining to adapt to concept drift.

---

## Summary

This project employs comprehensive data preparation and advanced feature engineering to predict ATP match outcomes with machine learning. Leveraging Jeff Sackmann’s publicly available match data (2015–2024), we have:

- **Engineered Form Metrics**  
  Computed rolling averages over each player’s last 10 matches for ace, double‑fault, and break‑point‑saved statistics, then combined these into three different features:  
  - `ace_diff` (Ace Difference)  
  - `df_diff` (Double‑Fault Difference)  
  - `bp_diff` (Break‑Points‑Saved Difference)

- **Head‑to‑Head Insights**  
  Introduced `h2h_diff`, capturing the win–loss differential between any two players to inform the model of their historical rivalry.

- **Dynamic Elo Ratings**  
  Implemented a zero‑sum Elo update:  
  ```
  E₁ = 1 / (1 + 10^((elo₂ – elo₁) / 400)),  
  new_elo₁ = elo₁ + K · (S₁ – E₁)
  ```  
  (and likewise for player 2), providing richer skill estimates than static ranking points.

- **Class Balance via Row Mirroring**  
  To avoid a one‑class (“Win”) target, every match row (A beats B) is mirrored: B vs. A with inverted feature signs or swapped values. This doubling ensures a balanced bidirectional dataset.

- **Feature Selection & Overfitting Control**  
  After experimenting with varying feature sets, we settled on 14 predictors. Hyperparameter tuning (number of estimators, tree depth, etc.) minimized overfitting, yielding only a 1.43 percentage‐point gap between train and test accuracy.

- **Theoretical Accuracy Ceiling**  
  Tennis match prediction is bounded by game unpredictability:  
  - **Lower bound (random):** 50 %  
  - **Upper bound (ideal statistical model):** ~70 %  
  Mapping our best test accuracy (66.34 %) onto this 50–70 % range gives a scaled success score of **81.7 %**.

- **Tournament‐Specific Performance**  
  On the Australian Open 2025 (Rounds of 32 through Final), our top XGBoost model achieved **86.67 %** accuracy—surpassing the overall theoretical limit, likely due to a smaller field of elite competitors and more consistent play.

This summary highlights the principal data transformations, modeling strategies, and performance benchmarks that underpin our ATP match‐prediction pipeline.

---

*Prepared by Damjan Zimbakov & Andrej Milevski*  
*June 2025*  
