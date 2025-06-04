# Tennis Match Prediction

Predict ATP tennis match outcomes using machine learning on 2019–2024 data.

---

## Overview

This repository contains code and data pipelines to build, train, and evaluate models that predict the winner of ATP tennis matches. We use JeffSackmann’s publicly available match data (2019–2024), engineer features (Elo ratings, head-to-head, recent form, etc.), and compare several classifiers (Decision Tree, Random Forest, XGBoost, Neural Network).

---

## Features

- **Data Ingestion & Cleaning**: Load raw CSVs (2019–2024), filter main-draw matches, standardize surfaces, handle missing values.
- **Feature Engineering**:
  - Global and surface-specific Elo ratings
  - Head-to-head counts
  - Recent-form metrics (last N matches)
  - Ranking and rank-points differences
  - Tournament importance and round encoding
- **Modeling**:
  - Decision Tree (baseline)
  - Random Forest
  - XGBoost
  - Neural Network (Keras/TensorFlow)
- **Evaluation**: Accuracy, AUC-ROC, confusion matrix, feature importance.
- **Prediction**: Script to load a trained model and predict match outcome given two player IDs, surface, and date.

---

## Repository Structure

```
tennis-prediction/
├── data/
│   └── raw/
│       ├── atp_matches_2019.csv
│       ├── atp_matches_2020.csv
│       ├── atp_matches_2021.csv
│       ├── atp_matches_2022.csv
│       ├── atp_matches_2023.csv
│       └── atp_matches_2024.csv
│
├── notebooks/
│   ├── 01_explore_data.ipynb
│   └── 02_feature_engineering.ipynb
│
├── src/
│   ├── data_loader.py          # Load and clean raw CSVs
│   ├── feature_engineer.py     # Compute Elo, H2H, recent form
│   ├── preprocessing.py        # Mirror matches, split, scale
│   ├── train_models.py         # Train DT, RF, XGB, NN
│   ├── evaluate_models.py      # Evaluate and plot results
│   ├── predict.py              # Build feature vector and predict
│   └── utils.py                # Helper functions (round mapping, etc.)
│
├── models/                     # Trained model files and scalers
│   ├── decision_tree_2024.joblib
│   ├── random_forest_2024.joblib
│   ├── xgb_2024.json
│   ├── nn_2024.h5
│   └── scaler_2024.joblib
│
├── results/
│   ├── metrics/                # CSV reports (accuracy, AUC)
│   ├── plots/                  # ROC curves, feature importance
│   └── logs/                   # Training logs
│
├── requirements.txt            # Python package dependencies
├── README.md                   # This file
└── .gitignore
```

---

## Setup & Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/<your-username>/tennis-prediction.git
   cd tennis-prediction
   ```

2. **Install Dependencies**  
   - If you want to isolate dependencies, create a virtual environment (recommended):  
     ```bash
     python3 -m venv venv
     source venv/bin/activate        # Linux/macOS
     # venv\Scripts\activate       # Windows
     ```
   - Install required packages:  
     ```bash
     pip install -r requirements.txt
     ```

3. **Data Placement**  
   - Download the ATP match CSVs for 2019–2024 from JeffSackmann’s GitHub and place them in `data/raw/` with filenames:  
     ```
     atp_matches_2019.csv
     atp_matches_2020.csv
     atp_matches_2021.csv
     atp_matches_2022.csv
     atp_matches_2023.csv
     atp_matches_2024.csv
     ```

---

## Usage

1. **Data Exploration**  
   - Open and run `notebooks/01_explore_data.ipynb` to inspect raw data distributions, missing values, and column formats.

2. **Feature Engineering**  
   - Run `notebooks/02_feature_engineering.ipynb` (or call functions in `src/feature_engineer.py`) to generate Elo ratings, H2H counts, and recent-form features.

3. **Preprocessing**  
   ```python
   from src.preprocessing import create_diff_rows, split_train_test, scale_and_encode

   # Example (in a Python script or REPL):
   df_all = ...  # Combined DataFrame after feature engineering
   X, y = create_diff_rows(df_all)
   X_tr, X_val, X_test, y_tr, y_val, y_test = split_train_test(X, y, test_start_date="2024-07-01")
   X_tr_scaled, X_val_scaled, X_test_scaled, scaler = scale_and_encode(X_tr, X_val, X_test)
   ```

4. **Training Models**  
   - Edit and run `src/train_models.py` to train each model. For example:  
     ```bash
     python src/train_models.py --model xgb --params params_xgb.yaml
     ```
   - Saved models and scalers will appear in `models/`.

5. **Evaluation**  
   - Run `src/evaluate_models.py` to load trained models from `models/`, compute metrics on the test set, and generate plots in `results/`.

6. **Prediction**  
   - Use `src/predict.py` to load a saved model and predict the outcome of a match. Example:  
     ```bash
     python src/predict.py --playerA 104223 --playerB 165335 --date 2025-06-05 --surface Clay
     ```
   - Output: probability of player A winning, and predicted winner ID.

---

## Development Notes

- All code in `src/` should be written as importable modules, with clear function signatures and docstrings.
- Use chronological splits (no random shuffling) to avoid data leakage.
- Ensure Elo and H2H features are computed strictly using pre-match information.
- Commit often with descriptive messages (e.g., “Add Elo feature engineering,” “Train XGBoost model on 2024 data”).

---

## Future Work

- Build a Flask/FastAPI wrapper around `predict.py` for a simple REST API.
- Create a web dashboard (React or static HTML) to display predictions and model insights.
- Automate data updates each week as new match CSVs become available.
- Explore additional features (injury flags, travel distance, betting odds).
- Schedule periodic retraining to handle concept drift.

---
