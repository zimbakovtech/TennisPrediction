# 🎾 Tennis Match Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

Predict ATP tennis match outcomes with advanced machine learning on 2015–2024 data.

---

## 📋 Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Repository Structure](#repository-structure)  
4. [Installation & Setup](#installation--setup)  
5. [Data Preparation](#data-preparation)  
6. [Training & Evaluation](#training--evaluation)  
7. [Results & Metrics](#results--metrics)  
8. [Feature Importances](#feature-importances)  
9. [Future Work](#future-work)  
10. [License](#license)

---

## Overview

This project builds, trains, and evaluates multiple classifiers to predict the winners of ATP tennis matches. Based on Jeff Sackmann’s publicly available match data from 2015 to 2024, we engineer rich features—Elo ratings, head‑to‑head stats, recent form metrics—and compare model performances.

---

## Features

- **Data Ingestion & Cleaning**  
  Load and combine raw CSVs, filter for main‑draw matches, normalize surfaces, and handle missing entries.

- **Feature Engineering**  
  - **Elo Ratings**: Global & surface‑specific, updated pre‑match with K = 32  
  - **Head‑to‑Head**: Win–loss differential for player rivalries  
  - **Recent Form**: Rolling statistics over last N matches (aces, double faults, break points saved)  
  - **Ranking Metrics**: Differences in ATP rank and ranking points  
  - **Tournament Encoding**: Importance level and round information  

- **Modeling Suite**  
  - Naive Bayes  
  - Decision Tree  
  - Random Forest  
  - XGBoost (top performer)  
  - CatBoost  
  - LightGBM  
  - Neural Network  

- **Evaluation Tools**  
  Calculate accuracy, confusion matrices, and feature importances.

---

## Repository Structure

```
TennisPrediction/
├── data/
│   ├── raw/                 # Raw match CSVs (2015–2024)
│   ├── processed/           # Cleaned & engineered datasets
│   └── players/             # Precomputed Elo ratings
│
├── src/
│   ├── feature_engineering/ # Elo, stats, head2head
│   ├── functions/           # Preprocessing utilities
│   ├── models/              # Individual model definitions
│   ├── train_models.py      # Training and evaluation pipeline
│   └── processing_data.py   # Data preparation pipeline
│
├── requirements.txt  
├── LICENSE  
└── README.md
```

---

## Installation & Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/zimbakovtech/TennisPrediction.git
   cd TennisPrediction
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**  
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

## Training & Evaluation

Train and evaluate models with a 70/30 train-test split:
```bash
python src/train_models.py
```
Outputs:
- Performance metrics for each classifier  
- Confusion matrices and ROC curves  
- Feature importance rankings

---

## Results & Metrics

> **Note:** the multi-model comparison below is historical. The current code
> trains only XGBoost and reports metrics from a leak-free, time-respecting
> evaluation: **Test ≈ 65.8% / Train ≈ 68.8%**, Brier ≈ 0.212, Log Loss ≈ 0.611,
> F1 ≈ 0.658, time-series CV ≈ 65.6% (±0.9%). These are slightly below the older
> figures because the previous CV was leaky and missing form values were
> zero-filled. Regenerate everything with `myenv/bin/python src/processing_data.py`
> then `myenv/bin/python src/train_models.py` (see `CLAUDE.md`).

**XGBoost** achieved the best test accuracy of **66.34%** (67.77% train)  
A concise comparison:

| Model           | Test Acc. (%) | Train Acc. (%) | Brier Score | Log Loss | Test Precision (%) | Test Recall (%) | Test F1 (%) |
|-----------------|--------------:|---------------:|------------:|---------:|-------------------:|----------------:|------------:|
| Naive Bayes     |         64.54 |          65.29 |        0.24 |     0.73 |              64.54 |           64.54 |       64.54 |
| Decision Tree   |         64.07 |          65.64 |        0.22 |     0.63 |              63.75 |           65.22 |       64.48 |
| Random Forest   |         65.36 |          67.07 |        0.21 |     0.61 |              65.34 |           65.41 |       65.37 |
| **XGBoost**     |       **66.34**|        **67.77**|      **0.21**| **0.61**|            **66.32**|       **66.36**|   **66.34** |
| CatBoost        |         65.55 |          66.21 |        0.21 |     0.61 |              65.52 |           65.64 |       65.58 |
| LightGBM        |         66.10 |          68.09 |        0.21 |     0.61 |              66.13 |           66.02 |       66.07 |
| Neural Network  |         65.40 |          66.48 |        0.21 |     0.61 |              65.63 |           64.66 |       65.14 |

---

## Feature Importances

Relative importances (as percentages) for the current XGBoost model. The
feature set is 12 predictors; surface enters only via `surface_elo_diff`, and
`tourney_level`/`round`/`surface` are folded into `match_importance` and the Elo
features rather than used directly:

| Feature             | Importance (%) |
|---------------------|---------------:|
| points_diff         |          43.8 |
| surface_elo_diff    |           9.5 |
| rank_diff           |           7.3 |
| best_of             |           6.2 |
| opponent_elo_before |           5.1 |
| player_elo_before   |           5.0 |
| age_diff            |           4.6 |
| bp_diff             |           4.2 |
| df_diff             |           3.8 |
| ace_diff            |           3.6 |
| h2h_diff            |           3.6 |
| match_importance    |           3.4 |

---

## Future Work

- Develop a REST API (Flask/FastAPI) for live predictions  
- Build a web dashboard with React or static HTML  
- Automate weekly data ingestion  
- Integrate new features (injury flags, travel distances)  
- Implement periodic retraining to address concept drift

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

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---

*Prepared by Damjan Zimbakov & Andrej Milevski*  
*June 2025*  
