from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from functions.utils import process_y, read_file
from models.xgboost import XGBoost
from model_evaluations import evaluate_model
import pandas as pd


if __name__ == "__main__":
    X, feature_names = read_file()
    y = process_y()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        shuffle=False
    )

    print("=== Training and Evaluating Models ===")
    
    xgb = XGBoost(X_train, y_train)
    evaluate_model(xgb, X_test, y_test, X_train, y_train, feature_names=feature_names)

    wimbledon_df = pd.read_csv("data/processed/wimbledon_2025_processed.csv")
    X_wimbledon = wimbledon_df.drop(columns='win_loss').values
    preds = xgb.predict(X_wimbledon)
    preds = [2 if pred == 0 else 1 for pred in preds]
    probs = xgb.predict_proba(X_wimbledon)
    print("\n=== Predictions for Wimbledon 2025 ===")
    print(preds[-1:])
    print("\n=== Probabilities ===")
    print(probs[-1:])
