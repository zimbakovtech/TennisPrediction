from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from functions.utils import process_y, read_file
from models.naive_bayes import NaiveBayes
from models.decision_tree import DecisionTree
from models.random_forest import RandomForest
from models.xgboost import XGBoost
from models.light_gbm import LightGBM
from models.catboost import CatBoost
from models.neural_network import NeuralNetwork
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("=== Training and Evaluating Models ===")

    nb = NaiveBayes(X_train, y_train)
    evaluate_model(nb, X_test, y_test, X_train, y_train, feature_names=feature_names)

    dt = DecisionTree(X_train, y_train)
    evaluate_model(dt, X_test, y_test, X_train, y_train, feature_names=feature_names)

    rf = RandomForest(X_train, y_train)
    evaluate_model(rf, X_test, y_test, X_train, y_train, feature_names=feature_names)
    
    xgb = XGBoost(X_train, y_train)
    evaluate_model(xgb, X_test, y_test, X_train, y_train, feature_names=feature_names)

    catboost = CatBoost(X_train, y_train)
    evaluate_model(catboost, X_test, y_test, X_train, y_train, feature_names=feature_names)

    lgm = LightGBM(X_train, y_train)
    evaluate_model(lgm, X_test, y_test, X_train, y_train, feature_names=feature_names)

    nn = NeuralNetwork(X_train_scaled, y_train, X_train_scaled.shape[1])
    evaluate_model(nn, X_test_scaled, y_test, X_train_scaled, y_train, is_keras=True, feature_names=feature_names)

    wimbledon_df = pd.read_csv("data/processed/wimbledon_2025_processed.csv")
    X_wimbledon = wimbledon_df.drop(columns='win_loss').values
    preds = xgb.predict(X_wimbledon)
    preds = [2 if pred == 0 else 1 for pred in preds]
    probs = xgb.predict_proba(X_wimbledon)
    print("\n=== Predictions for Wimbledon 2025 ===")
    print(preds)
    print("\n=== Probabilities ===")
    print(probs)
