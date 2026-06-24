from sklearn.model_selection import train_test_split
from functions.utils import process_y, read_file
from models.xgboost import XGBoost
from model_evaluations import evaluate_model
import pandas as pd
import joblib


if __name__ == "__main__":
    X, feature_names = read_file()
    y = process_y()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        shuffle=False
    )
    
    xgb = XGBoost(X_train, y_train)
    # joblib.dump(xgb, "src/models/joblib/xgboost_model_2025.pkl")
    evaluate_model(xgb, X, y, X_test, y_test, X_train, y_train, feature_names=feature_names)
    row = [3, 18, 405, 8.3128, -4.00, 0.2, 1.4, -2.0, -1, 2013, 1699, 388]
    # [3, 18, -405, np.float64(8.3129), -4, 0.20000000000000107, -1.4000000000000001, -2.0, 0, 2013.0, 1699.0, 388.0]
    # [3, 18, 405, np.float64(8.3129), -4, 0.20000000000000107, 1.4000000000000001, -2.0, 0, 2013.0, 1699.0, 388.0]
    print(xgb.predict([row]))
    print(xgb.predict_proba([row]))

    # model_input = [
    #         best_of, match_importance, rank_diff, points_diff, age_diff, 
    #         ace_diff, df_diff, bp_diff, h2h_diff, 
    #         player_1_elo, player_2_elo, surface_elo_diff
    #     ]

    # wimbledon_df = pd.read_csv("data/testing/wimbledon_2025.csv")
    # X_wimbledon = wimbledon_df.drop(columns='win_loss').values
    # preds = xgb.predict(X_wimbledon)
    # preds = [2 if pred == 0 else 1 for pred in preds]
    # probs = xgb.predict_proba(X_wimbledon)
    # print("\n=== Predictions for Wimbledon 2025 ===")
    # print(preds)
