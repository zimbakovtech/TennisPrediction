from xgboost import XGBClassifier


def XGBoost(X_train, y_train):
    xgb_params = {
        'n_estimators': 260,
        'max_depth': 7,
        'learning_rate': 0.01,
        'subsample': 0.7,
    }
    xgb = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
        n_jobs=2,
        tree_method='hist',
        xgb_params=xgb_params
    )
    print("----- Training XGBoost -----")
    xgb.fit(X_train, y_train)
    return xgb
