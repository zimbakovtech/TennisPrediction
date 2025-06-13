import lightgbm as lgb


def LightGBM(X_train, y_train):
    
    lgb_params = {
        'n_estimators': 560,
        'max_depth': 5,
        'learning_rate': 0.01,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'feature_fraction': 0.8,
        'num_leaves': 31,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'random_state': 42
    }

    # Initialize the classifier
    model = lgb.LGBMClassifier(
        **lgb_params,
        n_jobs=2
    )

    print("\n----- Training LightGBM -----")
    print(f"LightGBM parameters: {lgb_params}")

    # Fit the model
    model.fit(X_train,y_train)

    return model
