from catboost import CatBoostClassifier

def CatBoost(X_train, y_train):
    cb_params = {
        'iterations': 500,
        'depth': 5,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'random_seed': 42,
        'logging_level': 'Silent',
        'eval_metric': 'Logloss'
    }

    # Initialize the classifier
    model = CatBoostClassifier(
        **cb_params,
        thread_count=2
    )

    print("\n----- Training CatBoost -----")
    print(f"CatBoost parameters: {cb_params}")

    # Fit the model
    model.fit(
        X_train,
        y_train
    )

    return model