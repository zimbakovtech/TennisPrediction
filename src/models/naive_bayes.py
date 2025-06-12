from sklearn.naive_bayes import GaussianNB

def NaiveBayes(X_train, y_train):
    nb = GaussianNB(var_smoothing=1e-9)

    print("\n----- Training Naive Bayes -----")
    print(f"Naive Bayes parameters: {nb.get_params()}")
    nb.fit(X_train, y_train)

    return nb
