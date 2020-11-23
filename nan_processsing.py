def process_na(data):
    train_features, train_targets, test_features = data

    X_train = train_features.fillna(method='ffill').fillna(method='bfill') #both directions
    y_train = train_targets.fillna(method='ffill').fillna(method='bfill') #both directions
    X_test = test_features.fillna(method='ffill')# just forward direction
    # X_test = test_features.fillna(method='ffill').fillna(method='bfill') #NOT VALID WITHIN RULES

    return X_train, y_train, X_test