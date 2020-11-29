import pandas as pd

def load_data():
    train_features = pd.read_csv('data/train_features.csv', index_col="timestamp", parse_dates=["timestamp"])
    train_targets = pd.read_csv('data/train_targets.csv', index_col="timestamp", parse_dates=["timestamp"])
    test_features = pd.read_csv('data/test_features.csv', index_col="timestamp", parse_dates=["timestamp"])

    return train_features, train_targets, test_features