from constants import FEATURE_COLUMNS, FEATURE_GASES, TARGET_COLUMNS
import pandas as pd
import numpy as np
from funcy import rcompose


def process(data):
    return rcompose(
        shift,
        clean_outliers,
        fill_na,
        debug,
        smooth,
        add_specified_features
    )(data)

def debug(data):
    train_features, train_targets, test_features = data

    print(train_features[train_features.isna().any(axis=1)])
    print(train_targets[train_targets.isna().any(axis=1)])
    print(test_features[test_features.isna().any(axis=1)])

    return train_features, train_targets, test_features


def shift(data):
    train_features, train_targets, test_features = data
    train_df = pd.concat([train_features, train_targets], axis=1)
    df = pd.concat([train_df, test_features], axis=0)

    SHIFT = 184
    for variable in TARGET_COLUMNS + FEATURE_COLUMNS:
        if variable.startswith("A"):
            df[variable] = df[variable].shift(SHIFT)

    X_train = df[FEATURE_COLUMNS].loc["2020-01-01 00:00:00":"2020-04-30 23:30:00", :]
    y_train = df[TARGET_COLUMNS].loc["2020-01-01 00:00:00":"2020-04-30 23:30:00", :]
    X_test = df[FEATURE_COLUMNS].loc["2020-05-01 00:00:00":"2020-07-22 23:30:00", :]

    return X_train, y_train, X_test


def clean_outliers(data):
    X_train, y_train, X_test = data

    X_train.loc["2020-04-08 05:30": "2020-04-12 07:00", :] = np.nan
    y_train.loc["2020-04-08 05:30": "2020-04-12 07:00", :] = np.nan

    X_train.loc["2020-01-25 19:00": "2020-02-14 16:30", :] = np.nan
    y_train.loc["2020-01-25 19:00": "2020-02-14 16:30", :] = np.nan

    rate_minimal_value = 20.0
    clean_mask = (X_train["A_rate"] < rate_minimal_value) | \
                 (X_train["B_rate"] < rate_minimal_value)

    X_train[clean_mask] = np.nan
    y_train[clean_mask] = np.nan

    return X_train, y_train, X_test


def fill_na(data):
    train_features, train_targets, test_features = data

    X_train = train_features.fillna(method='ffill').fillna(
        method='bfill')  # both directions
    y_train = train_targets.fillna(method='ffill').fillna(
        method='bfill')  # both directions
    X_test = test_features.fillna(method='ffill')  # just forward direction
    # X_test = test_features.fillna(method='ffill').fillna(method='bfill') #NOT VALID WITHIN RULES

    return X_train, y_train, X_test


def smooth(data):
    SPAN = 30
    train_features, train_targets, test_features = data
    train_df = pd.concat([train_features, train_targets], axis=1)
    df = pd.concat([train_df, test_features], axis=0).ewm(
        span=SPAN, min_periods=1).mean()

    X_train = df[FEATURE_COLUMNS].loc["2020-01-01 00:00:00":"2020-04-30 23:30:00", :]
    y_train = df[TARGET_COLUMNS].loc["2020-01-01 00:00:00":"2020-04-30 23:30:00", :]
    X_test = df[FEATURE_COLUMNS].loc["2020-05-01 00:00:00":"2020-07-22 23:30:00", :]

    return X_train, y_train, X_test


def add_specified_features(data):
    train_features, train_targets, test_features = data
    features_df = pd.concat([train_features, test_features])

    for gas in FEATURE_GASES:
        features_df[f"{gas}_specified"] = features_df[gas] * \
            features_df["A_rate"]/features_df["B_rate"]

    X_train = features_df.loc["2020-01-01 00:00:00":"2020-04-30 23:30:00", :]
    y_train = train_targets
    X_test = features_df.loc["2020-05-01 00:00:00":"2020-07-22 23:30:00", :]

    return X_train, y_train, X_test
