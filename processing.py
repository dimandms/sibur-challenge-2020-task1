from constants import FEATURE_COLUMNS, FEATURE_GASES, TARGET_COLUMNS
import pandas as pd
import numpy as np
from funcy import rcompose


def process(data):
    return rcompose(
        clean_outliers,
        fill_na,
        fill_na_test,
        smooth_median_test,
        smooth,
        add_specified_features,
    )(data)


def fill_na_test(data):
    X_train, y_train, X_test = data

    def fillna_window_func(x):
        if np.isnan(x.iloc[-1]):
            return x.median()

        return x.iloc[-1]

    X_test = X_test.rolling(100, min_periods=1).apply(
        lambda x: fillna_window_func(x))

    return X_train, y_train, X_test


def smooth_median_test(data):
    X_train, y_train, X_test = data

    def smooth_window_func(x, out_precent=10):
        if abs(x.iloc[-1] - x.median()) > x.median()*out_precent/100:
            return x.median()

        return x.iloc[-1]

    X_test = X_test.rolling(100, min_periods=1).apply(
        lambda x: smooth_window_func(x, 20))

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

    return X_train, y_train, test_features


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
