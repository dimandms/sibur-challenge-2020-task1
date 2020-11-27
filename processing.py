from constants import FEATURE_GASES, TARGET_COLUMNS, FEATURE_GASES_MASS, TARGET_COLUMNS_MASS
import pandas as pd
import numpy as np
from funcy import compose


def process(data):
    return compose(
        fill_na,
        convert_to_masses,
        clean,
        smooth_test_df
    )(data)


def fill_na(data):
    train_features, train_targets, test_features = data

    X_train = train_features.fillna(method='ffill').fillna(
        method='bfill')  # both directions
    y_train = train_targets.fillna(method='ffill').fillna(
        method='bfill')  # both directions
    X_test = test_features.fillna(method='ffill')  # just forward direction
    # X_test = test_features.fillna(method='ffill').fillna(method='bfill') #NOT VALID WITHIN RULES

    return X_train, y_train, X_test


def convert_to_masses(data):
    """from % of components and total rates to masses/rates of components"""
    X_train, y_train, X_test = data

    X_train_processed = X_train.copy()
    for gas in FEATURE_GASES:
        X_train_processed[f"{gas}_mass"] = X_train[gas]*X_train['A_rate']/100

    y_train_processed = y_train.copy()
    for gas in TARGET_COLUMNS:
        y_train_processed[f"{gas}_mass"] = y_train[gas]*X_train['B_rate']/100

    X_test_processed = X_test.copy()
    for gas in FEATURE_GASES:
        X_test_processed[f"{gas}_mass"] = X_test[gas]*X_test['A_rate']/100

    X_train_converted = pd.concat(
        [X_train_processed[FEATURE_GASES_MASS], X_train['timestamp'].dt.strftime('%Y-%m-%d %H:%M')], axis=1)

    y_train_converted = pd.concat(
        [y_train_processed[TARGET_COLUMNS_MASS], y_train['timestamp'].dt.strftime('%Y-%m-%d %H:%M')], axis=1)

    X_test_converted = pd.concat(
        [X_test_processed[[FEATURE_GASES_MASS]], X_test['timestamp'].dt.strftime('%Y-%m-%d %H:%M')], axis=1)

    return X_train_converted, y_train_converted, X_test_converted


def clean(data):
    X_train, y_train, X_test = data
    X_y_train = pd.concat([X_train.drop("timestamp", axis=1), y_train], axis=1)

    MINIMAL_VALUE = 1
    clean_mask = X_y_train.drop(['timestamp', 'A_CH4_mass'], axis=1).apply(
        lambda x: mask_row(x, MINIMAL_VALUE), axis=1)
    X_y_train_clean = X_y_train[clean_mask]

    date_clean_mask = (X_y_train['timestamp'] < "2020-04-08 8:30") | (
        X_y_train['timestamp'] > "2020-04-11 12:00")
    X_y_train_clean_dates = X_y_train_clean[date_clean_mask]

    X_y_train_clean_dates.loc[4463:4465, 'A_C6H14_mass'] = 3.942185
    X_y_train_clean_dates.loc[4901, :] = X_y_train_clean_dates.loc[4900, :]
    X_y_train_clean_dates.loc[4902, :] = X_y_train_clean_dates.loc[4900, :]
    X_y_train_clean_dates.loc[4734, :] = X_y_train_clean_dates.loc[4733, :]
    X_y_train_clean_dates.loc[4738, :] = X_y_train_clean_dates.loc[4737, :]
    X_y_train_clean_dates.loc[4739, :] = X_y_train_clean_dates.loc[4737, :]
    X_y_train_clean_dates.loc[4873, :] = X_y_train_clean_dates.loc[4881, :]
    X_y_train_clean_dates.loc[4874, :] = X_y_train_clean_dates.loc[4881, :]
    X_y_train_clean_dates.loc[4875, :] = X_y_train_clean_dates.loc[4881, :]
    X_y_train_clean_dates.loc[4876, :] = X_y_train_clean_dates.loc[4881, :]
    X_y_train_clean_dates.loc[4877, :] = X_y_train_clean_dates.loc[4881, :]
    X_y_train_clean_dates.loc[4878, :] = X_y_train_clean_dates.loc[4881, :]
    X_y_train_clean_dates.loc[4879, :] = X_y_train_clean_dates.loc[4881, :]
    X_y_train_clean_dates.loc[4880, :] = X_y_train_clean_dates.loc[4881, :]

    X_train_clean_dates = X_y_train_clean_dates[FEATURE_GASES_MASS.append(
        'timestamp')]
    y_train_clean_dates = X_y_train_clean_dates[TARGET_COLUMNS_MASS.append(
        'timestamp')]

    return X_train_clean_dates, y_train_clean_dates, X_test


def smooth_test_df(data):
    X_train, y_train, X_test = data
    X_test_smoothed = X_test.copy()

    WINDOW_SIZE = 30
    OUT_PERCENT = 10
    for gas in FEATURE_GASES_MASS:
        X_test_smoothed[gas] = X_test_smoothed[gas].rolling(
            WINDOW_SIZE, min_periods=1).apply(lambda x: smooth_window_func(x, OUT_PERCENT))

    return X_train, y_train, X_test_smoothed


def smooth_window_func(x, out_precent=10):
    if abs(x.iloc[-1] - x.median()) > x.median()*out_precent/100:
        return np.median(x)

    return x.iloc[-1]


def smooth_series(series):
    WINDOW_SIZE = 30
    OUT_PERCENT = 5
    return series.rolling(WINDOW_SIZE, min_periods=1).apply(lambda x: smooth_window_func(x, OUT_PERCENT))
    # test_B_rate_smoothed = X_test['B_rate'].rolling(30, min_periods=1).apply(lambda x: window_func(x, 5))


def convert_to_percentes(gas_rate, total_rate):  # vectorized
    return gas_rate/total_rate*100


def has_value_less_then(arr, value):
    for x in arr:
        if x < value:
            return True

    return False


def mask_row(row, value):
    return not has_value_less_then(row, value)
