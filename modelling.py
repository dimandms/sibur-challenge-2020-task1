from sklearn.linear_model import ElasticNet, Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import make_scorer
from metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.base import TransformerMixin, BaseEstimator
from xgboost import XGBRegressor
from sklearn.kernel_ridge import KernelRidge

from sklearn.gaussian_process.kernels import ExpSineSquared

import numpy as np
import pandas as pd


def pass_columns(target):
    target_gas = target[2:]
    features = [
        f"A_{target_gas}_shift_175",
        f"A_{target_gas}_shift_185",
        f"A_{target_gas}_shift_195",
        "A_rate_shift_175",
        "A_rate_shift_185",
        "A_rate_shift_195",
    ]

    def filter(X):
        return X[features]

    return filter


def make_simple_model(target):
    model_pipline = Pipeline([
        ("shift", ShiftTransformer()),
        ("selection", FunctionTransformer(pass_columns(target))),
        ("scaler", StandardScaler()),
        ("regressor", Ridge())
    ])

    params_grid = {
        # "regressor__n_estimators": [15],
        "regressor__alpha": np.logspace(-8, -2, num=7, base=10),
        # "regressor__kernel": [ExpSineSquared(l, p)
        #                       for l in np.logspace(-2, 2, 10)
        #                       for p in np.logspace(0, 2, 10)],
        "shift__shifts": [[175, 185, 195]],
    }
    # model_pipline = Pipeline([
    #     ("shift", ShiftTransformer()),
    #     ("scaler", StandardScaler()),
    #     ("selection", SelectKBest(f_regression)),
    #     ("regressor", ElasticNet(random_state=42))
    #     # ("regressor", Ridge(random_state=42))
    # ])

    # params_grid = {
    #     "regressor__alpha": np.logspace(-8, -2, num=7, base=10),
    #     "regressor__l1_ratio": [0, 1],
    #     "regressor__fit_intercept": [True],
    #     "selection__k": [1, 3, 5],
    #     # "shift__shifts": [list(range(175, 196))],
    #     "shift__shifts": [[175, 185, 195]],
    # }

    model = GridSearchCV(model_pipline,
                         params_grid,
                         scoring=make_scorer(
                             mean_absolute_percentage_error, greater_is_better=False),
                         n_jobs=-1,
                         cv=KFold(n_splits=10, shuffle=True, random_state=42),
                         refit=True,
                         return_train_score=True
                         )

    return model


class ShiftTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, shifts=[]):
        self.shifts = shifts

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        for column in df:
            if column.startswith("A"):
                for shift in self.shifts:
                    df[f"{column}_shift_{shift}"] = df[column].shift(
                        shift, fill_value=df[column][0])

        return df
