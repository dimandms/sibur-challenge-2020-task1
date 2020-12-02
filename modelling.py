from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.base import TransformerMixin
from sklearn.ensemble import StackingRegressor

import numpy as np
import pandas as pd


def make_stacked_model(shifts):
    estimators = [(f"regressor_{shift}", make_simple_model(shift))
                  for shift in shifts]
    final_estimator = ElasticNetCV(
        alphas=np.logspace(-5, -2, num=3, base=10), l1_ratio=1, random_state=42)

    stacking_regressor = StackingRegressor(
        estimators=estimators, cv=KFold(n_splits=10, shuffle=True, random_state=42), final_estimator=final_estimator)
    return stacking_regressor


def make_simple_model(shift):
    model_pipline = Pipeline([
        ("shift", ShiftTransformer(shift)),
        ("scaler", StandardScaler()),
        ("selection", SelectKBest(f_regression)),
        ("regressor", ElasticNet())
    ])

    params_grid = {
        "regressor__alpha": np.logspace(-8, -2, num=7, base=10),
        "regressor__l1_ratio": [1],
        "regressor__fit_intercept": [True],
        "selection__k": [3],
    }

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


class ShiftTransformer(TransformerMixin):
    def __init__(self, shift_num):
        self.shift_num = shift_num

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        for column in df:
            if column.startswith("A"):
                df[column] = df[column].shift(
                    self.shift_num, fill_value=df[column][0])

        return df
