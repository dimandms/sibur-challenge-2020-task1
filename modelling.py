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
from sklearn.neural_network import MLPRegressor

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
        ("selection", FunctionTransformer(pass_columns(target))),
        ("scaler", StandardScaler()),
        ("regressor", MLPRegressor(random_state=42, max_iter=700, early_stopping=True))
    ])

    params_grid = {
        "regressor__hidden_layer_sizes": [(2, 2), (3, 3), (2, 1), (3, 1)],
        "regressor__alpha": np.logspace(-5, -2, num=4, base=10),
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
