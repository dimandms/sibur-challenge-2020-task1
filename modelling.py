from sklearn.linear_model import ElasticNet, Ridge
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import make_scorer
from metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.base import TransformerMixin, BaseEstimator
from xgboost import XGBRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures

from sklearn.gaussian_process.kernels import ExpSineSquared

import numpy as np
import pandas as pd


def pass_columns(target):
    target_gas = target[2:]

    def filter(X):
        features = [
            column for column in X if f"{target_gas}_shift" in column or "A_rate_shift" in column]
        # print(f"{target_gas}, features: {features}")
        return X[features]

    return filter


def make_simple_model(target):
    model_pipline = Pipeline([
        ("selection", FunctionTransformer(pass_columns(target))),
        ("polinom", PolynomialFeatures()),
        # ("kbest", SelectKBest(mutual_info_regression)),
        ("scaler", StandardScaler()),
        # ("regressor", Ridge(random_state=42)),
        ("nn", MLPRegressor(random_state=42, learning_rate="adaptive", max_iter=1000)),
    ])

    params_grid = {
        # "regressor__alpha": np.logspace(-5, -2, num=4, base=10),
        # "polinom__degree": [1, 2],
        "polinom__degree": [1],
        "polinom__interaction_only": [False],
        "polinom__include_bias": [False],
        # "kbest__k": [6, "all"],
        "nn__hidden_layer_sizes": [(1,)]
        # "nn__hidden_layer_sizes": [(256, 256)]
    }

    model = GridSearchCV(model_pipline,
                         params_grid,
                         scoring=make_scorer(
                             mean_absolute_percentage_error, greater_is_better=False),
                         n_jobs=-1,
                         #  cv=TimeSeriesSplit(n_splits=10),
                         cv=KFold(n_splits=5, shuffle=True, random_state=42),
                         refit=True,
                         return_train_score=True,
                         verbose=10
                         )

    return model
