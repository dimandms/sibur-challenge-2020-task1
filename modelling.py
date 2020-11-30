from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from metrics import mean_absolute_percentage_error, absolute_errors
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

from xgboost import XGBRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from constants import TARGET_COLUMNS


def train_regression(X_train, y_train):
    model_pipline = Pipeline([
        ("scaler", StandardScaler()),
        # ("pca", PCA()),
        ("regressor", ElasticNet())
    ])

    params_grid = {
        "regressor__alpha": np.logspace(-10, 10, num=21, base=10),
        "regressor__l1_ratio": [0, 0.25, 0.5, 0.75, 1],
        "regressor__fit_intercept": [True],
        # "pca__n_components": [3, 5, 7, 10]
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

    model.fit(X_train, y_train)

    return model


def evaluate_training(X_train, y_train):
    models = []
    for target in TARGET_COLUMNS:
        result = train_regression(X_train, y_train[target])
        models.append(result)

    return models
