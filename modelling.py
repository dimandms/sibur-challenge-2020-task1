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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from constants import TARGET_COLUMNS_MASS


def train_regression(X_train, y_train):
    model_pipline = Pipeline([
        # ("scaler", StandardScaler()),
        # ("polynomal", PolynomialFeatures()),
        # ("selection", SelectKBest(f_regression)),
        # ('ann', MLPRegressor(max_iter=500, batch_size=2000))
        ("regressor", ElasticNet())
    ])

    params_grid = {
        "regressor__alpha": np.logspace(-8, 8, num=17, base=10),
        "regressor__l1_ratio": [0, 0.25, 0.5, 0.75, 1],
        "regressor__fit_intercept": [True],
    }

    # params_grid = {
    #     # "ann__alpha": [1e-3]
    #     # "ann__alpha": [1e-4,1e-3,1e-2],
    #     # "ann__alpha": np.logspace(-8, 8, num=17, base=10),
    #     # "ann__hidden_layer_sizes": np.logspace(-8, 8, num=17, base=10),
    # }

    model = GridSearchCV(model_pipline,
                         params_grid,
                         scoring=make_scorer(
                             mean_absolute_percentage_error, greater_is_better=False),
                         n_jobs=-1,
                         cv=5,
                         refit=True,
                         return_train_score=True
                         )

    model.fit(X_train, y_train)
    # show_model_results(model)

    return model


def evaluate_training(X_train, y_train):
    models = []
    scores = []
    for target in TARGET_COLUMNS_MASS:
        result = train_regression(X_train, y_train[target])
        models.append(result.best_estimator_)
        scores.append(result.best_score_ * -1)

    print("\n==================== Train results ====================")
    print(f"scores: {scores}")
    print(f"total score: {np.mean(scores)}")
    print("========================= End =========================")

    return models


def show_model_results(model):
    cv_results_df = pd.DataFrame(model.cv_results_)
    print("\n")
    print(cv_results_df[["mean_train_score", "std_train_score",
                         "mean_test_score", "std_test_score",
                         "rank_test_score"]]
          )


def plot_fitted_values(y_pred, y_true, title):
    _, ax = plt.subplots(1, 1, figsize=(15, 3))

    ax.plot(y_true, label="true")
    ax.plot(y_pred, label="prediction")
    ax.legend()
    ax.set_title(title)
