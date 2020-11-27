from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score
from metrics import mean_absolute_percentage_error, absolute_errors
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb

from constants import TARGET_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMNS_MASS


def train_rf_regression(X_train, y_train):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=10, stop=500, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    params_grid = {'regressor__n_estimators': n_estimators,
                   'regressor__max_features': max_features,
                   'regressor__max_depth': max_depth,
                   'regressor__min_samples_split': min_samples_split,
                   'regressor__min_samples_leaf': min_samples_leaf,
                   'regressor__bootstrap': bootstrap}

    model_pipline = Pipeline([
        ("regressor", RandomForestRegressor())
    ])

    model = RandomizedSearchCV(model_pipline,
                               params_grid,
                               scoring=make_scorer(
                                   mean_absolute_percentage_error, greater_is_better=False),
                               n_jobs=-1,
                               cv=5,
                               verbose=1,
                               refit=True,
                               random_state=42,
                               return_train_score=True
                               )

    model.fit(X_train, y_train)

    return model


def train_regression(X_train, y_train):
    model_pipline = Pipeline([
        ("regressor", Ridge())
    ])

    params_grid = {
        # "regressor__alpha": np.logspace(-8, 3, num=12, base=10),
        "regressor__alpha": np.logspace(-8, 8, num=17, base=10),
        "regressor__fit_intercept": [False, True],
    }

    model = GridSearchCV(model_pipline,
                         params_grid,
                         scoring=make_scorer(
                             mean_absolute_percentage_error, greater_is_better=False),
                         n_jobs=-1,
                         cv=5,
                         verbose=1,
                         refit=True,
                         return_train_score=True
                         )

    model.fit(X_train, y_train)
    show_model_results(model)

    return model


def evaluate_training(X_train, y_train):
    models = []
    for target in TARGET_COLUMNS_MASS:
        result = train_regression(X_train, y_train[target])
        models.append(result.best_estimator_)

    return models


def show_model_results(model):
    cv_results_df = pd.DataFrame(model.cv_results_)
    print("\n")
    print(cv_results_df[["param_regressor__alpha", "param_regressor__fit_intercept",
                         "mean_train_score", "std_train_score",
                         "mean_test_score", "std_test_score",
                         "rank_test_score"]]
          )


def plot_fitted_values(estimator, X, y_true, title):
    y_pred = estimator.predict(X)
    _, ax = plt.subplots(1, 1, figsize=(15, 3))

    ax.plot(y_true, label="true")
    ax.plot(y_pred, label="prediction")
    ax.legend()
    ax.set_title(title)
