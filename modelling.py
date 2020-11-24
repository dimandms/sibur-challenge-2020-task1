from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, r2_score
from metrics import mean_absolute_percentage_error, absolute_errors
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from constants import TARGET_COLUMNS, FEATURE_COLUMNS


def evaluate_training(X_train, y_train, verbose=False, show_fit_plots=False):
    models = []
    for target in TARGET_COLUMNS:
        model_pipline = Pipeline([
            ("regressor", Ridge())
        ])

        params_grid = {
            "regressor__alpha": np.logspace(-8, 3, num=12, base=10),
            "regressor__fit_intercept": [True, False],
        }

        model = GridSearchCV(model_pipline,
                             params_grid,
                             scoring=make_scorer(r2_score),
                             n_jobs=-1,
                             cv=10,
                             verbose=1,
                             refit=True,
                             return_train_score=True
                             )

        model.fit(X_train[FEATURE_COLUMNS], y_train[target])

        if verbose:
            show_model_results(model)

            errs = absolute_errors(
                y_train[target], model.best_estimator_.predict(X_train[FEATURE_COLUMNS]),)
            print(f"errs mean: {errs.mean()} std: {errs.std()}")
            print(
                f"r2: {r2_score(y_train[target], model.best_estimator_.predict(X_train[FEATURE_COLUMNS]))}")

            if show_fit_plots:
                plot_fitted_values(model.best_estimator_,
                                   X_train[FEATURE_COLUMNS],
                                   y_train[target],
                                   target)

        models.append(model)

    return {target: model for target, model in zip(TARGET_COLUMNS, models)}


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
