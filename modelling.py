from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline

import numpy as np

from constants import TARGET_COLUMNS, FEATURE_COLUMNS


def evaluate_training(X_train, y_train):
    results = []
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
                             scoring=make_scorer(mean_absolute_error),
                             n_jobs=-1,
                             cv=10,
                             verbose=1,
                             refit=True,
                             return_train_score=True
                             )

        model.fit(X_train[FEATURE_COLUMNS], y_train[target])

        results.append(model)

    return {target: model for target, model in zip(TARGET_COLUMNS, results)}
