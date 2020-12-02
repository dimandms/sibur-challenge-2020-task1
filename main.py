from constants import TARGET_COLUMNS
import pickle
from args import parse_args

from load_data import load_data
from processing import process
from modelling import evaluate_training
from submission import create_submission, create_smoothed_submission
from evaluate import predict
from result_view import show_results
from metrics import mean_absolute_percentage_error

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor


def simple_model_preds(shift_num, loaded_data, verbose):
    train_features, train_targets, test_features = process(
        shift_num)(loaded_data)
    models = evaluate_training(train_features, train_targets)
    if verbose:
        print(f"shift_num: {shift_num}")
        show_results(models)

    estimators = [model.best_estimator_ for model in models]

    # -----
    fits = [pred for pred in predict(estimators, train_features)]
    _, axes = plt.subplots(4, 1, figsize=(15, 8))

    for target, ax, fit in zip(TARGET_COLUMNS, axes, fits):
        ax.plot(train_targets[target].values, label=f"true_{target}")
        ax.plot(fit, label=f"pred_{target}")
        ax.legend()

    _, axes = plt.subplots(4, 1, figsize=(15, 8))
    for target, ax, fit in zip(TARGET_COLUMNS, axes, fits):
        errs = train_targets[target].values - fit
        ax.hist(errs, bins=50)
        ax.legend()
    # -----
    return estimators, fits, test_features, train_targets


def lists_sub_mean(list1, list2):
    return [(v1+v2)/2 for v1, v2 in zip(list1, list2)]


def main():
    args = parse_args()
    loaded_data = load_data()

    results = []
    for shift_num in [184, 190]:
        result = simple_model_preds(shift_num, loaded_data, args.verbose)
        results.append(result)

    estimators1, fits1, test_features1, train_targets1 = results[0]
    estimators2, fits2, test_features2, _ = results[1]

    _, axes = plt.subplots(4, 1, figsize=(15, 8))

    for target, ax, fits in zip(TARGET_COLUMNS, axes, lists_sub_mean(fits1, fits2)):
        ax.plot(train_targets1[target].values, label=f"true_{target}")
        ax.plot(fits, label=f"pred_{target}")
        ax.legend()

    _, axes = plt.subplots(4, 1, figsize=(15, 8))
    for target, ax, fit in zip(TARGET_COLUMNS, axes, lists_sub_mean(fits1, fits2)):
        errs = train_targets1[target].values - fit
        ax.hist(errs, bins=50)
        ax.legend()

    plt.show()

    y_preds1 = [pred for pred in predict(estimators1, test_features1)]
    y_preds2 = [pred for pred in predict(estimators2, test_features2)]

    y_preds = lists_sub_mean(y_preds1, y_preds2)

    sub = create_submission(test_features1.index.values, y_preds)
    sub_smoothed = create_smoothed_submission(
        test_features1.index.values, y_preds)

    sub.to_csv(f'submission.csv', index=False)
    sub_smoothed.to_csv(f'submission_smoothed.csv', index=False)


if __name__ == "__main__":
    main()

"""1. Сделать преобразование в т/с и отбросить общие расходы. Потом восстанавливать проценты
2. Ridge тренировать на r2
3. Сделать полиномальные фичи
4. Бустинг
5. Своя стратегия cv
6. Зафиксировать random seed
7. построить модели для B_C2H6_mass и B_C3H8_mass. Модели B_iC4H10_mass	B_nC4H10_mass делать через корелляцию с B_C3H8_mass
8. Смещение по времени?
"""
