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


def main():
    args = parse_args()
    loaded_data = load_data()

    basic_models_preds = []
    basic_models_fits = []
    estimators = []
    for shift_num in [184, 190]:
        train_features, train_targets, test_features = process(
            shift_num)(loaded_data)
        models = evaluate_training(train_features, train_targets)
        if args.verbose:
            show_results(models)

        for model in models:
            estimators.append(model.best_estimator_)

        y_preds = [pred for pred in predict(estimators, test_features)]
        basic_models_preds = y_preds

        y_fits = [pred for pred in predict(estimators, train_features)]
        basic_models_fits = y_fits

    print(len(basic_models_fits))

    # -----------------------
    train_features, train_targets, test_features = process(184)(loaded_data)
    X_train_tree = pd.concat([pd.DataFrame(fit) for fit in basic_models_fits] +
                             [train_features["A_rate"].reset_index(drop=True)], axis=1)
    trees = []
    for target in TARGET_COLUMNS:
        dtree = DecisionTreeRegressor()
        dtree.fit(X_train_tree, train_targets[target])
        trees.append(dtree)

    if args.verbose:
        print("\n==================== Train results (tree) ====================")
        scores = [mean_absolute_percentage_error(train_targets[target], tree.predict(
            X_train_tree)) * -1 for tree, target in zip(trees, TARGET_COLUMNS)]
        print(f"scores: {scores}")
        print(f"total score: {np.mean(scores)}")
        print("========================= End =========================")
    # -----------------------

    # -----
    _, axes = plt.subplots(4, 1, figsize=(15, 8))
    for target, ax, pred in zip(TARGET_COLUMNS, axes, [pred for pred in predict(trees, X_train_tree)]):
        ax.plot(train_targets[target].values, label=f"true_{target}")
        ax.plot(pred, label=f"pred_{target}")
        ax.legend()
    # -----
    plt.show()

    # sub = create_submission(test_features.index.values, y_preds)
    # sub_smoothed = create_smoothed_submission(
    #     test_features.index.values, y_preds)

    # sub.to_csv(f'submission.csv', index=False)
    # sub_smoothed.to_csv(f'submission_smoothed.csv', index=False)


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
