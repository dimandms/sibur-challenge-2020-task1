from constants import TARGET_COLUMNS
from args import parse_args

from load_data import load_data
from processing import process
from modelling import evaluate_training
from submission import create_submission
from evaluate import predict
from result_view import show_results

import matplotlib.pyplot as plt


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


def mean_of_lists(x): return [sum(values)/len(values)
                              for values in list(zip(*x))]


def stacking(simple_results, verbose):
    simple_estimators = []
    simple_fits = []
    simple_test_features = []
    simple_train_targets = []

    for estimators, fits, test_features, train_targets in simple_results:
        simple_estimators.append(estimators)
        simple_fits.append(fits)
        simple_test_features.append(test_features)
        simple_train_targets.append(train_targets)

    if verbose:
        _, axes = plt.subplots(4, 1, figsize=(15, 8))

        for target, ax, fits in zip(TARGET_COLUMNS, axes, mean_of_lists(simple_fits)):
            ax.plot(simple_train_targets[0]
                    [target].values, label=f"true_{target}")
            ax.plot(fits, label=f"pred_{target}")
            ax.legend()

        _, axes = plt.subplots(4, 1, figsize=(15, 8))
        for target, ax, fit in zip(TARGET_COLUMNS, axes, mean_of_lists(simple_fits)):
            errs = simple_train_targets[0][target].values - fit
            ax.hist(errs, bins=50)
            ax.legend()

    y_preds = [predict(estimators, test_features) for estimators,
               test_features in zip(simple_estimators, simple_test_features)]

    y_preds_mean = mean_of_lists(y_preds)

    return y_preds_mean, simple_test_features[0]


def main():
    args = parse_args()
    loaded_data = load_data()

    results = []
    # for shift_num in [184, 190]:
    for shift_num in [175, 185, 195]:
        result = simple_model_preds(shift_num, loaded_data, args.verbose)
        results.append(result)

    y_preds, test_features = stacking(results, args.verbose)

    plt.show()

    sub = create_submission(test_features.index.values, y_preds)
    sub.to_csv(f'submission.csv', index=False)


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
