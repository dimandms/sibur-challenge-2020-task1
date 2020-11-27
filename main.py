import matplotlib.pyplot as plt

from constants import FEATURE_GASES_MASS

from load_data import load_data
from processing import process, smooth_series
from modelling import evaluate_training
from submission import create_submission
from evaluate import predict
from plotting import plot_submition


def main():
    train_features, train_targets, test_features = load_data()

    X_train, y_train, X_test = process(
        (train_features, train_targets, test_features))

    test_B_rate_smoothed = smooth_series(test_features['B_rate']).values

    models = evaluate_training(X_train[FEATURE_GASES_MASS], y_train)

    y_preds = [pred/test_B_rate_smoothed *
               100 for pred in predict(models, X_test[FEATURE_GASES_MASS])]

    sub = create_submission(test_features["timestamp"], y_preds)
    plot_submition(sub)

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
