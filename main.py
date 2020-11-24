import matplotlib.pyplot as plt

from constants import FEATURE_COLUMNS

from load_data import load_data
from nan_processsing import process_na
from modelling import evaluate_training
from submission import create_submission


def main():
    data = load_data()
    X_train, y_train, X_test = process_na(data)
    models = evaluate_training(
        X_train, y_train, verbose=True, show_fit_plots=True)
    plt.show()

    y_preds = []
    for _, model in models.items():
        y_pred = model.predict(X_test[FEATURE_COLUMNS])
        y_preds.append(y_pred)

    sub = create_submission(X_test["timestamp"], y_preds)
    sub.to_csv(f'submission.csv', index=False)


if __name__ == "__main__":
    main()

"""1. Сделать преобразование в т/с и отбросить общие расходы. Потом восстанавливать проценты
2. Ridge тренировать на r2
3. Сделать полиномальные фичи
4. Бустинг
5. Своя стратегия cv
6. Зафиксировать random seed"""
