import pickle
from args import parse_args

from load_data import load_data
from processing import process
from modelling import evaluate_training
from submission import create_submission, create_smoothed_submission
from evaluate import predict
from result_view import show_results


def main():
    args = parse_args()

    train_features, train_targets, test_features = process(load_data())
    models = evaluate_training(train_features, train_targets)
    if args.verbose:
        show_results(models)

    with open('results.pickle', 'wb') as f:
        pickle.dump(models, f)

    estimators = [model.best_estimator_ for model in models]
    y_preds = [pred for pred in predict(estimators, test_features)]

    sub = create_submission(test_features.index.values, y_preds)
    sub_smoothed = create_smoothed_submission(
        test_features.index.values, y_preds)

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
